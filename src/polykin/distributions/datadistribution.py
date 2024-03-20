# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import functools
from typing import Literal, Optional, Union

import numpy as np
from numpy import log10
from scipy import integrate, interpolate, optimize

from polykin.utils.math import vectorize
from polykin.utils.tools import check_bounds, check_subclass
from polykin.utils.types import FloatVectorLike

from .analyticaldistributions import Flory, LogNormal, Poisson, SchulzZimm
from .base import (AnalyticalDistribution, AnalyticalDistributionP2,
                   IndividualDistribution, MixtureDistribution)

__all__ = ['DataDistribution']


class DataDistribution(IndividualDistribution):
    r"""Arbitrary numerical chain-length distribution, defined by chain size
    and pdf data.

    Parameters
    ----------
    size_data : FloatVectorLike
        Chain length or molar mass data.
    pdf_data : FloatVectorLike
        Distribution data.
    kind : Literal['number', 'mass', 'gpc']
        Kind of distribution.
    sizeasmass : bool
        Switch size input between chain-length (if `False`) or molar
        mass (if `True`).
    M0 : float
        Molar mass of the repeating unit, $M_0$. Unit = kg/mol.
    name : str
        Name.
    """
    _continuous = True

    def __init__(self,
                 size_data: FloatVectorLike,
                 pdf_data: FloatVectorLike,
                 kind: Literal['number', 'mass', 'gpc'] = 'mass',
                 sizeasmass: bool = False,
                 M0: float = 0.1,
                 name: str = ''
                 ) -> None:

        # Check and clean input
        self.M0 = M0
        self.name = name
        size_data = np.array(size_data)
        pdf_data = np.array(pdf_data)
        if self._verify_sizeasmass(sizeasmass):
            size_data /= self.M0
        idx_valid = np.logical_and(pdf_data > 0., size_data >= 1.)
        if not idx_valid.all():
            print("Warning: Found and removed inconsistent values.")
        self._pdf_data = pdf_data[idx_valid]
        self._length_data = size_data[idx_valid]
        self._pdf_order = self.kind_order[self._verify_kind(kind)]

        # Base-line correction
        # y = self._pdf_data
        # x = self._length_data
        # baseline = y[0] + (y[-1] - y[0])/(np.log(x[-1]/x[0]))*np.log(x/x[0])
        # self._pdf_data -= baseline

        # Compute spline
        self._pdf_spline = \
            interpolate.UnivariateSpline(self._length_data,
                                         self._pdf_data,
                                         k=3,
                                         s=0,
                                         ext=1)

    @vectorize
    def _moment_quadrature(self,
                           xa: float,
                           xb: float,
                           order: int
                           ) -> float:
        xrange = (max(xa, self._length_data[0]),
                  min(xb, self._length_data[-1]))
        if order == self._pdf_order:
            result = self._pdf_spline.integral(*xrange)
        else:
            result, _ = integrate.quad(
                lambda x: x**(order-self._pdf_order)*self._pdf_spline(x),
                *xrange, limit=50)
        return result

    def _pdf0_length(self, x):
        return x**(-self._pdf_order)*self._pdf_spline(x)

    @functools.cached_property
    def _range_length_default(self):
        return self._length_data[(0, -1), ]

    def fit(self,
            dist_class: Union[type[Flory], type[Poisson], type[LogNormal],
                              type[SchulzZimm]],
            dim: int = 1,
            display_table: bool = True
            ) -> Optional[Union[AnalyticalDistribution, MixtureDistribution]]:
        """Fit (deconvolute) a `DataDistribution` into a linear combination of
        `AnalyticalDistribution`(s).

        Parameters
        ----------
        dist_class : type[Flory] | type[Poisson] | type[LogNormal] | type[SchulzZimm]
            Type of distribution to be used in the fit.
        dim : int
            Number of individual components to use in the fit.
        display_table : bool
            Option to display results table with information about individual
            components.

        Returns
        -------
        AnalyticalDistribution | MixtureDistribution | None
            If fit successful, it returns the fitted distribution.
        """

        check_subclass(dist_class, AnalyticalDistribution, 'dist_class')
        isP2 = issubclass(dist_class, AnalyticalDistributionP2)
        check_bounds(dim, 1, 10, 'dim')

        # Init fit distribution
        dfit = MixtureDistribution({})
        weight = np.full(dim, 1/dim)
        DPn = np.empty_like(weight)
        if isP2:
            PDI = np.full(dim, 2.0)
        for i in range(dim):
            DPn[i] = self._length_data[0] * \
                (self._length_data[-1]/self._length_data[0])**((i+1)/(dim+1))
            if isP2:
                args = (DPn[i], PDI[i])
            else:
                args = (DPn[i],)
            d = dist_class(*args, M0=self.M0)
            dfit = dfit + weight[i]*d

        # Define objective function
        xdata = self._length_data
        ydata = self._cdf_length(xdata, 1)

        def objective_fun(x):
            # Assign values
            for i, d in enumerate(dfit.components.keys()):
                dfit.components[d] = x[i]
                d.DPn = 10**x[dim+i]
                if isP2:
                    d.PDI = x[2*dim+i]
            yfit = dfit._cdf(xdata, 1, False)
            return np.sum((yfit - ydata)**2)/ydata.size

        # Initial guess and bounds
        # We do a log transform on DPn to normalize the changes
        x0 = np.concatenate([weight, log10(DPn)])
        bounds = [(0, 1) for _ in range(dim)] + \
            [(log10(3), log10(xdata[-1])) for _ in range(dim)]
        if isP2:
            x0 = np.concatenate([x0, PDI])
            bounds += [(1.01, 5.) for _ in range(dim)]

        # Equality constraint: w(1) + .. + w(N) = 1
        A = np.zeros(x0.size)
        A[:dim] = 1
        constraint = optimize.LinearConstraint(A, 1, 1)
        constraints = []
        constraints.append(constraint)

        # Inequality constraints: DPn(i) - DPn(i+1) <= 0
        for i in range(dim-1):
            A = np.zeros(x0.size)
            A[dim+i] = 1
            A[dim+i+1] = -1
            constraint = optimize.LinearConstraint(A, -np.inf, 0)
            constraints.append(constraint)

        # Call fit method
        # Tried all methods and 'trust-constr' is the most robust
        solution = optimize.minimize(objective_fun,
                                     x0=x0,
                                     method='trust-constr',
                                     bounds=bounds,
                                     constraints=constraints,
                                     options={'verbose': 0})
        if solution.success:
            if display_table:
                print(dfit.components_table)
            if dim == 1:
                dfit = next(iter(dfit.components))
            dfit.name = self.name + "-fit"
            result = dfit
        else:
            print("Failed to fit distribution: ", solution.message)
            result = None

        return result  # type : ignore
