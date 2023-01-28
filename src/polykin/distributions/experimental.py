# %% Experimental distribution

from polykin.utils import vectorize
from polykin.distributions.baseclasses import \
    IndividualDistribution, AnalyticalDistribution, \
    AnalyticalDistributionP1, AnalyticalDistributionP2

import numpy as np
from numpy import ndarray, dtype, float64
from scipy.optimize import curve_fit
from scipy import interpolate, integrate
from typing import Any, Literal, Union
import functools


class ExperimentalDistribution(IndividualDistribution):
    """Arbitrary experimental chain-length distribution, defined by chain size
    and corresponding pdf values.
    """
    _continuous = True

    def __init__(self,
                 size_data: Union[list[float], ndarray[Any, dtype[float64]]],
                 pdf_data: Union[list[float], ndarray[Any, dtype[float64]]],
                 kind: Literal['number', 'mass', 'gpc'] = 'mass',
                 sizeasmass: bool = False,
                 M0: float = 100,
                 name: str = ''
                 ) -> None:

        # Check and clean input
        self.M0 = M0
        self.name = name
        size_data = np.asarray(size_data)
        pdf_data = np.asarray(pdf_data)
        idx_valid = pdf_data > 0
        if not idx_valid.all():
            print("Warning: Found and removed `pdf_data` values <=0.")
        self._pdf_data = pdf_data[idx_valid]
        self._length_data = size_data[idx_valid]
        if self._verify_sizeasmass(sizeasmass):
            self._length_data /= self.M0
        self._pdf_order = self.kindnames[self._verify_kind(kind)]

        # Compute spline
        self.pdf_spline = \
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
        xrange = (self._length_data[0], xb)
        if order == self._pdf_order:
            result = self.pdf_spline.integral(*xrange)
        else:
            result, _ = integrate.quad(
                lambda x: x**(order-self._pdf_order)*self.pdf_spline(x),
                *xrange)
        return result

    def _pdf0_length(self, x):
        return x**(-self._pdf_order)*self.pdf_spline(x)

    @functools.cached_property
    def _range_length_default(self):
        return self._length_data[(0, -1),]

    def fit(self,
            dist: type[AnalyticalDistribution]
            ) -> AnalyticalDistribution:

        if isinstance(dist, AnalyticalDistributionP1):
            isP2 = False
        elif isinstance(dist, AnalyticalDistributionP2):
            isP2 = True
        else:
            raise TypeError("Invalid `dist` type.")

        # Define parametric function
        def f(x, *p):
            dist.DPn = p[1]
            if isP2:
                dist.PDI = p[2]
            return np.log(p[0]*dist.pdf(x, kind=self.pdf_type,
                                        sizeasmass=self.sizeasmass))

        # Call fit method
        p0 = (1,) + dist._pvalues
        bounds = ((-np.Inf,)+dist._pbounds[0], (np.Inf,)+dist._pbounds[1])
        solution = curve_fit(f,
                             xdata=self.size_data,
                             ydata=np.log(self._pdf_data),
                             p0=p0,
                             bounds=bounds,
                             method='trf',
                             full_output=True)

        if solution[4] > 0:
            # print(dist)
            popt = solution[0]
            print(f"scale:  {popt[0]:.2e}")
            print(f"DPn:    {dist.DPn:.1f}")
            print(f"PDI:    {dist.PDI:.2f}")
        else:
            print("Failed to fit distribution: ", solution[3])

        return dist
