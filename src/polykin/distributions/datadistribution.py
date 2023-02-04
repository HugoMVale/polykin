# %% Data distribution

from polykin.utils import vectorize, check_subclass
from polykin.distributions import Flory, Poisson, LogNormal, SchulzZimm
from polykin.distributions.baseclasses import \
    IndividualDistribution, AnalyticalDistribution, \
    AnalyticalDistributionP1, AnalyticalDistributionP2

import numpy as np
from numpy import ndarray, dtype, float64
from scipy.optimize import curve_fit
from scipy import interpolate, integrate
from typing import Any, Literal, Union
import functools


class DataDistribution(IndividualDistribution):
    """Arbitrary numerical chain-length distribution, defined by chain size
    and pdf data.
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
        xrange = (max(xa, self._length_data[0]), xb)
        if order == self._pdf_order:
            result = self._pdf_spline.integral(*xrange)
        else:
            result, _ = integrate.quad(
                lambda x: x**(order-self._pdf_order)*self._pdf_spline(x),
                *xrange)
        return result

    def _pdf0_length(self, x):
        return x**(-self._pdf_order)*self._pdf_spline(x)

    @functools.cached_property
    def _range_length_default(self):
        return self._length_data[(0, -1),]

    def fit(self,
            dist_class: Union[type[Flory], type[Poisson], type[LogNormal],
                              type[SchulzZimm]]
            ) -> AnalyticalDistribution:

        check_subclass(dist_class, AnalyticalDistribution, 'dist_class')
        isP1 = issubclass(dist_class, AnalyticalDistributionP1)

        # Init fit distribution
        if isP1:
            args = (self.DPn,)
        else:
            args = (self.DPn, self.PDI)
        dfit = dist_class(*args, M0=self.M0, name=self.name+"-fit")

        # Define parametric function
        def f(x, *p):
            dfit.DPn = p[0]
            if not isP1:
                dfit.PDI = p[1]
            return dfit._cdf(x, 1, False)

        # Call fit method
        xdata = self._length_data
        ydata = self._cdf_length(xdata, 1)
        solution = curve_fit(f,
                             xdata=xdata,
                             ydata=ydata,
                             p0=dfit._pvalues,
                             bounds=dfit._pbounds,
                             method='trf',
                             full_output=True)

        if solution[4] > 0:
            # print(dfit)
            print(f"DPn:    {dfit.DPn:.1f}")
            print(f"PDI:    {dfit.PDI:.2f}")
        else:
            print("Failed to fit distribution: ", solution[3])

        return dfit
