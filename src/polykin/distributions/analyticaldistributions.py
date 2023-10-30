# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from .base import AnalyticalDistributionP1, AnalyticalDistributionP2

from math import exp, log, sqrt, pi
import numpy as np
import scipy.special as sp
import scipy.stats as st
import functools

__all__ = ['Flory', 'Poisson', 'LogNormal', 'SchulzZimm']


class Flory(AnalyticalDistributionP1):
    r"""Flory-Schulz (aka most-probable) chain-length distribution.

    The Flory-Schulz _number_ probability mass function is given by:

    $$ p(k) = (1-a)a^{k-1} $$

    where $a=1-1/DP_n$. Mathematically speaking, this is a [geometric
    distribution](https://en.wikipedia.org/wiki/Geometric_distribution).

    Parameters
    ----------
    DPn : float
        Number-average degree of polymerization, $DP_n$.
    M0 : float
        Molar mass of the repeating unit, $M_0$. Unit = kg/mol.
    name : str
        Name
    """
    _continuous = False

    def __init__(self,
                 DPn: float,
                 M0: float = 0.1,
                 name: str = ''
                 ) -> None:

        super().__init__(DPn, M0, name)

    def _update_internal_parameters(self):
        super()._update_internal_parameters()
        self._a = 1 - 1/self.DPn

    def _pdf0_length(self, k):
        a = self._a
        return (1 - a) * a ** (k - 1)

    @functools.cache
    def _moment_length(self, order):
        a = self._a
        if order == 0:
            result = 1.0
        elif order == 1:
            result = 1/(1 - a)
        elif order == 2:
            result = 2/(1 - a)**2 - 1/(1 - a)
        elif order == 3:
            result = 6/(1 - a)**3 - 6/(1 - a)**2 + 1/(1 - a)
        else:
            raise ValueError("Not defined for order>3.")
        return result

    def _cdf_length(self, k, order):
        a = self._a
        if order == 0:
            result = 1 - a**k
        elif order == 1:
            result = (a**k*(-a*k + k + 1) - 1)/(a - 1)
        else:
            raise ValueError
        return result / self._moment_length(order)

    def _random_length(self, size):
        return self._rng.geometric((1-self._a), size)  # type: ignore

    @functools.cached_property
    def _range_length_default(self):
        return st.geom.ppf(self._ppf_bounds, p=(1-self._a))


class Poisson(AnalyticalDistributionP1):
    r"""Poisson chain-length distribution.

    The Poisson _number_ probability mass function is given by:

    $$ p(k) = \frac{a^{k-1} e^{-a}}{\Gamma(k)} $$

    where $a=DP_n-1$.

    Parameters
    ----------
    DPn : float
        Number-average degree of polymerization, $DP_n$.
    M0 : float
        Molar mass of the repeating unit, $M_0$. Unit = kg/mol.
    name : str
        Name
    """

    _continuous = False

    def __init__(self,
                 DPn: float,
                 M0: float = 0.1,
                 name: str = ''
                 ) -> None:

        super().__init__(DPn, M0, name)

    def _update_internal_parameters(self):
        super()._update_internal_parameters()
        self._a = self.DPn - 1

    def _pdf0_length(self, k):
        a = self._a
        return np.exp((k - 1)*np.log(a) - a - sp.gammaln(k))

    @functools.cache
    def _moment_length(self, order):
        a = self._a
        if order == 0:
            result = 1.0
        elif order == 1:
            result = a + 1
        elif order == 2:
            result = a**2 + 3*a + 1
        elif order == 3:
            result = a**3 + 6*a**2 + 7*a + 1
        else:
            raise ValueError("Not defined for order>3.")
        return result

    def _cdf_length(self, k, order):
        a = self._a
        if order == 0:
            result = sp.gammaincc(k, a)
        elif order == 1:
            result = (a + 1)*sp.gammaincc(k, a) - \
                np.exp(k*log(a) - a - sp.gammaln(k))
        else:
            raise ValueError
        return result/self._moment_length(order)

    def _random_length(self, size):
        return self._rng.poisson(self._a, size) + 1  # type: ignore

    @functools.cached_property
    def _range_length_default(self):
        return st.poisson.ppf(self._ppf_bounds, mu=self._a, loc=1)


class LogNormal(AnalyticalDistributionP2):
    r"""Log-normal chain-length distribution.

    The Log-normal _number_ probability density function is given by:

    $$ p(x) = \frac{1}{x \sigma \sqrt{2 \pi}}
    \exp\left (- \frac{(\ln{x}-\mu)^2}{2\sigma^2} \right ) $$

    where $\mu = \ln{(DP_n/\sqrt{PDI})}$ and $\sigma=\sqrt{\ln(PDI)}$.

    Parameters
    ----------
    DPn : float
        Number-average degree of polymerization, $DP_n$.
    PDI : float
        Polydispersity index, $PDI$.
    M0 : float
        Molar mass of the repeating unit, $M_0$. Unit = kg/mol.
    name : str
        Name.
    """
    # https://reference.wolfram.com/language/ref/LogNormalDistribution.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html

    _continuous = True

    def __init__(self,
                 DPn: float,
                 PDI: float,
                 M0: float = 0.1,
                 name: str = ''
                 ) -> None:

        super().__init__(DPn, PDI, M0, name)

    def _update_internal_parameters(self):
        super()._update_internal_parameters()
        try:
            PDI = self.PDI
            DPn = self.DPn
            self._sigma = sqrt(log(PDI))
            self._mu = log(DPn/sqrt(PDI))
        except AttributeError:
            pass

    def _pdf0_length(self, x):
        mu = self._mu
        sigma = self._sigma
        return np.exp(-(np.log(x) - mu)**2/(2*sigma**2))/(x*sigma*sqrt(2*pi))

    @functools.cache
    def _moment_length(self, order):
        mu = self._mu
        sigma = self._sigma
        return exp(order*mu + 0.5*order**2*sigma**2)

    def _cdf_length(self, x, order):
        mu = self._mu
        sigma = self._sigma
        if order == 0:
            result = (1 + sp.erf((np.log(x) - mu)/(sigma*sqrt(2))))/2
        elif order == 1:
            result = sp.erfc((mu + sigma**2 - np.log(x))/(sigma*sqrt(2)))/2
        else:
            raise ValueError
        return result

    def _random_length(self, size):
        mu = self._mu
        sigma = self._sigma
        return np.rint(self._rng.lognormal(mu, sigma, size))  # type: ignore

    @functools.cached_property
    def _range_length_default(self):
        mu = self._mu
        sigma = self._sigma
        return st.lognorm.ppf(self._ppf_bounds, s=sigma, scale=exp(mu), loc=1)


class SchulzZimm(AnalyticalDistributionP2):
    r"""Schulz-Zimm chain-length distribution.

    The Schulz-Zimm _number_ probability density function is given by:

    $$ p(x) = \frac{x^{k-1} e^{-x/\theta}}{\Gamma(k) \theta^k} $$

    where $k = 1/(DP_n-1)$ and $\theta = DP_n(PDI-1)$. Mathematically speaking,
    this is a
    [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution).

    Parameters
    ----------
    DPn : float
        Number-average degree of polymerization, $DP_n$.
    PDI : float
        Polydispersity index, $PDI$.
    M0 : float
        Molar mass of the repeating unit, $M_0$. Unit = kg/mol.
    name : str
        Name.
    """
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
    # https://goldbook.iupac.org/terms/view/S05502

    _continuous = True

    def __init__(self,
                 DPn: float,
                 PDI: float,
                 M0: float = 0.1,
                 name: str = ''
                 ) -> None:

        super().__init__(DPn, PDI, M0, name)

    def _update_internal_parameters(self):
        super()._update_internal_parameters()
        try:
            PDI = self.PDI
            DPn = self.DPn
            self._k = 1/(PDI - 1)
            self._theta = DPn*(PDI - 1)
        except AttributeError:
            pass

    def _pdf0_length(self, x):
        k = self._k
        theta = self._theta
        return x**(k-1)*np.exp(-x/theta)/(theta**k*sp.gamma(k))

    @functools.cache
    def _moment_length(self, order):
        k = self._k
        theta = self._theta
        return sp.poch(k, order)*theta**order

    def _cdf_length(self, x, order):
        k = self._k
        theta = self._theta
        if order == 0:
            result = sp.gammainc(k, x/theta)
        elif order == 1:
            result = 1 - sp.gammaincc(1+k, x/theta)
        else:
            raise ValueError
        return result

    def _random_length(self, size):
        k = self._k
        theta = self._theta
        return np.rint(self._rng.gamma(k, theta, size))  # type: ignore

    @functools.cached_property
    def _range_length_default(self):
        k = self._k
        theta = self._theta
        return st.gamma.ppf(self._ppf_bounds, a=k, scale=theta, loc=1)
