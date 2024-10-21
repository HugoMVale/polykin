# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import functools
from math import factorial
from typing import Union

import numpy as np
import scipy.special as sp
import scipy.stats as st
from numpy import exp, log, pi, sqrt
from scipy.optimize import root_scalar

from polykin.distributions.base import (AnalyticalDistributionP1,
                                        AnalyticalDistributionP2)
from polykin.utils.math import eps
from polykin.utils.types import FloatArray, IntArrayLike

__all__ = ['Flory',
           'Poisson',
           'LogNormal',
           'SchulzZimm',
           'WeibullNycanderGold_pdf']


class Flory(AnalyticalDistributionP1):
    r"""Flory-Schulz (aka most-probable) chain-length distribution.

    This distribution is based on the following number probability density
    function:

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


    Examples
    --------
    Define a Flory distribution and evaluate the corresponding probability
    density function and cumulative distribution function for representative
    chain lengths.

    >>> from polykin.distributions import Flory
    >>> a = Flory(100, M0=0.050, name='A')
    >>> a
    type: Flory
    name: A
    DPn:  100.0
    DPw:  199.0
    DPz:  298.5
    PDI:  1.99
    M0:   0.050 kg/mol
    Mn:   5.000 kg/mol
    Mw:   9.950 kg/mol
    Mz:   14.925 kg/mol

    >>> a.pdf(a.DPn)
    0.003697296376497271

    >>> a.cdf([a.DPn, a.DPw, a.DPz])
    array([0.26793532, 0.59535432, 0.80159978])

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

    This distribution is based on the following number probability density
    function:

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

    Examples
    --------
    Define a Poisson distribution and evaluate the corresponding probability
    density function and cumulative distribution function for representative
    chain lengths.

    >>> from polykin.distributions import Poisson
    >>> a = Poisson(100, M0=0.050, name='A')
    >>> a
    type: Poisson
    name: A
    DPn:  100.0
    DPw:  101.0
    DPz:  102.0
    PDI:  1.01
    M0:   0.050 kg/mol
    Mn:   5.000 kg/mol
    Mw:   5.050 kg/mol
    Mz:   5.099 kg/mol

    >>> a.pdf(a.DPn)
    0.04006147193133002

    >>> a.cdf([a.DPn, a.DPw, a.DPz])
    array([0.48703481, 0.52669305, 0.56558077])

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
        return poisson(k, a, False)

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
                exp(k*log(a) - a - sp.gammaln(k))
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

    This distribution is based on the following number probability density
    function:

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

    Examples
    --------
    Define a log-normal distribution and evaluate the corresponding probability
    density function and cumulative distribution function for representative
    chain lengths.

    >>> from polykin.distributions import LogNormal
    >>> a = LogNormal(100, PDI=3., M0=0.050, name='A')
    >>> a
    type: LogNormal
    name: A
    DPn:  100.0
    DPw:  300.0
    DPz:  900.0
    PDI:  3.00
    M0:   0.050 kg/mol
    Mn:   5.000 kg/mol
    Mw:   15.000 kg/mol
    Mz:   45.000 kg/mol

    >>> a.pdf(a.DPn)
    0.003317780747597256

    >>> a.cdf([a.DPn, a.DPw, a.DPz])
    array([0.3001137, 0.6998863, 0.9420503])

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
        return exp(-(log(x) - mu)**2/(2*sigma**2))/(x*sigma*sqrt(2*pi))

    @functools.cache
    def _moment_length(self, order):
        mu = self._mu
        sigma = self._sigma
        return exp(order*mu + 0.5*order**2*sigma**2)

    def _cdf_length(self, x, order):
        mu = self._mu
        sigma = self._sigma
        if order == 0:
            result = (1 + sp.erf((log(x) - mu)/(sigma*sqrt(2))))/2
        elif order == 1:
            result = sp.erfc((mu + sigma**2 - log(x))/(sigma*sqrt(2)))/2
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

    This distribution is based on the following number probability density
    function:

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

    Examples
    --------
    Define a Schulz-Zimm distribution and evaluate the corresponding
    probability density function and cumulative distribution function for
    representative chain lengths.

    >>> from polykin.distributions import SchulzZimm
    >>> a = SchulzZimm(100, PDI=3., M0=0.050, name='A')
    >>> a
    type: SchulzZimm
    name: A
    DPn:  100.0
    DPw:  300.0
    DPz:  500.0
    PDI:  3.00
    M0:   0.050 kg/mol
    Mn:   5.000 kg/mol
    Mw:   15.000 kg/mol
    Mz:   25.000 kg/mol

    >>> a.pdf(a.DPn)
    0.0024197072451914337

    >>> a.cdf([a.DPn, a.DPw, a.DPz])
    array([0.19874804, 0.60837482, 0.82820286])

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
        return x**(k-1)*exp(-x/theta)/(theta**k*sp.gamma(k))

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

# %% Gold distribution


def poisson(k, a: float, k0: bool):
    shift = 0
    if not k0:
        shift = -1
    return exp((k + shift)*log(a) - a - sp.gammaln(k + 1 + shift))


def WeibullNycanderGold_pdf(k: Union[int, IntArrayLike],
                            v: float,
                            r: float
                            ) -> Union[float, FloatArray]:
    r"""Weibull, Nycander and Golds's analytical chain-length distribution for
    living polymerization with different initiation and polymerization rate
    coefficients.

    For a living polymerization with only initiation and propagation (i.e.,
    constant number of chains), the number fraction of chains of length
    $k$ can computed in two steps. First, the number fraction of unreacted 
    initiator molecules, $p_0=p(0)$, is found by solving the equation:

    $$ v + r \ln(p_0) + (r - 1)(1 - p_0) = 0 $$

    where $v$ denotes the number-average degree of polymerization of all chains,
    including unreacted initiator molecules, and $r=k_p/k_i$ is the ratio of the
    polymerization and initiation rate coefficients. Then, the number fraction
    of chains with $k \ge 1$ monomer units can be evaluated by:

    \begin{aligned}
    p(k) & = p_0 \frac{r^{k-1}}{(r-1)^k} \left[1- \Gamma(k,(1-r)\ln{p_0}) \right] & \text{if } r>1 \\
    p(k) & = p_0 \frac{r^{k-1}}{(1-r)^k} (-1)^k\left[1- \Gamma(k,(1-r)\ln{p_0}) \right] & \text{if } r<1
    \end{aligned}

    where $\Gamma$ is the regularized upper incomplete gamma function. For $r>1$,
    the argument of $\Gamma$ is always positive, while for $r<1$ it is negative. 
    This analytical solution has an obvious singularity at $r=1$; in that case,
    the solution reduces to the well-known Poisson distribution:

    $$ p(k) = \frac{v^k}{k!}e^{-v} $$

    valid for $k \ge 0$.

    !!! note

        * The solution is numerically unstable in certain domains, namely for
        $r$ close to 1, and also for $k>>v$. This is an intrinsic feature  of
        the equation.
        * For $|r-1|<10^{-2}$, the algorithm automatically switches to the Poisson
        distribution. Some numerical discontinuity at this boundary is to be
        expected. 
        * For $r<1$, no solution is currently computed, because the incomplete
        gamma function algorithm available in SciPy is restricted to positive
        arguments.    

    **References**

    * Weibull, B.; Nycander, E.. "The Distribution of Compounds Formed in the
    Reaction." Acta Chemica Scandinavica 49 (1995): 207-216.
    * Gold, L. "Statistics of polymer molecular size distribution for an
    invariant number of propagating chains." The Journal of Chemical Physics
    28.1 (1958): 91-99.

    Parameters
    ----------
    k : int | IntArrayLike
        Chain length (>=0).
    v : float
        Number-average degree of polymerization considering chains with zero
        length.
    r : float
        Ratio of propagation and initiation rate coefficients.

    Returns
    -------
    float | FloatArray
        Number probability density.

    Examples
    --------
    Compute the fraction of chains with lengths 0 to 2 for a system with $r=5$
    and $v=1$.

    >>> from polykin.distributions import WeibullNycanderGold_pdf
    >>> WeibullNycanderGold_pdf([0, 1, 2], 1., 5)
    array([0.58958989, 0.1295864 , 0.11493254])
    """

    if isinstance(k, (list, tuple)):
        k = np.asarray(k)

    # Special case where it reduces to Poisson(kmin=0)
    if abs(r - 1.) < 1e-2:
        return poisson(k, v, True)

    # Find p0=p(k=0)
    def find_p0(p0):
        return v + r*log(p0) + (r - 1.)*(1. - p0)

    sol = root_scalar(find_p0, method='brentq', bracket=(0. + eps, 1. - eps))
    p0 = sol.root

    # Avoid issue 'Integers to negative integer powers'
    r = float(r)

    # Weibull-Nycander-Gold solution p(k>=1)
    if r > 1.:
        A = sp.gammaincc(k, (1. - r)*log(p0))
        result = p0 * r**(k - 1) / (r - 1.)**k * (1. - A)
    else:
        # gammaincc in scipy is restricted to positive arguments
        # A = sp.gammaincc(k, (1. - r)*log(p0))
        # result = p0 * r**(k - 1) / (1. - r)**k * (-1)**k * (1. - A)
        result = NotImplemented

    # Overwite p(k=0)
    result[k == 0] = p0

    return result
