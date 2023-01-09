# %% Single distributions

from polykin.distributions.baseclasses import IndividualDistributionP1
from polykin.distributions.baseclasses import IndividualDistributionP2

from math import exp, log, sqrt
import numpy as np
import scipy.special as sp
import scipy.stats as st


class Flory(IndividualDistributionP1):
    r"""Flory-Schulz (aka most-probable) chain-length distribution, with
    _number_ probability mass function given by:

    $$ p(k) = (1-a)a^{k-1} $$

    where $a=1-1/DP_n$. Mathematically speaking, this is a [geometric
    distribution](https://en.wikipedia.org/wiki/Geometric_distribution).
    """

    def _compute_parameters(self):
        self._a = 1 - 1 / self.DPn

    def _moment(self, order):
        a = self._a
        if order == 0:
            result = 1
        elif order == 1:
            result = 1/(1 - a)
        elif order == 2:
            result = 2/(1 - a)**2 - 1/(1 - a)
        elif order == 3:
            result = 6/(1 - a)**3 - 6/(1 - a)**2 + 1/(1 - a)
        else:
            raise ValueError("Not defined for order>3.")
        return result

    def _pdf(self, k):
        a = self._a
        return (1 - a) * a ** (k - 1)

    def _cdf(self, k, order):
        a = self._a
        if order == 0:
            result = 1 - a**k
        elif order == 1:
            result = (a**k*(-a*k + k + 1) - 1)/(a - 1)
        elif order == 2:
            result = (-((a - 1)**2*k**2 - 2*(a - 1)*k + a + 1)
                      * a**k + a + 1)/(a - 1)**2
        else:
            raise ValueError("Not defined for order>2.")
        return result/self._moment(order)

    def _xrange_auto(self):
        return (1, 10*self.DPn)

    def _random(self, size):
        a = self._a
        return self._rng.geometric((1-a), size)  # type: ignore


class Poisson(IndividualDistributionP1):
    r"""Poisson chain-length distribution, with _number_ probability mass
    function given by:

    $$ p(k) = \frac{a^{k-1} e^{-a}}{\Gamma(k)} $$

    where $a=DP_n-1$.
    """

    def _compute_parameters(self):
        self._a = self.DPn - 1

    def _moment(self, order):
        a = self._a
        if order == 0:
            result = 1
        elif order == 1:
            result = a + 1
        elif order == 2:
            result = a**2 + 3*a + 1
        elif order == 3:
            result = a**3 + 6*a**2 + 7*a + 1
        else:
            raise ValueError("Not defined for order>3.")
        return result

    def _pdf(self, k):
        a = self._a
        return np.exp((k - 1)*np.log(a) - a - sp.gammaln(k))

    def _cdf(self, k, order):
        a = self._a
        if order == 0:
            result = sp.gammaincc(k, a)
        elif order == 1:
            result = (a + 1)*sp.gammaincc(k, a) - \
                np.exp(k*log(a) - a - sp.gammaln(k))
        elif order == 2:
            result = (a*(a + 3) + 1)*sp.gammaincc(k, a) - \
                np.exp(k*log(a) + np.log(a + k + 2) - a - sp.gammaln(k))
        else:
            raise ValueError("Not defined for order>2.")
        return result/self._moment(order)

    def _xrange_auto(self):
        return (max(1, self.DPn/2 - 10), 1.5*self.DPn + 10)

    def _random(self, size):
        a = self._a
        return self._rng.poisson(a, size) + 1  # type: ignore


class LogNormal(IndividualDistributionP2):
    r"""Log-normal chain-length distribution, with _number_ probability density
    function given by:

    $$ p(x) = \frac{1}{x \sigma \sqrt{2 \pi}}
    \exp\left (- \frac{(\ln{x}-\mu)^2}{2\sigma^2} \right ) $$

    where $\mu = \ln{(DP_n/\sqrt{PDI})}$ and $\sigma=\sqrt{\ln(PDI)}$.
    """
    # https://reference.wolfram.com/language/ref/LogNormalDistribution.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html

    def _compute_parameters(self):
        try:
            PDI = self.PDI
            DPn = self.DPn
            self._sigma = sqrt(log(PDI))
            self._mu = log(DPn/sqrt(PDI))
        except AttributeError:
            pass

    def _moment(self, order: int):
        mu = self._mu
        sigma = self._sigma
        return exp(order*mu + 0.5*order**2*sigma**2)

    def _pdf(self, x):
        mu = self._mu
        sigma = self._sigma
        return st.lognorm.pdf(x, s=sigma, scale=exp(mu))

    def _cdf(self, x, order):
        mu = self._mu
        sigma = self._sigma
        if order == 0:
            result = st.lognorm.cdf(x, s=sigma, scale=exp(mu))
        elif order == 1:
            result = sp.erfc((mu + sigma**2 - np.log(x))/(sigma*sqrt(2)))/2
        elif order == 2:
            result = sp.erfc((mu + 2*sigma**2 - np.log(x))/(sigma*sqrt(2)))/2
        else:
            raise ValueError("Not defined for order>2.")
        return result

    def _xrange_auto(self):
        return (1, 100*self.DPn)

    def _random(self, size):
        mu = self._mu
        sigma = self._sigma
        return np.rint(self._rng.lognormal(mu, sigma, size))  # type: ignore


class SchulzZimm(IndividualDistributionP2):
    r"""Schulz-Zimm chain-length distribution, with _number_ probability
    density function given by:

    $$ p(x) = \frac{x^{k-1} e^{-x/\theta}}{\Gamma(k) \theta^k} $$

    where $k = 1/(DP_n-1)$ and $\theta = DP_n(PDI-1)$. Mathematically speaking,
    this is a [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution).
    """
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
    # https://goldbook.iupac.org/terms/view/S05502

    def _compute_parameters(self):
        try:
            PDI = self.PDI
            DPn = self.DPn
            self._k = 1/(PDI - 1)
            self._theta = DPn*(PDI - 1)
        except AttributeError:
            pass

    def _moment(self, order: int):
        k = self._k
        theta = self._theta
        return sp.poch(k, order)*theta**order

    def _pdf(self, x):
        k = self._k
        theta = self._theta
        return st.gamma.pdf(x, a=k, scale=theta)

    def _cdf(self, x, order):
        k = self._k
        theta = self._theta
        if order == 0:
            result = st.gamma.cdf(x, a=k, scale=theta)
        elif order == 1:
            result = 1 - sp.gammaincc(1+k, x/theta)
        elif order == 2:
            result = 1 - sp.gammaincc(2+k, x/theta)
        else:
            raise ValueError("Not defined for order>2.")
        return result

    def _xrange_auto(self):
        return (1, 10*self.DPn)

    def _random(self, size):
        k = self._k
        theta = self._theta
        return np.rint(self._rng.gamma(k, theta, size))  # type: ignore
