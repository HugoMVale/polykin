# %%

from base import Base
from utils import check_bounds, check_type, check_in_set

from math import exp, log, sqrt
import numpy as np
from numpy.typing import ArrayLike
import scipy.special as sp
import scipy.stats as st
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Distribution(Base, ABC):
    """Abstract class for all chain-length distributions."""

    distnames = {'number': 0, 'mass': 1, 'gpc': 2}

    @property
    @abstractmethod
    def M0(self) -> float:
        r"""Molar mass of the repeating unit, $M_0$."""
        return 0.0

    @property
    @abstractmethod
    def DPn(self) -> float:
        r"""Number-average degree of polymerization, $DP_n$."""
        return 0.0

    @abstractmethod
    def moment(self, order: int) -> float:
        r"""$m$-th moment of the number chain-length distribution,
        $\lambda_m$."""
        return 0.0

    @property
    def DPw(self) -> float:
        r"""Mass-average degree of polymerization, $DP_w$."""
        return self.moment(2) / self.moment(1)

    @property
    def DPz(self) -> float:
        r"""z-average degree of polymerization, $DP_z$."""
        return self.moment(3) / self.moment(2)

    @property
    def PDI(self) -> float:
        r"""Polydispersity index, $M_w/M_n$."""
        return self.Mw / self.Mn

    @property
    def Mn(self) -> float:
        r"""Number-average molar mass, $M_n$."""
        return self.M0 * self.DPn

    @property
    def Mw(self) -> float:
        r"""Weight-average molar mass, $M_w$."""
        return self.M0 * self.DPw

    @property
    def Mz(self) -> float:
        r"""z-average molar mass, $M_z$."""
        return self.M0 * self.DPz


class BaseDistribution(Distribution):
    """Abstract class for all base chain-length distributions."""

    def __init__(self, DPn: int, M0: float = 100.0, name: str = ""):
        """Initialize chain-length distribution.

        Parameters
        ----------
        DPn : int
            Number-average degree of polymerization, $DP_n$.
        M0 : float
            Molar mass of the repeating unit, $M_0$.
        name : str
            Name
        """
        self.DPn = DPn
        self.M0 = M0
        self.name = name
        self._rng = None

    def __str__(self) -> str:
        return f"name: {self.name}\n" + \
            f"DPn:  {self.DPn:.1f}\n" + \
            f"DPw:  {self.DPw:.1f}\n" + \
            f"DPz:  {self.DPz:.1f}\n" + \
            f"PDI:  {self.PDI:.2f}\n" + \
            f"M0:   {self.M0:,.1f}\n" + \
            f"Mn:   {self.Mn:,.0f}\n" + \
            f"Mw:   {self.Mw:,.0f}\n" + \
            f"Mz:   {self.Mz:,.0f}"

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return CombinedDistribution([self], [other], name=self.name)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, BaseDistribution):
            return CombinedDistribution([self, other], [1, 1],
                                        name=self.name+'+'+other.name)
        else:
            return NotImplemented

    @property
    def M0(self) -> float:
        r"""Molar mass of the repeating unit, $M_0$."""
        return self.__M0

    @M0.setter
    def M0(self, M0: float):
        self.__M0 = check_bounds(M0, 0.0, np.Inf, "M0")

    @property
    def DPn(self) -> float:
        r"""Number-average degree of polymerization, $DP_n$."""
        return self.__DPn

    @DPn.setter
    def DPn(self, DPn: int = 100):
        self.__DPn = check_bounds(DPn, 2, np.Inf, "DPn")
        self._compute_parameters()

    def _compute_parameters(self):
        pass

    def moment(self, order: int) -> float:
        r"""$m$-th moment of the number chain-length distribution:

        $$ \lambda_m=\sum_{k=1}^{\infty }k^m\,p(k) $$

        or 

        $$ \lambda_m=\int_{0}^{\infty }x^m\,p(x)\mathrm{d}x $$

        Parameters
        ----------
        order : int
            Order of the moment, $0 \le m \le 3$.

        Returns
        -------
        float
            Moment, $\lambda_m$
        """

        check_bounds(order, 0, 3, 'order')
        return self._moment(order)

    def _process_size(self, size, unit_size) -> np.ndarray:
        if isinstance(size, list):
            size = np.asarray(size)
        check_in_set(unit_size, {'chain_length', 'molar_mass'}, 'unit_size')
        if unit_size == "molar_mass":
            size = size/self.M0
        return size

    def pdf(self, size: int | float | ArrayLike, dist: str = "mass",
            unit_size: str = "chain_length") -> float | np.ndarray:
        r"""Evaluate the probability density function, $p(k)$.

        Parameters
        ----------
        size : int | float | ArrayLike
            Chain length or molar mass.
        dist : str
            Type of distribution. Options: `'number'`, `'mass'`, `'gpc'`.
        unit_size : str
            Unit of variable `size`. Options: `'chain_length'` or
            `'molar_mass'`.

        Returns
        -------
        float | np.ndarray
            Chain-length probability.
        """

        # Convert size, if required
        size = self._process_size(size, unit_size)

        # Compute distribution
        check_in_set(dist, set(self.distnames.keys()), 'dist')
        order = self.distnames[dist]
        return self._pdf(size) * size**order / self._moment(order)

    def cdf(self, size: int | float | ArrayLike, dist: str = "mass",
            unit_size: str = "chain_length") -> float | np.ndarray:
        r"""Evaluate the cumulative density function:

        $$ F(s) = \frac{\sum_{k=1}^{s}k^m\,p(k)}{\lambda_m} $$

        or

        $$ F(s) = \frac{1}{\lambda_m} {\int_{0}^{s}x^m\,p(x)\mathrm{d}x} $$

        Parameters
        ----------
        size : int | float | ArrayLike
            Chain length or molar mass.
        dist : str
            Type of distribution. Options: `'number'`, `'mass'`, `'gpc'`.
        unit_size : str
            Unit of variable `size`. Options: `'chain_length'` or
            `'molar_mass'`.

        Returns
        -------
        float | np.ndarray
            Cumulative probability.
        """
        # Convert size, if required
        size = self._process_size(size, unit_size)

        # Compute distribution
        check_in_set(dist, set(self.distnames.keys()), 'dist')
        order = self.distnames[dist]
        return self._cdf(size, order)/self._moment(order)

    def random(self, size: int | tuple[int, ...] | None = None) -> int | np.ndarray:
        r"""Generate random sample of chain lengths according to the
        corresponding number probability density/mass function.

        Parameters
        ----------
        size : int | tuple[int, ...] | None
            Sample size.

        Returns
        -------
        int | np.ndarray
            Random sample of chain lengths.
        """
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._random(size)

    def plot(self, dist: str | list[str, ...] = 'mass',
             unit_size: str = 'chain_length',
             xscale: str = 'auto', xrange: tuple = (), ax=None):
        """Plot the chain-length distribution.

        Parameters
        ----------
        dist : str | list[str, ...]
            Type of distribution. Options: `'number'`, `'mass'`, `'gpc'`.
        unit_size : str
            Unit of variable `size`. Options: `'chain_length'` or
            `'molar_mass'`.
        xscale : str
            x-axis scale. Options: `'linear'`, `'log'`, `'auto'`.
        xrange : tuple
            x-axis range.
        ax : matplotlib.axes
            Matplotlib axes object.

        Returns
        -------
        matplotlib.axes
            Matplotlib axes object.

        """
        # Check inputs
        check_in_set(unit_size, {'chain_length', 'molar_mass'}, 'unit_size')
        check_in_set(xscale, {'linear', 'log', 'auto'}, 'xscale')
        check_type(dist, (str, list, tuple), 'dist')
        if isinstance(dist, str):
            dist = [dist]

        # Create axis if none is provided
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(f"Distribution: {self.name}")
            self.fig = fig

        # x-axis
        if len(xrange) != 2:
            xrange = self._xrange_auto()
        if xscale == 'log' or (xscale == 'auto' and set(dist) == {'gpc'}):
            x = np.geomspace(xrange[0], xrange[1], 200)
            xscale = 'log'
        else:
            x = np.linspace(xrange[0], xrange[1], 200)
            xscale = 'linear'
        if unit_size == "chain_length":
            xp = x
            label_x = "Chain length"
        elif unit_size == "molar_mass":
            xp = x * self.M0
            label_x = "Molar mass"
        else:
            raise ValueError

        # y-axis
        for item in dist:
            y = self.pdf(x, dist=item, unit_size="chain_length")
            ax.plot(xp, y, label=item)

        # Other properties
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        ax.grid(True)
        ax.set_xlabel(label_x)
        ax.set_ylabel("Relative abundance")
        ax.set_xscale(xscale)

        return ax

    @abstractmethod
    def _moment(self, order: int) -> float:
        """Moments of the _number_ probability density/mass function.

        Each child class must implement a method delivering the first four
        moments (0-3) of the _number_ pdf.

        Parameters
        ----------
        order : int
            Order of the moment.

        Returns
        -------
        float
            Moment of the _number_ distribution.
        """
        return 0.0

    @abstractmethod
    def _pdf(self, k: int | float | np.ndarray) -> float | np.ndarray:
        """Probability density/mass function.

        Each child class must implement a method to delivering the probability
        density function for the _number_ probability density/mass function.

        Parameters
        ----------
        k : int | float | np.ndarray
            Chain length.

        Returns
        -------
        float | np.ndarray
            Probability density/mass value.
        """
        return 0.0

    @abstractmethod
    def _cdf(self, k: int | float | np.ndarray, order: int) -> float | np.ndarray:
        """Cumulative density function.

        Each child class must implement a method to delivering the cumulative
        density function for the number, mass _and_ GPC distribution. All three
        cases must be covered, because it is not straightforward to convert
        from one kind of distribution to another.

        Parameters
        ----------
        k : int | float | np.ndarray
            Chain length.
        order : int
            Order of the distribution (0: number, 1: mass, 2: GPC).

        Returns
        -------
        float | np.ndarray
            Cumulative density value.
        """
        return 0.0

    @abstractmethod
    def _random(self, size: int | tuple | None) -> int | np.ndarray:
        """Random chain-length generator.

        Each child class must implement a method to generate random chain
        lengths according to the statistics of corresponding number 
        density/mass function.

        Parameters
        ----------
        size : int | tuple[int, ...] | None
            Sample size.

        Returns
        -------
        int | np.ndarray
            Random sample of chain lengths.
        """
        return 0

    @abstractmethod
    def _xrange_auto(self) -> tuple[int, int]:
        """Default chain-length range for distribution plots.

        Returns
        -------
        tuple[int, int]
            (xmin, xmax)
        """
        return (0, 1)


class Flory(BaseDistribution):
    r"""Flory-Schulz (aka most-probable) chain-length distribution, with
    *number* probability mass function given by:

    $$ p(k) = (1-a)a^{k-1} $$

    where $a=1-1/DP_n$.
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

    def _cdf(self, s, order):
        a = self._a
        if order == 0:
            result = 1 - a**s
        elif order == 1:
            result = (a**s*(-a*s + s + 1) - 1)/(a - 1)
        elif order == 2:
            result = (-((a - 1)**2*s**2 - 2*(a - 1)*s + a + 1)
                      * a**s + a + 1)/(a - 1)**2
        else:
            raise ValueError("Not defined for order>2.")
        return result

    def _xrange_auto(self):
        return (1, 10*self.DPn)

    def _random(self, size):
        return self._rng.geometric((1-self._a), size=size)


class Poisson(BaseDistribution):
    r"""Poisson chain-length distribution, with *number* probability mass
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

    def _cdf(self, s, order):
        a = self._a
        if order == 0:
            result = sp.gammaincc(s, a)
        elif order == 1:
            result = (a + 1)*sp.gammaincc(s, a) - \
                np.exp(s*np.log(a) - a - sp.gammaln(s))
        elif order == 2:
            result = (a*(a + 3) + 1)*sp.gammaincc(s, a) - \
                np.exp(s*np.log(a) + np.log(a + s + 2) - a - sp.gammaln(s))
        else:
            raise ValueError("Not defined for order>2.")
        return result

    def _xrange_auto(self):
        return (max(1, self.DPn/2 - 10), 1.5*self.DPn + 10)

    def _random(self, size):
        return self._rng.poisson(self._a, size=size) + 1


class LogNormal(BaseDistribution):
    r"""Log-normal chain-length distribution, with _number_ probability density
    function given by:

    $$ p(x) = \frac{1}{x \sigma \sqrt{2 \pi}}
    \exp\left (- \frac{(\ln{x}-\mu)^2}{2\sigma^2} \right ) $$

    where $\u = \log(DP_n/\sqrt(PDI))$ and $\sigma=\sqrt(\log(PDI))$.
    """
    # https://reference.wolfram.com/language/ref/LogNormalDistribution.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html

    def __init__(self, DPn: int = 100, PDI: float = 2.0,  M0: float = 100.0,
                 name: str = ""):
        """Initialize chain-length distribution.

        Parameters
        ----------
        DPn : int
            Number-average degree of polymerization, $DP_n$.
        PDI : float
            Polydispersity index, $PDI$.
        M0 : float
            Molar mass of the repeating unit, $M_0$.
        name : str
            Name.
        """
        super().__init__(DPn=DPn, M0=M0, name=name)
        self.PDI = PDI

    @property
    def PDI(self) -> float:
        """Polydispersity index, $M_w/M_n$."""
        return self.__PDI

    @PDI.setter
    def PDI(self, PDI: float):
        self.__PDI = check_bounds(PDI, 1.001, np.Inf, 'PDI')
        self._compute_parameters()

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
        return exp(order*mu + (1/2)*order**2*sigma**2)

    def _pdf(self, x):
        mu = self._mu
        sigma = self._sigma
        scale = np.exp(mu)
        return st.lognorm.pdf(x, s=sigma, loc=0, scale=scale)

    def _cdf(self, s, order):
        mu = self._mu
        sigma = self._sigma
        if order == 0:
            scale = np.exp(mu)
            result = st.lognorm.cdf(s, s=sigma, loc=0, scale=scale)
        elif order == 1:
            result = sp.erfc((mu + sigma**2 - np.log(s)) /
                             (sigma*sqrt(2)))*exp(mu + (1/2)*sigma**2)/2
        elif order == 2:
            result = sp.erfc((mu + 2*sigma**2 - np.log(s)) /
                             (sigma*sqrt(2)))*exp(2*mu + 2*sigma**2)/2
        else:
            raise ValueError("Not defined for order>2.")
        return result

    def _xrange_auto(self):
        return (1, 100*self.DPn)

    def _random(self, size):
        return np.rint(self._rng.lognormal(self._mu, self._sigma, size=size))


class CombinedDistribution(Distribution):
    """Combined chain-length distribution."""

    def __init__(self, dists: list[BaseDistribution], weights=list[float],
                 name: str = ""):
        # Validate input !!!
        self._dists = dists
        self._mass_weights = weights
        self.name = name

        # Convert mass to mole fractions
        self._calc_mole_weights()

    def __add__(self, other):
        if isinstance(other, CombinedDistribution):
            return CombinedDistribution(self._dists + other._dists,
                                        self._mass_weights + other._mass_weights,
                                        name=self.name+'+'+other.name)
        elif isinstance(other, BaseDistribution):
            return CombinedDistribution(self._dists + [other],
                                        self._mass_weights + [1],
                                        name=self.name+'+'+other.name)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def _calc_mole_weights(self) -> None:
        """Calculate mole fraction of each base distribution."""
        w = np.asarray(self._mass_weights)
        Mn = np.asarray([d.Mn for d in self._dists])
        x = w/Mn
        x[:] /= x.sum()
        self._mole_weights = x

    def moment(self, order: int) -> float:
        r"""$m$-th moment of the number chain-length distribution"""
        mom = [d.moment(order) for d in self._dists]
        return np.sum(self._mole_weights*np.asarray(mom))

    @ property
    def M0(self) -> float:
        """Molar mass of the repeating unit, $M_0$."""
        m0 = [d.M0 for d in self._dists]
        return np.sum(self._mole_weights*np.asarray(m0))

    @ property
    def DPn(self) -> float:
        """Number-average degree of polymerization, $DP_n$."""
        return self.moment(1)/self.moment(0)

# %%


DPn = 100
d = LogNormal(DPn, 2)
x = np.asarray([i for i in range(0, 10*DPn)])
cdf = d.cdf(x, 'number')
print(cdf[-1])
