# %%

from base import Base
from utils import check_bounds, check_type, check_in_set

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
        r"""Polydispersity index, $DP_w/DP_n$."""
        return self.DPw / self.DPn

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


class SingleDistribution(Distribution):
    """Abstract class for all single chain-length distributions."""

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
        self._rnginit = None

    @abstractmethod
    def _pdf(self, k: int | float | np.ndarray) -> tuple[int, float | np.ndarray]:
        """Probability density/mass function.

        Each child class must implement a method to delivering the probability
        density function for the number (order=0), mass (order=1) _or_
        GPC (order=2) distribution. Only one of the three kinds of pdf must be
        supplied, because converting among different kinds of pdf is
        straightforward.

        Parameters
        ----------
        k : int | float | np.ndarray
            Chain length.

        Returns
        -------
        tuple[int, float | np.ndarray]
            The first element of the tuple is the order of the pdf and the
            second element is the pdf value.
        """
        return (0, 0.0)

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
            Cumulative density function.
        """
        return 0.0

    @abstractmethod
    def _moment(self, order: int) -> float:
        """Moments of the _number_ probability density/mass function.

        Each child class must implement a method delivering the first four
        moments of the _number_ pdf.

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
    def _rng(self, size: int | tuple | None) -> int:
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

    def __str__(self) -> str:
        return f"name: {self.name}\n" + \
            f"DPn:  {self.DPn:.1f}\n" + \
            f"DPw:  {self.DPw:.1f}\n" + \
            f"DPz:  {self.DPw:.1f}\n" + \
            f"PDI:  {self.PDI:.2f}\n" + \
            f"M0:   {self.M0:,.1f}\n" + \
            f"Mn:   {self.Mn:,.0f}\n" + \
            f"Mw:   {self.Mw:,.0f}\n" + \
            f"Mz:   {self.Mz:,.0f}"

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return CompositeDistribution([self], [other], name=self.name)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, SingleDistribution):
            return CompositeDistribution([self, other], [1, 1],
                                         name=self.name+'+'+other.name)
        else:
            return NotImplemented

    def moment(self, order: int) -> float:
        r"""$m$-th moment of the number chain-length distribution:

        $$ \lambda_m=\sum_{k=1}^{\infty }k^m\,p(k) $$

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
        order_ref, p = self._pdf(size)
        delta_order = order - order_ref

        return p * size**delta_order / self._moment(delta_order)

    def cdf(self, size: int | float | ArrayLike, dist: str = "mass",
            unit_size: str = "chain_length") -> float | np.ndarray:
        r"""Evaluate the cumulative density function:

        $$ F(s) = \frac{\sum_{k=1}^{s}k^m\,p(k)}{\lambda_m} $$

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
        if self._rnginit is None:
            self._rnginit = np.random.default_rng()
        return self._rng(size)

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
            x = np.geomspace(xrange[0], xrange[1], 100)
            xscale = 'log'
        else:
            x = np.linspace(xrange[0], xrange[1], 100)
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


class Flory(SingleDistribution):
    r"""Flory-Schulz (aka most-probable) chain-length distribution, with
    *number* probability mass function given by:

    $$ p(k) = (1-a)a^{k-1} $$

    and first leading moments given by:

    $$\begin{align}
    \lambda_0 &= 1 \\
    \lambda_1 &= DP_n \\
    \lambda_2 &= DP_n (2 DP_n - 1) \\
    \lambda_3 &= DP_n (1 + 6 DP_n (DP_n - 1))
    \end{align}$$

    where $a=1-1/DP_n$.
    """

    def _pdf(self, k):
        a = 1 - 1 / self.DPn
        p = (1 - a) * a ** (k - 1)
        order = 0
        return (order, p)

    def _cdf(self, s, order):
        a = 1 - 1 / self.DPn
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

    def _moment(self, order):
        if order == 0:
            result = 1
        elif order == 1:
            result = self.DPn
        elif order == 2:
            result = self.DPn * (2 * self.DPn - 1)
        elif order == 3:
            result = self.DPn * (1 + 6 * self.DPn * (self.DPn - 1))
        else:
            raise ValueError("Not defined for order>3.")
        return result

    def _rng(self, size):
        return self._rnginit.geometric(p=1/self.DPn, size=size)


class Poisson(SingleDistribution):
    r"""Poisson chain-length distribution, with *number* probability mass
    function given by:

    $$ p(k) = \frac{a^{k-1} e^{-a}}{\Gamma(k)} $$

    and first leading moments given by:

    $$\begin{align}
    \lambda_0 &= 1 \\
    \lambda_1 &= DP_n \\
    \lambda_2 &= a^2 + 3a + 1 \\
    \lambda_3 &= a^3 + 6a^2 + 7a + 1
    \end{align}$$

    where $a=DP_n-1$.
    """

    def _pdf(self, k):
        a = self.DPn - 1
        p = np.exp((k - 1)*np.log(a) - a - sp.gammaln(k))
        order = 0
        return (order, p)

    def _cdf(self, s, order):
        a = self.DPn - 1
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

    def _moment(self, order):
        a = self.DPn - 1
        if order == 0:
            result = 1
        elif order == 1:
            result = self.DPn
        elif order == 2:
            result = a**2 + 3*a + 1
        elif order == 3:
            result = a**3 + 6*a**2 + 7*a + 1
        else:
            raise ValueError("Not defined for order>3.")
        return result

    def _rng(self, size):
        return self._rnginit.poisson(lam=(self.DPn - 1), size=size) + 1


class LogNormal(SingleDistribution):
    r"""Log-normal chain-length distribution, with *number* probability density
    function given by:

    $$ p(k) = \frac{a^{k-1} e^{-a}}{\Gamma(k)} $$

    and first leading moments given by:

    $$\lambda_0 = 1 $$

    $$\lambda_1 = 1 $$

    $$\lambda_2 = 1 $$

    $$\lambda_3 = 1 $$

    where $a=DP_n-1$.
    """

    def __init__(self, DPn: int = 100, PDI: float = 2.0,  M0: float = 100.0,
                 name: str = ""):
        """Initialize chain-length distribution.
        Args:
            DPn (int, optional): Number-average degree of polymerization, $DP_n$.
            PDI (float, optional): Polydispersity index, $PDI$.
            M0 (float, optional): Molar mass of the repeating unit, $M_0$.
            name (str, optional): Name.
        """
        super().__init__(DPn=DPn, M0=M0, name=name)
        self.PDI = PDI

    @property
    def PDI(self) -> float:
        """Polydispersity index, $DP_w/DP_n$."""
        return self.__PDI

    @PDI.setter
    def PDI(self, PDI: float):
        """Polydispersity index, $DP_w/DP_n$."""
        self.__PDI = check_bounds(PDI, 1, np.Inf, 'PDI')

    def _pdf(self, x):
        mu = np.log(self.DPn)
        sigma = 1
        scale = np.exp(mu)
        p = st.lognorm.pdf(x, s=sigma, loc=0, scale=scale)
        order = 0
        return (order, p)

    def _cdf(self, s, order):
        a = 0
        if order == 0:
            result = 0
        elif order == 1:
            result = 0
        elif order == 2:
            result = 0
        else:
            raise ValueError("Not defined for order>2.")
        return result

    def _xrange_auto(self):
        return (1, 10*self.DPn)

    def _moment(self, order: int):
        a = self.DPn - 1
        if order == 0:
            result = 1
        elif order == 1:
            result = 1
        elif order == 2:
            result = 1
        elif order == 3:
            result = 1
        else:
            raise ValueError("Not defined for order>3.")
        return result

    def _rng(self, size):
        return self._rnginit.lognormal(mean=(self.DPn - 1), size=size)


class CompositeDistribution(Distribution):
    """Composite chain-length distribution."""

    def __init__(self, dists: list[SingleDistribution], weights=list[float],
                 name: str = ""):
        self.name = name
        self._dists = dists
        self._weights = weights

    def __add__(self, other):
        if isinstance(other, CompositeDistribution):
            return CompositeDistribution(self._dists + other._dists,
                                         self._weights + other._weights,
                                         name=self.name+'+'+other.name)
        elif isinstance(other, SingleDistribution):
            return CompositeDistribution(self._dists + [other],
                                         self._weights + [1],
                                         name=self.name+'+'+other.name)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    @property
    def M0(self) -> float:
        """Molar mass of the repeating unit, $M_0$."""
        return 0.0

    @property
    def DPn(self) -> float:
        """Number-average degree of polymerization, $DP_n$."""
        return self.moment(1)/self.moment(0)

    def moment(self, order: int) -> float:
        w = self._weights
        Mn = [d.Mn for d in self._dists]
        x = np.asarray(w)/np.asarray(Mn)
        x[:] /= x.sum()
        result = 0
        for i, d in enumerate(self._dists):
            result += x[i]*d.moment(order)
        return result
