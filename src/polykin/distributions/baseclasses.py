# %%

from polykin.base import Base
from polykin.utils import check_bounds, check_type, check_in_set, add_dicts

import numpy as np
from numpy import int64, float64, dtype, ndarray
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from typing import Any
from abc import ABC, abstractmethod


class Distribution(Base, ABC):
    """Abstract class for all chain-length distributions."""

    distnames = {'number': 0, 'mass': 1, 'gpc': 2}

    @property
    def DPn(self) -> float:
        r"""Number-average degree of polymerization, $DP_n$."""
        return self._moment_ratio(1)

    @property
    def DPw(self) -> float:
        r"""Mass-average degree of polymerization, $DP_w$."""
        return self._moment_ratio(2)

    @property
    def DPz(self) -> float:
        r"""z-average degree of polymerization, $DP_z$."""
        return self._moment_ratio(3)

    @property
    def Mn(self) -> float:
        r"""Number-average molar mass, $M_n$."""
        return self._moment_ratio(1, False)

    @property
    def Mw(self) -> float:
        r"""Weight-average molar mass, $M_w$."""
        return self._moment_ratio(2, False)

    @property
    def Mz(self) -> float:
        r"""z-average molar mass, $M_z$."""
        return self._moment_ratio(3, False)

    @property
    def PDI(self) -> float:
        r"""Polydispersity index, $M_w/M_n$."""
        return self.Mw / self.Mn

    def _moment_ratio(self, order: int, length: bool = True) -> float:
        return self.moment(order, length)/self.moment(order-1, length)

    # @property
    # @abstractmethod
    # def M0(self) -> float:
    #     r"""Molar mass of the repeating unit, $M_0$."""
    #     return 0.0

    @abstractmethod
    def moment(self, order: int, length: bool) -> float:
        """$m$-th moment of the number chain-length / molar mass distribution.
        """
        return 0.0


class Single(Distribution):
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
        self._rng = None  # type: ignore

    def __str__(self) -> str:
        return f"type: {self.__class__.__name__}\n" + \
            f"name: {self.name}\n" + \
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
            return Combined({self: other}, name=self.name)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, Single):
            return Combined(add_dicts({self: 1}, {other: 1}),
                            name=self.name+'+'+other.name)
        else:
            return NotImplemented

    @property
    def M0(self) -> float:
        r"""Molar mass of the repeating unit, $M_0$."""
        return self.__M0  # type: ignore

    @M0.setter
    def M0(self, M0: float):
        self.__M0 = check_bounds(M0, 0.0, np.Inf, "M0")

    @property
    def DPn(self) -> float:
        r"""Number-average degree of polymerization, $DP_n$."""
        return self.__DPn  # type: ignore

    @DPn.setter
    def DPn(self, DPn: int = 100):
        self.__DPn = check_bounds(DPn, 2, np.Inf, "DPn")
        self._compute_parameters()

    def _compute_parameters(self):
        pass

    def moment(self, order: int, length: bool = True) -> float:
        r"""$m$-th moment of the number chain-length (or molar mass)
        distribution:

        $$ \lambda_m=\sum_{k=1}^{\infty }k^m\,p(k) $$

        or

        $$ \lambda_m=\int_{0}^{\infty }x^m\,p(x)\mathrm{d}x $$

        Parameters
        ----------
        order : int
            Order of the moment, $0 \le m \le 3$.
        length : bool
            If `True`, the result will be the moment of the chain-length
            distribution; if `False`, the result will be the moment of the
            molar mass distribution.

        Returns
        -------
        float
            Moment, $\lambda_m$
        """
        check_bounds(order, 0, 3, 'order')
        result = self._moment(order)
        if not length:
            result *= self.M0**order
        return result

    def _process_size(self, size, length) -> ndarray:
        if isinstance(size, list):
            size = np.asarray(size)
        check_type(length, bool, 'length')
        if not length:
            size = size/self.M0
        return size

    def pdf(self, size: int | float | ArrayLike, dist: str = "mass",
            length: bool = True) -> float | ndarray[Any, dtype[float64]]:
        r"""Evaluate the probability density function, $p(k)$.

        Parameters
        ----------
        size : int | float | ArrayLike
            Chain length or molar mass.
        dist : str
            Type of distribution. Options: `'number'`, `'mass'`, `'gpc'`.
        length : bool
            Set `True` if `size` refers to chain-length or `False` if `size`
            refers to molar mass.

        Returns
        -------
        float | ndarray
            Probability density.
        """

        # Convert size, if required
        size = self._process_size(size, length)

        # Compute distribution
        check_in_set(dist, set(self.distnames.keys()), 'dist')
        order = self.distnames[dist]
        return self._pdf(size) * size**order / self._moment(order)

    def cdf(self, size: int | float | ArrayLike, dist: str = "mass",
            length: bool = True) -> float | ndarray[Any, dtype[float64]]:
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
        length : bool
            Set `True` if `size` refers to chain-length or `False` if `size`
            refers to molar mass.

        Returns
        -------
        float | ndarray
            Cumulative probability.
        """
        # Convert size, if required
        size = self._process_size(size, length)

        # Compute distribution
        check_in_set(dist, set(self.distnames.keys()), 'dist')
        order = self.distnames[dist]
        return self._cdf(size, order)

    def random(self, size: int | tuple[int, ...] | None = None) \
            -> int | ndarray[Any, dtype[int64]]:
        r"""Generate random sample of chain lengths according to the
        corresponding number probability density/mass function.

        Parameters
        ----------
        size : int | tuple[int, ...] | None
            Sample size.

        Returns
        -------
        int | ndarray
            Random sample of chain lengths.
        """
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._random(size)

    def plot(self, dist: str | list[str] = 'mass',
             unit_size: str = 'chain_length',
             xscale: str = 'auto', xrange: tuple = (), ax=None) -> None:
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
            y = self.pdf(x, dist=item, length=True)
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
    def _pdf(self,
             k: int | float | ndarray
             ) -> float | ndarray[Any, dtype[float64]]:
        """Probability density/mass function.

        Each child class must implement a method to delivering the probability
        density function for the _number_ probability density/mass function.

        Parameters
        ----------
        k : int | float | ndarray
            Chain length.

        Returns
        -------
        float | ndarray
            Probability density/mass value.
        """
        return 0.0

    @abstractmethod
    def _cdf(self,
             k: int | float | ndarray,
             order: int) \
            -> float | ndarray[Any, dtype[float64]]:
        """Cumulative density function.

        Each child class must implement a method to delivering the cumulative
        density function for the number, mass _and_ GPC distribution. All three
        cases must be covered, because it is not straightforward to convert
        from one kind of distribution to another.

        Parameters
        ----------
        k : int | float | ndarray
            Chain length.
        order : int
            Order of the distribution (0: number, 1: mass, 2: GPC).

        Returns
        -------
        float | ndarray
            Cumulative density value.
        """
        return 0.0

    @abstractmethod
    def _random(self,
                size: int | tuple | None,
                ) -> int | ndarray[Any, dtype[int64]]:
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
        int | ndarray
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


class Single1P(Single):
    """Abstract class for 1-parameter single chain-length distributions."""
    pass


class Single2P(Single):
    """Abstract class for 2-parameter single chain-length distributions."""

    def __init__(self,
                 DPn: int,
                 PDI: float,
                 M0: float = 100.0,
                 name: str = "") -> None:
        """Initialize 2-parameter single chain-length distribution.

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
        return self.__PDI  # type: ignore

    @PDI.setter
    def PDI(self, PDI: float):
        self.__PDI = check_bounds(PDI, 1.001, np.Inf, 'PDI')
        self._compute_parameters()


class Combined(Distribution):
    """Combined chain-length distribution."""

    def __init__(self,
                 components: dict[Single, int | float],
                 name: str = "") -> None:

        self._components = components
        self.name = name
        self._molefracs = None

    def __add__(self, other):
        if isinstance(other, Combined):
            return Combined(add_dicts(self._components, other._components),
                            name=self.name+'+'+other.name)
        elif isinstance(other, Single):
            return Combined(add_dicts(self._components, {other: 1}),
                            name=self.name+'+'+other.name)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self) -> str:
        return f"type: {self.__class__.__name__}\n" + \
            f"name: {self.name}\n" + \
            f"DPn:  {self.DPn:.1f}\n" + \
            f"PDI:  {self.PDI:.2f}\n" + \
            f"Mn:   {self.Mn:,.0f}\n" + \
            f"Mw:   {self.Mw:,.0f}\n" + \
            f"Mz:   {self.Mz:,.0f}"

    def _calc_molefracs(self) -> None:
        """Calculate mole fraction of each base distribution."""
        components = self._components
        w = np.asarray(list(components.values()))
        Mn = np.asarray([d.Mn for d in components.keys()])
        x = w/Mn
        x[:] /= x.sum()
        self._molefracs = x

    def moment(self, order: int, length: bool = True) -> float:
        """$m$-th moment of the number chain-length / molar mass distribution.
         """
        if self._molefracs is None:
            self._calc_molefracs()
        mom = np.asarray([d.moment(order) for d in self._components.keys()])
        M0 = np.asarray([d.M0 for d in self._components.keys()])
        result = self._molefracs*mom
        if not length:
            result *= M0**order
        return result.sum()

    # @ property
    # def M0(self) -> float:
    #     """Molar mass of the repeating unit, $M_0$."""
    #     m0 = [d.M0 for d in self._dists]
    #     return np.sum(self._mole_weights*np.asarray(m0))