# %%

from polykin.base import Base
from polykin.utils import check_bounds, check_type, check_in_set, add_dicts

import numpy as np
from numpy import int64, float64, dtype, ndarray
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from typing import Any, Literal, Union
from abc import ABC, abstractmethod


class GeneralDistribution(Base, ABC):
    """Abstract class for all chain-length distributions."""

    typenames = {'number': 0, 'mass': 1, 'gpc': 2}
    sizenames = {'length', 'mass'}

    @property
    def DPn(self) -> float:
        r"""Number-average degree of polymerization, $DP_n$."""
        return self._moment_ratio(1, 'length')

    @property
    def DPw(self) -> float:
        r"""Mass-average degree of polymerization, $DP_w$."""
        return self._moment_ratio(2, 'length')

    @property
    def DPz(self) -> float:
        r"""z-average degree of polymerization, $DP_z$."""
        return self._moment_ratio(3, 'length')

    @property
    def Mn(self) -> float:
        r"""Number-average molar mass, $M_n$."""
        return self._moment_ratio(1, 'mass')

    @property
    def Mw(self) -> float:
        r"""Weight-average molar mass, $M_w$."""
        return self._moment_ratio(2, 'mass')

    @property
    def Mz(self) -> float:
        r"""z-average molar mass, $M_z$."""
        return self._moment_ratio(3, 'mass')

    @property
    def PDI(self) -> float:
        r"""Polydispersity index, $M_w/M_n$."""
        return self.Mw / self.Mn

    def _moment_ratio(self,
                      order: int,
                      sizeas: Literal['length', 'mass']
                      ) -> float:
        r"""Ratio of consecutive moments of the number chain-length / molar
        mass distribution

        $$ \rho_m = frac{\lambda_{m}}{\lambda_{m-1}} $$

        Parameters
        ----------
        order : int
            order of moment on the numerator
        sizeas : Literal['length', 'mass']
            Switch between chain-*length* or molar *mass* moments.

        Returns
        -------
        float
            Ratio of moments, $\rho_m$.
        """
        return self.moment(order, sizeas)/self.moment(order-1, sizeas)

    @property
    @abstractmethod
    def M0(self) -> float:
        r"""Number-average molar mass of the repeating units, $M_0=M_n/DP_n$."""
        return 0.0

    @abstractmethod
    def moment(self,
               order: int,
               sizeas: Literal['length', 'mass'] = 'length'
               ) -> float:
        """$m$-th moment of the number chain-length / molar mass distribution.
        """
        return 0.0


class IndividualDistribution(GeneralDistribution):
    """Abstract class for all individual chain-length distributions."""

    def __init__(self,
                 DPn: int,
                 M0: float = 100.0,
                 name: str = ''
                 ) -> None:
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
            return MixtureDistribution({self: other}, name=self.name)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, IndividualDistribution):
            return MixtureDistribution(add_dicts({self: 1}, {other: 1}),
                                       name=self.name+'+'+other.name)
        else:
            return NotImplemented

    @property
    def M0(self) -> float:
        r"""Number-average molar mass of the repeating units, $M_0=M_n/DP_n$."""
        return self.__M0  # type: ignore

    @M0.setter
    def M0(self, M0: float):
        self.__M0 = check_bounds(M0, 0.0, np.Inf, 'M0')

    @property
    def DPn(self) -> float:
        r"""Number-average degree of polymerization, $DP_n$."""
        return self.__DPn  # type: ignore

    @DPn.setter
    def DPn(self, DPn: int = 100):
        self.__DPn = check_bounds(DPn, 2, np.Inf, 'DPn')
        self._compute_parameters()

    def _compute_parameters(self):
        pass

    def moment(self,
               order: int,
               sizeas: Literal['length', 'mass'] = 'length'
               ) -> float:
        r"""$m$-th moment of the number chain-length (or molar mass)
        distribution:

        $$ \lambda_m=\sum_{k=1}^{\infty }k^m\,p(k) $$

        or

        $$ \lambda_m=\int_{0}^{\infty }x^m\,p(x)\mathrm{d}x $$

        Parameters
        ----------
        order : int
            Order of the moment, $0 \le m \le 3$.
        length : Literal['length', 'mass']
            If `length`, the result will be the moment of the chain-length
            distribution; if `mass`, the result will be the moment of the
            molar mass distribution.

        Returns
        -------
        float
            Moment, $\lambda_m$
        """
        check_bounds(order, 0, 3, 'order')
        check_in_set(sizeas, self.sizenames, 'sizeas')
        result = self._moment(order)
        if sizeas == 'mass':
            result *= self.M0**order
        return result

    def _preprocess_size(self,
                         size: Union[int, float, ArrayLike],
                         sizeas: Literal['length', 'mass']
                         ) -> Union[int, float, ndarray]:
        if isinstance(size, list):
            size = np.asarray(size)
        check_in_set(sizeas, self.sizenames, 'sizeas')
        if sizeas == 'mass':
            size = size/self.M0
        return size  # type: ignore

    def pdf(self,
            size: Union[int, float, ArrayLike],
            type: Literal['number', 'mass', 'gpc'] = 'mass',
            sizeas: Literal['length', 'mass'] = 'length',
            ) -> Union[float, ndarray[Any, dtype[float64]]]:
        r"""Evaluate the probability density function, $p(k)$.

        Parameters
        ----------
        size : int | float | ArrayLike
            Chain length or molar mass.
        type : Literal['number', 'mass', 'gpc']
            Type of distribution.
        sizeas : Literal['length', 'mass']
            Set `length` if `size` refers to chain-*length* or `mass` if `size`
            refers to molar *mass*.

        Returns
        -------
        float | ndarray
            Probability density.
        """

        # Convert size, if required
        size = self._preprocess_size(size, sizeas)

        # Compute distribution
        check_in_set(type, set(self.typenames.keys()), 'type')
        order = self.typenames[type]
        return self._pdf(size) * size**order / self._moment(order)

    def cdf(self,
            size: Union[int, float, ArrayLike],
            type: Literal['number', 'mass', 'gpc'] = 'mass',
            sizeas: Literal['length', 'mass'] = 'length',
            ) -> Union[float, ndarray[Any, dtype[float64]]]:
        r"""Evaluate the cumulative density function:

        $$ F(s) = \frac{\sum_{k=1}^{s}k^m\,p(k)}{\lambda_m} $$

        or

        $$ F(s) = \frac{1}{\lambda_m} {\int_{0}^{s}x^m\,p(x)\mathrm{d}x} $$

        Parameters
        ----------
        size : int | float | ArrayLike
            Chain length or molar mass.
        type : Literal['number', 'mass', 'gpc']
            Type of distribution.
        sizeas : Literal['length', 'mass']
            Set `length` if `size` refers to chain-*length* or `mass` if `size`
            refers to molar *mass*.

        Returns
        -------
        float | ndarray
            Cumulative probability.
        """
        # Convert size, if required
        size = self._preprocess_size(size, sizeas)

        # Compute distribution
        check_in_set(type, set(self.typenames.keys()), 'type')
        order = self.typenames[type]
        return self._cdf(size, order)

    def random(self,
               size: Union[int, tuple[int, ...], None] = None
               ) -> Union[int, ndarray[Any, dtype[int64]]]:
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

    def plot(self,
             type: Literal['number', 'mass', 'gpc'] = 'mass',
             sizeas: Literal['length', 'mass'] = 'length',
             xscale: Literal['auto', 'linear', 'log'] = 'auto',
             xrange: tuple = (),
             ax=None
             ) -> None:
        """Plot the chain-length distribution.

        Parameters
        ----------
        type : Literal['number', 'mass', 'gpc']
            Type of distribution.
        sizeas : Literal['length', 'mass']
            Set `length` if `size` refers to chain-*length* or `mass` if `size`
            refers to molar *mass*.
        xscale : Literal['auto', 'linear', 'log']
            x-axis scale.
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
        check_in_set(type, set(self.typenames.keys()), 'type')
        check_in_set(sizeas, self.sizenames, 'sizeas')
        check_in_set(xscale, {'linear', 'log', 'auto'}, 'xscale')
        check_type(type, (str, list, tuple), 'type')
        if isinstance(type, str):
            type = [type]

        # Create axis if none is provided
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(f"Distribution: {self.name}")
            self.fig = fig

        # x-axis
        if len(xrange) != 2:
            xrange = self._xrange_auto()
        if xscale == 'log' or (xscale == 'auto' and set(type) == {'gpc'}):
            x = np.geomspace(xrange[0], xrange[1], 200)
            xscale = 'log'
        else:
            x = np.linspace(xrange[0], xrange[1], 200)
            xscale = 'linear'
        if sizeas == 'length':
            xp = x
            label_x = "Chain length"
        elif sizeas == 'mass':
            xp = x * self.M0
            label_x = "Molar mass"
        else:
            raise ValueError

        # y-axis
        for item in type:
            y = self.pdf(x, type=item, sizeas='length')
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
             k: Union[int, float, ndarray]
             ) -> Union[float, ndarray[Any, dtype[float64]]]:
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
             k: Union[int, float, ndarray],
             order: int) \
            -> Union[float, ndarray[Any, dtype[float64]]]:
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
                size: Union[int, tuple, None],
                ) -> Union[int, ndarray[Any, dtype[int64]]]:
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


class IndividualDistributionP1(IndividualDistribution):
    """Abstract class for 1-parameter single chain-length distributions."""
    pass


class IndividualDistributionP2(IndividualDistribution):
    """Abstract class for 2-parameter single chain-length distributions."""

    def __init__(self,
                 DPn: int,
                 PDI: float,
                 M0: float = 100.0,
                 name: str = '') -> None:
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


class MixtureDistribution(GeneralDistribution):
    """Mixture chain-length distribution."""

    def __init__(self,
                 components: dict[IndividualDistribution, Union[int, float]],
                 name: str = "") -> None:

        self._components = components
        self.name = name
        self._molefracs = None

    def __add__(self, other):
        if isinstance(other, MixtureDistribution):
            return MixtureDistribution(add_dicts(self._components, other._components),
                                       name=self.name+'+'+other.name)
        elif isinstance(other, IndividualDistribution):
            return MixtureDistribution(add_dicts(self._components, {other: 1}),
                                       name=self.name+'+'+other.name)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self) -> str:
        return f"type: {self.__class__.__name__}\n" + \
            f"name: {self.name}\n" + \
            f"DPn:  {self.DPn:.1f}\n" + \
            f"M0:   {self.M0:.1f}\n" + \
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

    def moment(self,
               order: int,
               sizeas: Literal['length', 'mass'] = 'length'
               ) -> float:
        r"""$m$-th moment of the number chain-length (or molar mass)
        distribution.

        Parameters
        ----------
        order : int
            Order of the moment, $0 \le m \le 3$.
        length : Literal['length', 'mass']
            If `length`, the result will be the moment of the chain-length
            distribution; if `mass`, the result will be the moment of the
            molar mass distribution.

        Returns
        -------
        float
            Moment, $\lambda_m$
        """
        check_bounds(order, 0, 3, 'order')
        check_in_set(sizeas, self.sizenames, 'sizeas')

        if self._molefracs is None:
            self._calc_molefracs()
        mom = np.asarray([d.moment(order, 'length')
                         for d in self._components.keys()])
        M0 = np.asarray([d.M0 for d in self._components.keys()])
        result = self._molefracs*mom
        if sizeas == 'mass':
            result *= M0**order

        return result.sum()

    @ property
    def M0(self) -> float:
        """Number-average molar mass of the repeating units, $M_0=M_n/DP_n$."""
        return self.Mn/self.DPn
