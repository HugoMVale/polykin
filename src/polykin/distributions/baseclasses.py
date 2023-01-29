# %% Base distribution clases

from polykin.base import Base
from polykin.utils import \
    check_bounds, check_type, check_in_set, add_dicts, vectorize

from math import log10
import numpy as np
import mpmath
from numpy import int64, float64, dtype, ndarray
from scipy import integrate
import matplotlib.pyplot as plt
from typing import Any, Literal, Union
from abc import ABC, abstractmethod
import functools


class GeneralDistribution(Base, ABC):
    """Abstract class for all chain-length distributions."""

    kindnames = {'number': 0, 'mass': 1, 'gpc': 2}
    units = {'molar_mass': 'g/mol'}

    def __str__(self) -> str:
        unit_M = self.units['molar_mass']
        return f"type: {self.__class__.__name__}\n" + \
            f"name: {self.name}\n" + \
            f"DPn:  {self.DPn:.1f}\n" + \
            f"DPw:  {self.DPw:.1f}\n" + \
            f"DPz:  {self.DPz:.1f}\n" + \
            f"PDI:  {self.PDI:.2f}\n" + \
            f"M0:   {self.M0:,.1f} {unit_M}\n" + \
            f"Mn:   {self.Mn:,.0f} {unit_M}\n" + \
            f"Mw:   {self.Mw:,.0f} {unit_M}\n" + \
            f"Mz:   {self.Mz:,.0f} {unit_M}"

    @property
    def DPn(self) -> float:
        r"""Number-average degree of polymerization, $DP_n$."""
        return self._moment_mass(1, 1)/self._moment_mass(0)

    @property
    def DPw(self) -> float:
        r"""Mass-average degree of polymerization, $DP_w$."""
        return self._moment_mass(2, 1)/self._moment_mass(1)

    @property
    def DPz(self) -> float:
        r"""z-average degree of polymerization, $DP_z$."""
        return self._moment_mass(3, 1)/self._moment_mass(2)

    @property
    def Mn(self) -> float:
        r"""Number-average molar mass, $M_n$."""
        return self._moment_mass(1)/self._moment_mass(0)

    @property
    def Mw(self) -> float:
        r"""Weight-average molar mass, $M_w$."""
        return self._moment_mass(2)/self._moment_mass(1)

    @property
    def Mz(self) -> float:
        r"""z-average molar mass, $M_z$."""
        return self._moment_mass(3)/self._moment_mass(2)

    @property
    def PDI(self) -> float:
        r"""Polydispersity index, $M_w/M_n$."""
        return self.Mw / self.Mn

    @property
    def M0(self) -> float:
        """Number-average molar mass of the repeating units, $M_0=M_n/DP_n$."""
        return self.Mn / self.DPn

    def pdf(self,
            size: Union[float, list[float], ndarray[Any, dtype[float64]]],
            kind: Literal['number', 'mass', 'gpc'] = 'mass',
            sizeasmass: bool = False,
            ) -> Union[float, ndarray[Any, dtype[float64]]]:
        r"""Evaluate the probability density function, $p(k)$.

        Parameters
        ----------
        size : float | list | ndarray
            Chain length or molar mass.
        kind : Literal['number', 'mass', 'gpc']
            Kind of distribution.
        sizeasmass : bool
            Switch size input between chain-*length* (if `False`) or molar
            *mass* (if `True`).

        Returns
        -------
        float | ndarray
            Probability density.
        """
        # Check inputs
        self._verify_kind(kind)
        self._verify_sizeasmass(sizeasmass)
        order = self.kindnames[kind]
        # Convert list to ndarray
        if isinstance(size, list):
            size = np.asarray(size)
        # Math is done by the corresponding subclass method
        return self._pdf(size, order, sizeasmass)

    def cdf(self,
            size: Union[float, list[float], ndarray[Any, dtype[float64]]],
            kind: Literal['number', 'mass'] = 'mass',
            sizeasmass: bool = False,
            ) -> Union[float, ndarray[Any, dtype[float64]]]:
        r"""Evaluate the cumulative density function:

        $$ F(s) = \frac{\sum_{k=1}^{s}k^m\,p(k)}{\lambda_m} $$

        or

        $$ F(s) = \frac{\int_{0}^{s}x^m\,p(x)\mathrm{d}x}{\lambda_m} $$

        where $m$ is the order (0: number, 1: mass).

        Parameters
        ----------
        size : float | list | ndarray
            Chain length or molar mass.
        kind : Literal['number', 'mass']
            Kind of distribution.
        sizeasmass : bool
            Switch size input between chain-*length* (if `False`) or molar
            *mass* (if `True`).

        Returns
        -------
        float | ndarray
            Cumulative probability.
        """
        # Check inputs
        check_in_set(kind, {'number', 'mass'}, 'kind')
        self._verify_sizeasmass(sizeasmass)
        order = self.kindnames[kind]
        # Convert list to ndarray
        if isinstance(size, list):
            size = np.asarray(size)
        # Math is done by the corresponding subclass method
        result = self._cdf(size, order, sizeasmass)
        return result

    def plot(self,
             kind: Literal['number', 'mass', 'gpc'] = 'mass',
             sizeasmass: bool = False,
             xscale: Literal['auto', 'linear', 'log'] = 'auto',
             xrange: Union[list[float], tuple[float, float],
                           ndarray[Any, dtype[float64]]] = [],
             cdf: Literal[0, 1, 2] = 0,
             ax=None
             ) -> None:
        """Plot the chain-length distribution.

        Parameters
        ----------
        kind : Literal['number', 'mass', 'gpc']
            Kind of distribution.
        sizeasmass : bool
            Switch size input between chain-*length* (if `False`) or molar
            *mass* (if `True`).
        xscale : Literal['auto', 'linear', 'log']
            x-axis scale.
        xrange : Union[list, tuple, ndarray]
            x-axis range.
        cdf : Literal[0, 1, 2]
            y-axis where cdf is displayed. If `0` the cdf if not displayed; if
            `1` the cdf is displayed on the primary y-axis; if `2` the cdf is
            displayed on the secondary axis.
        ax : matplotlib.axes
            Matplotlib axes object.
        """
        # Check inputs
        self._verify_kind(kind)
        self._verify_sizeasmass(sizeasmass)
        check_in_set(xscale, {'linear', 'log', 'auto'}, 'xscale')
        check_in_set(cdf, {0, 1, 2}, cdf)
        if isinstance(kind, str):
            kind = [kind]

        # Create axis if none is provided
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(f"Distribution: {self.name}")
            self.fig = fig

        # x-axis range
        if not (len(xrange) == 2 and xrange[1] > xrange[0]):
            xrange = self._xrange_plot(sizeasmass)
            xrange_user = False
        else:
            xrange = np.asarray(xrange)
            xrange_user = True

        npoints = 200
        if isinstance(self, MixtureDistribution):
            npoints += 100*(len(self._components)-1)
        # x-axis vector and scale
        if xscale == 'log' or (xscale == 'auto' and set(kind) == {'gpc'}):
            if not (xrange_user) and log10(xrange[1]/xrange[0]) > 3:
                xrange[1] *= 10
            x = np.geomspace(*xrange, npoints)  # type: ignore
            xscale = 'log'
        else:
            x = np.linspace(*xrange, npoints)  # type: ignore
            xscale = 'linear'
        # x-axis label
        if sizeasmass:
            label_x = f"Molar mass [{self.units['molar_mass']}]"
        else:
            label_x = "Chain length"

        # y-axis
        label_y_pdf = 'Relative abundance'
        label_y_cdf = 'Cumulative probability'
        bbox_to_anchor = (1.05, 1.0)
        if cdf == 0:
            label_y1 = label_y_pdf
            for item in kind:
                y1 = self.pdf(x, kind=item, sizeasmass=sizeasmass)
                ax.plot(x, y1, label=item)
        elif cdf == 1:
            label_y1 = label_y_cdf
            for item in kind:
                if item == 'gpc':
                    item = 'mass'
                y1 = self.cdf(x, kind=item, sizeasmass=sizeasmass)
                ax.plot(x, y1, label=item)
        elif cdf == 2:
            label_y1 = label_y_pdf
            label_y2 = label_y_cdf
            ax.set_ylabel(label_y1)
            ax2 = ax.twinx()
            ax2.set_ylabel(label_y2)
            bbox_to_anchor = (1.1, 1.0)
            for item in kind:
                y1 = self.pdf(x, kind=item, sizeasmass=sizeasmass)
                ax.plot(x, y1, label=item)
                if item == 'gpc':
                    item = 'mass'
                y2 = self.cdf(x, kind=item, sizeasmass=sizeasmass)
                ax2.plot(x, y2, label=item, linestyle='--')
        else:
            raise ValueError

        # Other properties
        ax.legend(bbox_to_anchor=bbox_to_anchor, loc="upper left")
        ax.grid(True)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y1)
        ax.set_xscale(xscale)

        return None

    def _verify_kind(self, kind):
        """Verify `type` input."""
        return check_in_set(kind, set(self.kindnames.keys()), 'kind')

    def _verify_sizeasmass(self, sizeasmass):
        """Verify `sizeasmass` input."""
        return check_type(sizeasmass, bool, 'sizeasmass')

    @abstractmethod
    def _pdf(self,
             size: Union[float, ndarray],
             order: int,
             sizeasmass: bool = False
             ) -> Union[float, ndarray[Any, dtype[float64]]]:
        """$m$-th order chain-length / molar mass probability density
        function."""
        pass

    @abstractmethod
    def _cdf(self,
             size: Union[float, ndarray],
             order: int,
             sizeasmass: bool = False
             ) -> Union[float, ndarray[Any, dtype[float64]]]:
        """$m$-th order chain-length / molar mass cumulative density
        function."""
        pass

    @abstractmethod
    def _moment_mass(self,
                     order: int,
                     shift: int = 0
                     ) -> float:
        """Molar-mass moment of the _number_ probability density/mass
        function.
        """
        pass

    @abstractmethod
    def _xrange_plot(self,
                     sizeasmass: bool
                     ) -> ndarray:
        """Default chain-length or molar mass range for distribution plots.
        """
        pass


class IndividualDistribution(GeneralDistribution):
    """Abstract class for all individual chain-length distributions."""

    _continuous = True

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
        r"""Number-average molar mass of the repeating units, $M_0=M_n/DP_n$.
        """
        return self.__M0  # type: ignore

    @M0.setter
    def M0(self, M0: float):
        self.__M0 = check_bounds(M0, 0.0, np.Inf, 'M0')

    def _moment_mass(self,
                     order: int,
                     shift: int = 0
                     ) -> float:
        """Molar-mass moments of the _number_ probability density/mass
        function.
        """
        return self._moment_length(order)*self.M0**(order-shift)

    def _pdf(self, size, order, sizeasmass):
        """$m$-th order chain-length / molar mass probability density
        function."""
        factor = 1
        if sizeasmass:
            size = size/self.M0
            factor = self.M0
        return self._pdf0_length(size) * size**order \
            / (self._moment_length(order)*factor)

    def _cdf(self, size, order, sizeasmass):
        """$m$-th order chain-length / molar mass probability cumulative
        function."""
        if sizeasmass:
            size = size/self.M0
        return self._cdf_length(size, order)

    def _xrange_plot(self, sizeasmass):
        """Default chain-length or molar mass range for distribution plots.
        """
        xrange = self._range_length_default
        if sizeasmass:
            xrange = xrange*self.M0
        return xrange

    def _update_internal_parameters(self) -> None:
        """Update internal parameters that depend on the defining parameters
        of the distribution (DPn, PDI, ...).
        """
        # clear cached properties
        self._moment_length.cache_clear()
        self.__dict__.pop('_range_length_default', None)

    @property
    def _range_length_default(self) -> ndarray:
        """Default chain-length range for distribution plots.

        This implementation is just a fallback solution. More specific
        implementations should be made in subclasses.
        """
        return np.asarray([1, self.DPz])

    @vectorize
    def _moment_quadrature(self,
                           xa: float,
                           xb: float,
                           order: int
                           ) -> float:
        r"""Evaluate the partial moment sum / integral:

        $$ F(a,b) = \frac{\sum_{k=a}^{b}k^m\,p(k)}{\lambda_m} $$

        or

        $$ F(a,b) = \frac{\int_{a}^{b}x^m\,p(x)\mathrm{d}x}{\lambda_m} $$

        where $m$ is the moment order.

        Parameters
        ----------
        xa : float
            lower limit of the partial sum / integral.
        xb : float
            upper limit of the partial sum / integral.
        order : int
            moment order.

        Returns
        -------
        float
            numerical approximation to partial moment.
        """
        # cast to float is required to use mpmath with ufuncs
        def f(z): return (z**order)*self._pdf0_length(float(z))

        if self._continuous:
            result, _ = integrate.quad(f, xa, xb, epsrel=1e-4)
        else:
            result = float(mpmath.nsum(f, [max(1, int(xa)), xb]))
        return result

    def _cdf_length(self,
                    x: Union[float, ndarray],
                    order: int
                    ) -> Union[float, ndarray[Any, dtype[float64]]]:
        """Cumulative density function.

        This implementation is a general low-performance fallback solution.
        Preferably, child classes should implement a specific (e.g., analytic)
        method delivering the cumulative density function for the number _and_
        mass distribution.

        Parameters
        ----------
        x : float | ndarray
            Chain length.
        order : int
            Order of the distribution (0: number, 1: mass).

        Returns
        -------
        float | ndarray
            Cumulative density value.
        """
        return self._moment_quadrature(np.zeros_like(x), x, order) \
            / self._moment_length(order)

    @functools.cache
    def _moment_length(self,
                       order: int
                       ) -> float:
        """Chain-length moments of the _number_ probability density/mass
        function.

        This implementation is a general low-performance fallback solution.
        Preferably, child classes should implement a specific (e.g., analytic)
        method delivering the first four moments (0-3) of the number pdf.

        Parameters
        ----------
        order : int
            Order of the moment.

        Returns
        -------
        float
            Moment of the number distribution.
        """
        # print("Warning: using low performance 'moment_length' method.")
        return self._moment_quadrature(0, np.Inf, order)

    @abstractmethod
    def _pdf0_length(self,
                     k: Union[float, ndarray]
                     ) -> Union[float, ndarray[Any, dtype[float64]]]:
        """Probability density/mass function.

        Each child class must implement a method delivering the _number_
        probability density/mass function.

        Parameters
        ----------
        k : float | ndarray
            Chain length.

        Returns
        -------
        float | ndarray
            Probability density/mass value.
        """
        pass


class AnalyticalDistribution(IndividualDistribution):
    """Abstract class for all analytical chain-length distributions."""

    # (min-DPn, max-DPn)
    _pbounds = ((2,), (np.Inf,))
    _ppf_bounds = (1e-4, 0.9999)

    def __init__(self,
                 DPn: int,
                 M0: float = 100.0,
                 name: str = ''
                 ) -> None:
        """Initialize 1-parameter analytical distribution.

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

    @property
    def DPn(self) -> float:
        r"""Number-average degree of polymerization, $DP_n$."""
        return self.__DPn  # type: ignore

    @DPn.setter
    def DPn(self, DPn: int):
        self.__DPn = check_bounds(DPn,
                                  self._pbounds[0][0], self._pbounds[1][0],
                                  'DPn')
        self._update_internal_parameters()

    @property
    def _pvalues(self) -> tuple:
        """Value(s) defining the chain-length pdf. Used for generalized access
        by fit method."""
        return (self.DPn,)

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
        return self._random_length(size)

    @abstractmethod
    def _random_length(self,
                       size: Union[int, tuple, None],
                       ) -> Union[int, ndarray[Any, dtype[int64]]]:
        """Random chain-length generator.

        Each child class must implement a method to generate random chain
        lengths according to the statistics of the corresponding number
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
        pass


class AnalyticalDistributionP1(AnalyticalDistribution):
    """Abstract class for 1-parameter analytical chain-length distributions."""
    pass


class AnalyticalDistributionP2(AnalyticalDistribution):
    """Abstract class for 2-parameter analytical chain-length distributions."""

    # ((min-DPn, min-PDI), (max-DPn, max-PDI))
    _pbounds = ((2, 1.000001), (np.Inf, np.Inf))

    def __init__(self,
                 DPn: int,
                 PDI: float,
                 M0: float = 100.0,
                 name: str = ''
                 ) -> None:
        """Initialize 2-parameter analytical chain-length distribution.

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
        self.__PDI = check_bounds(PDI,
                                  self._pbounds[0][1], self._pbounds[1][1],
                                  'PDI')
        self._update_internal_parameters()

    @property
    def _pvalues(self) -> tuple:
        """Value(s) defining the chain-length pdf."""
        return (self.DPn, self.PDI)


class MixtureDistribution(GeneralDistribution):
    """Mixture chain-length distribution."""

    def __init__(self,
                 components: dict[IndividualDistribution, float],
                 name: str = ''
                 ) -> None:

        self._components = components
        self.name = name

    def __add__(self, other):
        if isinstance(other, MixtureDistribution):
            return MixtureDistribution(add_dicts(self._components,
                                                 other._components),
                                       name=self.name+'+'+other.name)
        elif isinstance(other, IndividualDistribution):
            return MixtureDistribution(add_dicts(self._components, {other: 1}),
                                       name=self.name+'+'+other.name)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    @property
    def _molefrac(self) -> ndarray:
        """Mole fraction of each individual distribution."""
        xn = np.empty(len(self._components))
        for i, (d, w) in enumerate(self._components.items()):
            xn[i] = w/d.Mn
        xn[:] /= xn.sum()
        return xn

    def _moment_mass(self, order, shift=0):
        xn = self._molefrac
        result = 0
        for i, d in enumerate(self._components.keys()):
            result += xn[i]*d._moment_mass(order, shift)
        return result

    def _pdf(self, size, order, sizeasmass):
        xn = self._molefrac
        numerator = 0
        denominator = 0
        for i, d in enumerate(self._components.keys()):
            term1 = xn[i]*d._moment_mass(order)
            term2 = term1*d._pdf(size, order, sizeasmass)
            denominator += term1
            numerator += term2
        return numerator/denominator

    def _cdf(self, size, order, sizeasmass):
        xn = self._molefrac
        numerator = 0
        denominator = 0
        for i, d in enumerate(self._components.keys()):
            term1 = xn[i]*d._moment_mass(order)
            term2 = term1*d._cdf(size, order, sizeasmass)
            denominator += term1
            numerator += term2
        return numerator/denominator

    def _xrange_plot(self, sizeasmass):
        """Default chain-length or molar mass range for distribution plots.
        """
        xrange = np.empty(2)
        xrange[0] = min([d._xrange_plot(sizeasmass)[0]
                        for d in self._components.keys()])
        xrange[1] = max([d._xrange_plot(sizeasmass)[1]
                         for d in self._components.keys()])
        return xrange

# %%


def plotdists(dists: list[GeneralDistribution],
              **kwargs):

    # Create matplot objects
    fig, ax = plt.subplots(1, 1)
    # fig.suptitle(f"Distribution: {self.name}")

    # Build plots sequentially
    for d in dists:
        d.plot(ax=ax, **kwargs)

    return (fig, ax)

# %%
