# %% Base distribution classes

from polykin.base import Base
from polykin.utils import check_bounds, check_type, check_in_set, \
    custom_error, add_dicts, vectorize, FloatOrArrayLike, FloatOrArray

from math import log10
import numpy as np
import mpmath
from numpy import int64, float64, dtype, ndarray
from scipy import integrate
import matplotlib.pyplot as plt
from typing import Any, Literal, Union
from abc import ABC, abstractmethod
import functools

# %% Types

Kind = Literal['number', 'mass', 'gpc']

# %% Classes


class GeneralDistribution(Base, ABC):
    """Abstract class for all chain-length distributions."""

    kind_order = {'number': 0, 'mass': 1, 'gpc': 2}
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

    def __lt__(self, other) -> bool:
        if isinstance(other, GeneralDistribution):
            return self.Mw < other.Mw
        else:
            return NotImplemented

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
            size: FloatOrArrayLike,
            kind: Kind = 'mass',
            sizeasmass: bool = False,
            ) -> FloatOrArray:
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
        self._verify_sizeasmass(sizeasmass)
        order = self.kind_order[self._verify_kind(kind)]
        # Convert list to ndarray
        if isinstance(size, list):
            size = np.asarray(size)
        # Math is done by the corresponding subclass method
        return self._pdf(size, order, sizeasmass)

    def cdf(self,
            size: FloatOrArrayLike,
            kind: Literal['number', 'mass'] = 'mass',
            sizeasmass: bool = False,
            ) -> FloatOrArray:
        r"""Evaluate the cumulative distribution function:

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
        kind = self._verify_kind(kind)
        if kind == 'gpc':
            custom_error('kind', kind, ValueError,
                         "Please use `mass` instead.")
        order = self.kind_order[kind]
        self._verify_sizeasmass(sizeasmass)
        # Convert list to ndarray
        if isinstance(size, list):
            size = np.asarray(size)
        # Math is done by the corresponding subclass method
        result = self._cdf(size, order, sizeasmass)
        return result

    def plot(self,
             kinds: Union[Kind, list[Kind]] = 'mass',
             sizeasmass: bool = False,
             xscale: Literal['auto', 'linear', 'log'] = 'auto',
             xrange: Union[list[float], tuple[float, float],
                           ndarray[Any, dtype[float64]]] = [],
             cdf: Literal[0, 1, 2] = 0,
             title: Union[str, None] = None,
             axes: Union[list[plt.Axes], None] = None,
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
        title: Union[str, None]
            Title
        axes : Union[plt.Axes, None]
            Matplotlib Axes object.
        """
        # Check inputs
        kinds = self._verify_kind(kinds, accept_list=True)
        self._verify_sizeasmass(sizeasmass)
        check_in_set(xscale, {'linear', 'log', 'auto'}, 'xscale')
        check_in_set(cdf, {0, 1, 2}, 'cdf')
        if isinstance(kinds, str):
            kinds = [kinds]

        # x-axis scale
        if xscale == 'auto' and set(kinds) == {'gpc'}:
            xscale = 'log'
        elif xscale == 'log':
            pass
        else:
            xscale = 'linear'

        # x-axis range
        if not (len(xrange) == 2 and xrange[1] > xrange[0]):
            xrange = np.array(self._xrange_plot(sizeasmass))
            if xscale == 'log' and log10(xrange[1]/xrange[0]) > 3 and \
                    isinstance(self, (AnalyticalDistribution,
                                      MixtureDistribution)):
                xrange[1] *= 10
        else:
            xrange = np.asarray(xrange)

        # x-axis vector
        npoints = 200
        if isinstance(self, MixtureDistribution):
            npoints += 100*(len(self.components)-1)
        if xscale == 'log':
            x = np.geomspace(*xrange, npoints)  # type: ignore
        else:
            x = np.linspace(*xrange, npoints)  # type: ignore

        # x-axis label
        if sizeasmass:
            label_x = f"Molar mass [{self.units['molar_mass']}]"
        else:
            label_x = "Chain length"

        # Create axis if none is provided
        if axes is None:
            ext_mode = False
            fig, ax = plt.subplots(1, 1)
            self.fig = fig
            if title is None:
                title = f"Distribution: {self.name}"
            fig.suptitle(title)
            if cdf == 2:
                ax2 = ax.twinx()
        else:
            ext_mode = True
            ax = axes[0]
            if cdf == 2:
                ax2 = axes[1]

        # y-values
        for kind in kinds:
            if cdf != 1:
                y1 = self.pdf(x, kind=kind, sizeasmass=sizeasmass)
            if cdf > 0:
                if kind == 'gpc':
                    _kind = 'mass'
                else:
                    _kind = kind
                y2 = self.cdf(x, kind=_kind, sizeasmass=sizeasmass)
            if cdf == 1:
                y1 = y2
            if ext_mode:
                label = self.name
                if label == '':
                    label = '?'
            else:
                label = kind
            ax.plot(x, y1, label=label)
            if cdf == 2:
                ax2.plot(x, y2, linestyle='--')

        # y-axis and labels
        label_y_pdf = 'Relative abundance'
        label_y_cdf = 'Cumulative probability'
        bbox_to_anchor = (1.05, 1.0)
        if cdf == 0:
            label_y1 = label_y_pdf
        elif cdf == 1:
            label_y1 = label_y_cdf
        elif cdf == 2:
            label_y1 = label_y_pdf
            label_y2 = label_y_cdf
            ax.set_ylabel(label_y1)
            ax2.set_ylabel(label_y2)
            bbox_to_anchor = (1.1, 1.0)
        else:
            raise ValueError
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y1)
        ax.set_xscale(xscale)
        ax.grid(True)
        ax.legend(bbox_to_anchor=bbox_to_anchor, loc="upper left")

        return None

    @classmethod
    def _verify_kind(cls, kind, accept_list=False):
        """Verify `kind` input."""
        if isinstance(kind, str):
            kind = kind.lower()
        elif isinstance(kind, list) and accept_list:
            check_type(kind, str, 'kind', check_inside=True)
            kind = [item.lower() for item in kind]
        else:
            if accept_list:
                valid_types = (str, list)
            else:
                valid_types = (str,)
            check_type(kind, valid_types, 'kind')
        return check_in_set(kind, set(cls.kind_order.keys()), 'kind')

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
             size: FloatOrArray,
             order: int,
             sizeasmass: bool = False
             ) -> FloatOrArray:
        """$m$-th order chain-length / molar mass cumulative distribution
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
            if other > 0:
                return MixtureDistribution({self: other}, name=self.name)
            else:
                return MixtureDistribution({})
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
            result, _ = integrate.quad(f, xa, xb, limit=50, epsrel=1e-4)
        else:
            result = float(mpmath.nsum(f, [max(1, int(xa)), xb]))
        return result

    def _cdf_length(self,
                    x: FloatOrArray,
                    order: int
                    ) -> FloatOrArray:
        """Cumulative distribution function.

        This implementation is a general low-performance fallback solution.
        Preferably, child classes should implement a specific (e.g., analytic)
        method delivering the cumulative distribution function for the number
        _and_ mass distribution.

        Parameters
        ----------
        x : float | ndarray
            Chain length.
        order : int
            Order of the distribution (0: number, 1: mass).

        Returns
        -------
        float | ndarray
            Cumulative distribution value.
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
                     k: FloatOrArray
                     ) -> FloatOrArray:
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

        self.__components = components
        self.name = name

    def __add__(self, other):
        if isinstance(other, MixtureDistribution):
            return MixtureDistribution(add_dicts(self.components,
                                                 other.components),
                                       name=self.name+'+'+other.name)
        elif isinstance(other, IndividualDistribution):
            return MixtureDistribution(add_dicts(self.components, {other: 1}),
                                       name=self.name+'+'+other.name)
        elif isinstance(other, (int, float)):
            return self
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __iter__(self):
        return self.components

    def __contains__(self, component):
        return (component in self.components)

    def __str__(self):
        if len(self.components) > 0:
            return super().__str__()
        else:
            return 'empty'

    @property
    def components(self) -> dict:
        """Individual components of the mixture distribution.

        Returns
        -------
        dict[IndividualDistribution: float]
            Dictionary of individual distributions and corresponding mass
            weight.
        """
        return self.__components

    @property
    def components_table(self) -> Union[str, None]:
        """Table of individual components of the mixture distribution.

        Returns
        -------
        str
            Table with key properties of each component.
        """
        result = 'empty'
        if len(self.components) > 0:
            spacer = f"{' '*3}"
            header = [f"{'#':>2}", f"{'Weight':>6}", f"{'Distribution':>12}",
                      f"{'DPn':>8}", f"{'DPw':>8}", f"{'PDI':>4}"]
            header = (spacer).join(header)
            ruler = f"{'-'*len(header)}"
            table = [header, ruler]
            for i, (d, w) in enumerate(self.components.items()):
                row = [f"{i+1:2}", f"{w:>6.3f}",
                       f"{d.__class__.__name__:>12}",
                       f"{d.DPn:>4.2e}", f"{d.DPw:>4.2e}", f"{d.PDI:>4.2f}"]
                table.append((spacer).join(row))
            result = ("\n").join(table)
        return result

    @property
    def _molefrac(self) -> ndarray:
        """Mole fraction of each individual distribution."""
        xn = np.empty(len(self.components))
        for i, (d, w) in enumerate(self.__iter__().items()):
            xn[i] = w/d.Mn
        xn[:] /= xn.sum()
        return xn

    def _moment_mass(self, order, shift=0):
        xn = self._molefrac
        result = 0
        for i, d in enumerate(self.__iter__()):
            result += xn[i]*d._moment_mass(order, shift)
        return result

    def _pdf(self, size, order, sizeasmass):
        xn = self._molefrac
        numerator = 0
        denominator = 0
        for i, d in enumerate(self.__iter__()):
            term1 = xn[i]*d._moment_mass(order)
            term2 = term1*d._pdf(size, order, sizeasmass)
            denominator += term1
            numerator += term2
        return numerator/denominator

    def _cdf(self, size, order, sizeasmass):
        xn = self._molefrac
        numerator = 0
        denominator = 0
        for i, d in enumerate(self.__iter__()):
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
                        for d in self.__iter__()])
        xrange[1] = max([d._xrange_plot(sizeasmass)[1]
                         for d in self.__iter__()])
        return xrange

# %% Aux functions


def plotdists(dists: list[GeneralDistribution],
              kind: Literal['number', 'mass', 'gpc'],
              title: Union[str, None] = None,
              **kwargs):

    # Check input
    kind = GeneralDistribution._verify_kind(kind)

    # Create matplot and axes objects
    fig, ax = plt.subplots(1, 1)
    if kwargs.get('cdf', 1) == 2:
        ax.twinx()

    # Title
    titles = {'number': 'Number', 'mass': 'Mass', 'gpc': 'GPC'}
    if title is None:
        title = f"{titles.get(kind,'')} distributions"
    fig.suptitle(title)

    # Build plots sequentially
    for d in dists:
        d.plot(kinds=kind, axes=fig.axes, **kwargs)

    return fig
