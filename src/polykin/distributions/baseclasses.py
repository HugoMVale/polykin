# %% Base distribution clases

from polykin.base import Base
from polykin.utils import check_bounds, check_type, check_in_set, add_dicts

import numpy as np
from numpy import int64, float64, dtype, ndarray
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import Any, Literal, Union
from abc import ABC, abstractmethod


class GeneralDistribution(Base, ABC):
    """Abstract class for all chain-length distributions."""

    typenames = {'number': 0, 'mass': 1, 'gpc': 2}

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

    @ property
    def M0(self) -> float:
        """Number-average molar mass of the repeating units, $M_0=M_n/DP_n$."""
        return self.Mn / self.DPn

    def pdf(self,
            size: Union[float, list, ndarray],
            type: Literal['number', 'mass', 'gpc'] = 'mass',
            sizeasmass: bool = False,
            ) -> Union[float, ndarray[Any, dtype[float64]]]:
        r"""Evaluate the probability density function, $p(k)$.

        Parameters
        ----------
        size : int | float | list | ndarray
            Chain length or molar mass.
        type : Literal['number', 'mass', 'gpc']
            Type of distribution.
        sizeasmass : bool
            Switch between chain-*length* (if `False`) or molar *mass*
            (if `True`) size.

        Returns
        -------
        float | ndarray
            Probability density.
        """
        # Check inputs
        self._verify_type(type)
        self._verify_sizeasmass(sizeasmass)
        order = self.typenames[type]
        # Convert list to ndarray
        if isinstance(size, list):
            size = np.asarray(size)
        # Math is done by the corresponding subclass method
        return self._pdf(size, order, sizeasmass)

    def cdf(self,
            size: Union[float, list, ndarray],
            type: Literal['number', 'mass'] = 'mass',
            sizeasmass: bool = False,
            ) -> Union[float, ndarray[Any, dtype[float64]]]:
        r"""Evaluate the cumulative density function:

        $$ F(s) = \frac{\sum_{k=1}^{s}k^m\,p(k)}{\lambda_m} $$

        or

        $$ F(s) = \frac{1}{\lambda_m} {\int_{0}^{s}x^m\,p(x)\mathrm{d}x} $$

        Parameters
        ----------
        size : int | float | ArrayLike
            Chain length or molar mass.
        type : Literal['number', 'mass']
            Type of distribution.
        sizeasmass : bool
            Switch between chain-*length* (if `False`) or molar *mass*
            (if `True`) size.

        Returns
        -------
        float | ndarray
            Cumulative probability.
        """
        # Check inputs
        check_in_set(type, {'number', 'mass'}, 'type')
        self._verify_sizeasmass(sizeasmass)
        order = self.typenames[type]
        # Convert list to ndarray
        if isinstance(size, list):
            size = np.asarray(size)
        # Math is done by the corresponding subclass method
        return self._cdf(size, order, sizeasmass)

    def plot(self,
             type: Literal['number', 'mass', 'gpc'] = 'mass',
             sizeasmass: bool = False,
             xscale: Literal['auto', 'linear', 'log'] = 'auto',
             xrange: Union[list, ndarray] = [],
             cdf: bool = False,
             ax=None
             ) -> None:
        """Plot the chain-length distribution.

        Parameters
        ----------
        type : Literal['number', 'mass', 'gpc']
            Type of distribution.
        sizeasmass : bool
            Switch between chain-*length* (if `False`) or molar *mass*
            (if `True`) size.
        xscale : Literal['auto', 'linear', 'log']
            x-axis scale.
        xrange : Union[list, ndarray]
            x-axis range.
        cdf : bool
            Switch between differential (if `False`) or cumulative (if `True`)
            density function.
        ax : matplotlib.axes
            Matplotlib axes object.

        Returns
        -------
        matplotlib.axes
            Matplotlib axes object.

        """
        # Check inputs
        self._verify_type(type)
        self._verify_sizeasmass(sizeasmass)
        check_in_set(xscale, {'linear', 'log', 'auto'}, 'xscale')
        if isinstance(type, str):
            type = [type]

        # Create axis if none is provided
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(f"Distribution: {self.name}")
            self.fig = fig

        # x-axis range
        if not (len(xrange) == 2 and xrange[1] > xrange[0]):
            xrange = self._xrange_plot(sizeasmass)
        npoints = 200
        if isinstance(self, MixtureDistribution):
            npoints += 100*(len(self._components)-1)
        # x-axis vector and scale
        if xscale == 'log' or (xscale == 'auto' and set(type) == {'gpc'}):
            x = np.geomspace(*xrange, npoints)  # type: ignore
            xscale = 'log'
        else:
            x = np.linspace(*xrange, npoints)  # type: ignore
            xscale = 'linear'
        # x-axis label
        if sizeasmass:
            label_x = "Molar mass"
        else:
            label_x = "Chain length"

        # y-axis
        if cdf:
            fdist = self.cdf
            label_y = 'Cumulative probability'
        else:
            fdist = self.pdf
            label_y = 'Relative abundance'
        for item in type:
            if cdf and item == 'gpc':
                item = 'mass'
            y = fdist(x, type=item, sizeasmass=sizeasmass)
            ax.plot(x, y, label=item)

        # Other properties
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        ax.grid(True)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_xscale(xscale)

        return None

    def _verify_type(self, type):
        """Verify `type` input."""
        return check_in_set(type, set(self.typenames.keys()), 'type')

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
        return 0.0

    @abstractmethod
    def _cdf(self,
             size: Union[float, ndarray],
             order: int,
             sizeasmass: bool = False
             ) -> Union[float, ndarray[Any, dtype[float64]]]:
        """$m$-th order chain-length / molar mass cumulative density
        function."""
        return 0.0

    @abstractmethod
    def _moment_mass(self,
                     order: int,
                     shift: int = 0
                     ) -> float:
        """Molar-mass moment of the _number_ probability density/mass
        function.
        """
        return 0.0

    @abstractmethod
    def _xrange_plot(self,
                     sizeasmass: bool
                     ) -> ndarray:
        """Default chain-length or molar mass range for distribution plots.
        """
        return [0, 1]


class IndividualDistribution(GeneralDistribution):
    """Abstract class for all individual chain-length distributions."""

    # (min-DPn, max-DPn)
    _pbounds = ((2,), (np.Inf,))

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
    def DPn(self, DPn: int):
        self.__DPn = check_bounds(DPn,
                                  self._pbounds[0][0], self._pbounds[1][0],
                                  'DPn')
        self._update_internal_parameters()

    def _update_internal_parameters(self):
        pass

    @property
    def _pvalues(self) -> tuple:
        """Value(s) defining the chain-length pdf. Used for generalized access
        by fit method."""
        return (self.DPn,)

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
        return self._pdf_length(size) * size**order \
            / (self._moment_length(order)*factor)

    def _cdf(self, size, order, sizeasmass):
        """$m$-th order chain-length / molar mass probability cumulative
        function."""
        if sizeasmass:
            size = size/self.M0
        return self._cdf_length(size, order)

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

    def _xrange_plot(self, sizeasmass):
        """Default chain-length or molar mass range for distribution plots.
        """
        xrange = np.asarray(self._xrange_length, dtype=np.float64)
        if sizeasmass:
            xrange *= self.M0
        return xrange

    def fit(self,
            size_data: Union[list, ndarray],
            pdf_data: Union[list, ndarray],
            sizeasmass: bool = False,
            type: Literal['number', 'mass', 'gpc'] = 'mass',
            ) -> None:

        # Preprocess data values
        if not (isinstance(size_data, ndarray)):
            size_data = np.asarray(size_data)
        if not (isinstance(pdf_data, ndarray)):
            pdf_data = np.asarray(pdf_data)
        idx_valid = pdf_data > 0.0
        size_data = size_data[idx_valid]
        pdf_data = pdf_data[idx_valid]

        isP2 = isinstance(self, IndividualDistributionP2)

        # Define parametric function
        def f(x, *p):
            self.DPn = p[1]
            if isP2:
                self.PDI = p[2]
            return np.log(p[0]*self.pdf(x, type=type, sizeasmass=sizeasmass))

        # Call fit method
        p0 = (1,) + self._pvalues
        bounds = ((-np.Inf,)+self._pbounds[0], (np.Inf,)+self._pbounds[1])
        solution = curve_fit(f,
                             xdata=size_data,
                             ydata=np.log(pdf_data),
                             p0=p0,
                             bounds=bounds,
                             method='trf',
                             full_output=True)

        if solution[4] > 0:
            # print(self)
            popt = solution[0]
            print(f"scale:  {popt[0]:.2e}")
            print(f"DPn:    {self.DPn:.1f}")
            print(f"PDI:    {self.PDI:.2f}")
        else:
            print("Failed to fit distribution: ", solution[3])

    @abstractmethod
    def _moment_length(self,
                       order: int
                       ) -> float:
        """Chain-length moments of the _number_ probability density/mass 
        function.

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
    def _pdf_length(self,
                    k: Union[float, ndarray]
                    ) -> Union[float, ndarray[Any, dtype[float64]]]:
        """Probability density/mass function.

        Each child class must implement a method to delivering the _number_
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
        return 0.0

    @abstractmethod
    def _cdf_length(self,
                    k: Union[float, ndarray],
                    order: int
                    ) -> Union[float, ndarray[Any, dtype[float64]]]:
        """Cumulative density function.

        Each child class must implement a method to delivering the cumulative
        density function for the number _and_ mass distribution. Both cases
        must be covered, because it is not straightforward to convert from one
        kind of distribution to another.

        Parameters
        ----------
        k : float | ndarray
            Chain length.
        order : int
            Order of the distribution (0: number, 1: mass).

        Returns
        -------
        float | ndarray
            Cumulative density value.
        """
        return 0.0

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
        return 0

    @abstractmethod
    def _xrange_length(self) -> tuple[int, int]:
        """Default chain-length range for distribution plots.
        """
        return (0, 1)


class IndividualDistributionP1(IndividualDistribution):
    """Abstract class for 1-parameter single chain-length distributions."""
    pass


class IndividualDistributionP2(IndividualDistribution):
    """Abstract class for 2-parameter single chain-length distributions."""

    # ((min-DPn, min-PDI), (max-DPn, max-PDI))
    _pbounds = ((2, 1.001), (np.Inf, np.Inf))

    def __init__(self,
                 DPn: int,
                 PDI: float,
                 M0: float = 100.0,
                 name: str = ''
                 ) -> None:
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
                 name: str = ""
                 ) -> None:

        self._components = components
        self.name = name
        self._molefracs = None

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

    def _calc_molefracs(self) -> ndarray:
        """Calculate mole fraction of each individual distribution."""
        if self._molefracs is None:
            x = np.empty(len(self._components))
            for i, (d, w) in enumerate(self._components.items()):
                x[i] = w/d.Mn
            x[:] /= x.sum()
            self._molefracs = x
        return self._molefracs

    def _moment_mass(self, order, shift=0):
        xn = self._calc_molefracs()
        result = 0
        for i, d in enumerate(self._components.keys()):
            result += xn[i]*d._moment_mass(order, shift)
        return result

    def _pdf(self, size, order, sizeasmass):
        xn = self._calc_molefracs()
        numerator = 0
        denominator = 0
        for i, d in enumerate(self._components.keys()):
            term1 = xn[i]*d._moment_mass(order)
            term2 = term1*d._pdf(size, order, sizeasmass)
            denominator += term1
            numerator += term2
        return numerator/denominator

    def _cdf(self, size, order, sizeasmass):
        xn = self._calc_molefracs()
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
