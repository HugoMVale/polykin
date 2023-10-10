# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.utils import check_in_set, check_valid_range, check_bounds, \
    convert_check_temperature, eps, \
    FloatOrArray, FloatOrArrayLike

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from abc import ABC, abstractmethod
from typing import Optional, Literal, Any


class PropertyEquation(ABC):
    r"""_Abstract_ class for all property equaitons, $p(...)$."""

    name: str
    unit: str
    symbol: str

    def __init__(self,
                 unit: str,
                 symbol: str,
                 name: str
                 ) -> None:
        self.unit = unit
        self.symbol = symbol
        self.name = name

    @abstractmethod
    def __call__(self, *args) -> FloatOrArray:
        pass

    @abstractmethod
    def eval(self, *args) -> FloatOrArray:
        pass


class PropertyEquationT(PropertyEquation):
    r"""_Abstract_ temperature-dependent property equation, $p(T)$"""

    Trange: tuple[FloatOrArray, FloatOrArray]
    _params: tuple[tuple[str, ...], tuple[str, ...]]
    _shape: Optional[tuple]

    def __init__(self,
                 Trange: tuple[FloatOrArray, FloatOrArray],
                 unit: str,
                 symbol: str,
                 name: str
                 ) -> None:

        check_bounds(Trange[0], 0, np.inf, 'Tmin')
        check_bounds(Trange[1], 0, np.inf, 'Tmax')
        check_bounds(Trange[1]-Trange[0], eps, np.inf, 'Tmax-Tmin')
        self.Trange = Trange
        self._shape = None
        super().__init__(unit, symbol, name)

    def __call__(self,
                 T: FloatOrArrayLike,
                 Tunit: Literal['C', 'K'] = 'K'
                 ) -> FloatOrArray:
        r"""Evaluate property equation at given temperature, including unit
        conversion and range check.

        Parameters
        ----------
        T : FloatOrArrayLike
            Temperature.
            Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        FloatOrArray
            Correlation value.
        """
        TK = convert_check_temperature(T, Tunit, self.Trange)
        return self.eval(TK)

    def __repr__(self) -> str:
        s1 = (
            f"name:        {self.name}\n"
            f"symbol:      {self.symbol}\n"
            f"unit:        {self.unit}\n"
            f"Trange [K]:  {self.Trange}"
        )
        s2 = ""
        params = getattr(self, '_params')
        if params is not None:
            for p in params[0] + params[1]:
                pvalue = getattr(self, p)
                s2 += "\n" + p + ":" + " "*(12 - len(p)) + f"{pvalue}"
        return s1 + s2

    @abstractmethod
    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate property equation at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Property value.
        """
        pass

    def plot(self,
             kind: Literal['linear', 'semilogy', 'Arrhenius'] = 'linear',
             Trange: Optional[tuple[float, float]] = None,
             Tunit: Literal['C', 'K'] = 'K',
             title: Optional[str] = None,
             axes: Optional[Axes] = None,
             return_objects: bool = False
             ) -> Any:
        """Plot quantity as a function of temperature.

        Parameters
        ----------
        kind : Literal['linear', 'semilogy', 'Arrhenius']
            Kind of plot to be generated.
        Trange : tuple[float, float] | None
            Temperature range for x-axis. If `None`, the validity range
            (Tmin, Tmax) will be used. If no validity range was defined, the
            range will default to 0-100°C.
        Tunit : Literal['C', 'K']
            Temperature unit.
        title : str | None
            Title of plot. If `None`, the object name will be used.
        axes : Axes | None
            Matplotlib Axes object.
        return_objects : bool
            If `True`, the Figure and Axes objects are returned (for saving or
            further manipulations).

        Returns
        -------
        tuple[Figure | None, Axes] | None
            Figure and Axes objects if return_objects is `True`.    
        """

        # Check inputs
        check_in_set(kind, {'linear', 'semilogy', 'Arrhenius'}, 'kind')
        check_in_set(Tunit, {'K', 'C'}, 'Tunit')
        if Trange is not None:
            Trange_min = 0.
            if Tunit == 'C':
                Trange_min = -273.15
            check_valid_range(Trange, Trange_min, np.inf, 'Trange')

        # Plot objects
        if axes is None:
            fig, ax = plt.subplots()
            self.fig = fig
            if title is None:
                title = self.name
            if title:
                fig.suptitle(title)
        else:
            fig = None
            ax = axes

        # Plot labels
        Tunit_range = Tunit
        if kind == 'Arrhenius':
            Tunit = 'K'
        Tsymbol = Tunit
        if Tunit == 'C':
            Tsymbol = '°' + Tunit

        xlabel = fr"$T$ [{Tsymbol}]"
        ylabel = fr"${self.symbol}$ [{self.unit}]"
        label = None
        if fig is None:
            label = ylabel
            ylabel = "$y(T)$"
        if kind == 'Arrhenius':
            xlabel = r"$1/T$ [" + Tsymbol + r"$^{-1}$]"
            ylabel = r"$\ln$" + ylabel
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        # x-axis vector
        if Trange is not None:
            if Tunit_range == 'C':
                Trange = tuple(np.asarray(Trange) + 273.15)
        else:
            Trange = (np.min(self.Trange[0]), np.max(self.Trange[1]))
            if Trange == (0.0, np.inf):
                Trange = (273.15, 373.15)

        try:
            shape = getattr(self, '_shape')
        except AttributeError:
            shape = None
        if shape is not None:
            print("Plot method not yet implemented for array-like equations.")
        else:
            TK = np.linspace(Trange[0], Trange[1], 100)
            y = self.__call__(TK, 'K')
            TC = TK - 273.15
            if Tunit == 'C':
                T = TC
            else:
                T = TK
            if kind == 'linear':
                ax.plot(T, y, label=label)
            elif kind == 'semilogy':
                ax.semilogy(T, y, label=label)
            elif kind == 'Arrhenius':
                ax.semilogy(1/TK, y, label=label)

        if fig is None:
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        if return_objects:
            return (fig, ax)


# %% Functions


def plotequations(eqs: list[PropertyEquationT],
                  kind: Literal['linear', 'semilogy', 'Arrhenius'] = 'linear',
                  title: Optional[str] = None,
                  **kwargs
                  ) -> Figure:
    """Plot a list of temperature-dependent property equations in a combined
    plot.

    Parameters
    ----------
    eqs : list[PropertyEquationT]
        List of property equations to be ploted together.
    kind : Literal['linear', 'semilogy', 'Arrhenius']
        Kind of plot to be generated.
    title : str | None
        Title of plot.

    Returns
    -------
    Figure
        Matplotlib Figure object holding the combined plot.
    """

    # Create matplotlib objects
    fig, ax = plt.subplots()

    # Title
    if title is None:
        title = "Equation overlay"
    fig.suptitle(title)

    # Draw plots sequentially
    for item in eqs:
        item.plot(kind=kind, axes=ax, **kwargs)

    return fig
