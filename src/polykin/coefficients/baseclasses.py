from polykin.utils import check_in_set, check_valid_range, \
    convert_check_temperature, \
    FloatOrArray, FloatOrArrayLike
from polykin.base import Base

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from abc import ABC, abstractmethod
from typing import Union, Literal


class Coefficient(Base, ABC):
    """_Abstract_ class for all coefficients, c(...)."""

    def __init__(self) -> None:
        self._shape = None

    @property
    def shape(self) -> Union[tuple[int, ...], None]:
        """Shape of underlying coefficient array."""
        return self._shape

    @abstractmethod
    def __call__(self, *args) -> FloatOrArray:
        pass

    @abstractmethod
    def eval(self, *args) -> FloatOrArray:
        pass


class CoefficientX1(Coefficient):
    r"""_Abstract_ class for 1-argument coefficient, $c(x)$."""
    pass


class CoefficientX2(Coefficient):
    r"""_Abstract_ class for 2-arguments coefficient, $c(x, y)$."""
    pass


class CoefficientT(CoefficientX1):
    r"""_Abstract_ temperature-dependent coefficient, $c(T)$"""

    Tmin: FloatOrArray
    Tmax: FloatOrArray
    unit: str
    symbol: str

    def __call__(self,
                 T: FloatOrArrayLike,
                 Tunit: Literal['C', 'K'] = 'C'
                 ) -> FloatOrArray:
        r"""Evaluate coefficient at given temperature, including unit
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
            Coefficient value.
        """
        TK = convert_check_temperature(T, Tunit, self.Tmin, self.Tmax)
        return self.eval(TK)

    @abstractmethod
    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate coefficient at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """
        pass

    def plot(self,
             kind: Literal['linear', 'semilogy', 'Arrhenius'] = 'linear',
             Trange: Union[tuple[float, float], None] = None,
             Tunit: Literal['C', 'K'] = 'C',
             title: Union[str, None] = None,
             axes: Union[Axes, None] = None
             ) -> None:
        """Plot the coefficient as a function of temperature.

        Parameters
        ----------
        kind : Literal['linear', 'semilogy', 'Arrhenius']
            Kind of plot to be generated.
        Trange : tuple[float, float] | None
            Temperature range for x-axis. If `None`, the validity range
            (Tmin, Tmax) will be used. If no validity range was defined, the
            range will fall back to 0-100°C.
        Tunit : Literal['C', 'K']
            Temperature unit.
        title : str | None
            Title of plot. If `None`, the object name will be used.
        axes : Axes | None
            Matplotlib Axes object.
        """

        # Check inputs
        check_in_set(kind, {'linear', 'semilogy', 'Arrhenius'}, 'kind')
        check_in_set(Tunit, {'K', 'C'}, 'Tunit')
        if Trange is not None:
            check_valid_range(Trange, 0., np.inf, 'Trange')

        # Plot objects
        if axes is None:
            ext_mode = False
            fig, ax = plt.subplots()
            self.fig = fig
            if title is None:
                title = self.name
            if title:
                fig.suptitle(title)
        else:
            ext_mode = True
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
        if ext_mode:
            label = ylabel
            ylabel = "$c(T)$"
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
            Trange = (np.min(self.Tmin), np.max(self.Tmax))
            if Trange == (0.0, np.inf):
                Trange = (273.15, 373.15)

        if self._shape:
            print("Plot method not yet implemented for array-like coefficients.")
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

        if ext_mode:
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        return None


# %% Functions


def plotcoeffs(coeffs: list[CoefficientT],
               kind: Literal['linear', 'semilogy', 'Arrhenius'] = 'linear',
               title: Union[str, None] = None,
               **kwargs
               ) -> Figure:
    """Plot a list of temperature-dependent coefficients in a combined plot.

    Parameters
    ----------
    coeffs : list[CoefficientT]
        List of coefficients to be ploted together.
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
        title = "Coefficient overlay"
    fig.suptitle(title)

    # Draw plots sequentially
    for c in coeffs:
        c.plot(kind=kind, axes=ax, **kwargs)

    return fig
