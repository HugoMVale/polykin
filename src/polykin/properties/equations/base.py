# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from numpy import log
from scipy.optimize import curve_fit

from polykin.utils.math import eps
from polykin.utils.tools import (check_bounds, check_in_set, check_valid_range,
                                 convert_check_temperature)
from polykin.utils.types import FloatArray, FloatArrayLike, FloatVectorLike


class PropertyEquation(ABC):
    r"""_Abstract_ class for all property equations, $Y(...)$."""

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


class PropertyEquationT(PropertyEquation):
    r"""_Abstract_ temperature-dependent property equation, $Y(T)$"""

    _pinfo: dict[str, tuple[str, bool]]
    p: dict[str, Any]
    Trange: tuple[Union[float, FloatArray], Union[float, FloatArray]]

    def __init__(self,
                 Trange: tuple[Union[float, FloatArray],
                               Union[float, FloatArray]],
                 unit: str,
                 symbol: str,
                 name: str
                 ) -> None:

        check_bounds(Trange[0], 0, np.inf, 'Tmin')
        check_bounds(Trange[1], 0, np.inf, 'Tmax')
        check_bounds(Trange[1]-Trange[0], eps, np.inf, 'Tmax-Tmin')
        self.Trange = Trange
        super().__init__(unit, symbol, name)

    def __call__(self,
                 T: Union[float, FloatArrayLike],
                 Tunit: Literal['C', 'K'] = 'K'
                 ) -> Union[float, FloatArray]:
        r"""Evaluate property equation at given temperature, including unit
        conversion and range check.

        Parameters
        ----------
        T : float | FloatArrayLike
            Temperature.
            Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        float | FloatArray
            Correlation value.
        """
        TK = convert_check_temperature(T, Tunit, self.Trange)
        return self.equation(TK, **self.p)

    @staticmethod
    @abstractmethod
    def equation(T: Union[float, FloatArray],
                 *args
                 ) -> Union[float, FloatArray]:
        """Property equation, $Y(T,p...)$."""
        pass

    def __repr__(self) -> str:
        string1 = (
            f"name:            {self.name}\n"
            f"symbol:          {self.symbol}\n"
            f"unit:            {self.unit}\n"
            f"Trange [K]:      {self.Trange}"
        )
        string2 = ""
        for pname, pvalue in self.p.items():
            punits = self._pinfo[pname][0]
            if not punits:
                punits = '—'
            punits = punits.replace('#', self.unit)
            string2 += "\n" + f"{pname} [{punits}]:" + \
                " "*(13 - len(pname+punits)) + f"{pvalue}"
        return string1 + string2

    def plot(self,
             kind: Literal['linear', 'semilogy', 'Arrhenius'] = 'linear',
             Trange: Optional[tuple[float, float]] = None,
             Tunit: Literal['C', 'K'] = 'K',
             title: Optional[str] = None,
             axes: Optional[Axes] = None,
             return_objects: bool = False
             ) -> Optional[tuple[Optional[Figure], Axes]]:
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
            if title is None:
                title = self.name
            if title:
                fig.suptitle(title)
            label = None
        else:
            fig = None
            ax = axes
            label = self.name

        # Units and xlabel
        Tunit_range = Tunit
        if kind == 'Arrhenius':
            Tunit = 'K'
        Tsymbol = Tunit
        if Tunit == 'C':
            Tsymbol = '°' + Tunit

        if kind == 'Arrhenius':
            xlabel = r"$1/T$ [" + Tsymbol + r"$^{-1}$]"
        else:
            xlabel = fr"$T$ [{Tsymbol}]"

        # ylabel
        ylabel = fr"${self.symbol}$ [{self.unit}]"
        if axes is not None:
            ylabel0 = ax.get_ylabel()
            if ylabel0 and ylabel not in ylabel0:
                ylabel = ylabel0 + ", " + ylabel

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        # x-axis vector
        if Trange is not None:
            if Tunit_range == 'C':
                Trange = (Trange[0]+273.15, Trange[1]+273.15)
        else:
            Trange = (np.min(self.Trange[0]), np.max(self.Trange[1]))
            if Trange == (0.0, np.inf):
                Trange = (273.15, 373.15)

        try:
            shape = self._shape
        except AttributeError:
            shape = None
        if shape is not None:
            print("Plot method not yet implemented for array-like equations.")
        else:
            TK = np.linspace(*Trange, 100)
            y = self.__call__(TK, 'K')
            T = TK
            if Tunit == 'C':
                T -= 273.15
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

    def fit(self,
            T: FloatVectorLike,
            Y: FloatVectorLike,
            sigmaY: Optional[FloatVectorLike] = None,
            fit_only: Optional[list[str]] = None,
            logY: bool = False,
            plot: bool = True,
            ) -> dict:
        """Fit equation to data using non-linear regression.

        Parameters
        ----------
        T : FloatVector
            Temperature. Unit = K.
        Y : FloatVector
            Property to be fitted. Unit = Any.
        sigmaY : FloatVector | None
            Standard deviation of Y. Unit = [Y].
        fit_only : list[str] | None
            List with name of parameters to be fitted.
        logY : bool
            If `True`, the fit will be done in terms of log(Y).
        plot : bool
            If `True` a plot comparing data and fitted correlation will be
            generated.

        Returns
        -------
        dict
            A dictionary of results with the following keys: 'success',
            'parameters', 'covariance', and 'plot'.
        """

        # Current parameter values
        pdict = self.p.copy()

        # Select parameters to be fitted
        pnames_fit = [name for name, info in self._pinfo.items() if info[1]]
        if fit_only:
            pnames_fit = set(fit_only) & set(pnames_fit)
        p0 = [pdict[pname] for pname in pnames_fit]

        # Fit function
        def ffit(x, *p):
            for pname, pvalue in zip(pnames_fit, p):
                pdict[pname] = pvalue
            Yfit = self.equation(T=x, **pdict)
            if logY:
                Yfit = log(Yfit)
            return Yfit

        solution = curve_fit(ffit,
                             xdata=T,
                             ydata=log(Y) if logY else Y,
                             p0=p0,
                             sigma=sigmaY,
                             absolute_sigma=False,
                             full_output=True)
        result = {}
        result['success'] = bool(solution[4])
        if solution[4]:
            popt = solution[0]
            pcov = solution[1]
            print("Fit successful.")
            for pname, pvalue in zip(pnames_fit, popt):
                print(f"{pname}: {pvalue}")
            print("Covariance:")
            print(pcov)
            result['covariance'] = pcov

            # Update attributes
            self.Trange = (min(T), max(T))
            for pname, pvalue in zip(pnames_fit, popt):
                self.p[pname] = pvalue
            result['parameters'] = pdict

            # plot
            if plot:
                kind = 'semilogy' if logY else 'linear'
                fig, ax = self.plot(kind=kind, return_objects=True)  # ok
                ax.plot(T, Y, 'o', mfc='none')
                result['plot'] = (fig, ax)
        else:
            print("Fit error: ", solution[3])
            result['message'] = solution[3]

        return result


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
    for eq in eqs:
        eq.plot(kind=kind, axes=ax, **kwargs)

    return fig
