# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy import exp
from numpy import log as ln

from polykin.properties.pvt.io import load_PVT_parameters
from polykin.utils.math import eps
from polykin.utils.tools import (
    check_bounds,
    convert_check_pressure,
    convert_check_temperature,
)
from polykin.utils.types import FloatArray, FloatVectorLike

__all__ = ["Tait"]


class Tait:
    r"""Tait equation of state for the specific volume of a liquid.

    This EoS implements the following explicit PVT dependence:

    $$ \hat{v}(T,P) = \hat{v}(T,0)
       \left[1 - C \ln \left( 1 + \frac{P}{B(T)} \right) \right] $$

    with:

    $$ \begin{gather*}
    \hat{v}(T,0) = A_0 + A_1(T - 273.15) + A_2(T - 273.15)^2 \\
    B(T) = B_0\exp\left [-B_1(T - 273.15)\right]
    \end{gather*} $$

    where $\hat{v}$ is the specific volume, $T$ is the absolute temperature, 
    $P$ is the pressure, and $A_i$ and $B_i$ are constant parameters.

    **References**

    *   Danner, Ronald P., and Martin S. High. Handbook of polymer
        solution thermodynamics. John Wiley & Sons, 2010.

    Parameters
    ----------
    A0 : float
        Parameter of equation [m³/kg].
    A1 : float
        Parameter of equation [m³/(kg·K)].
    A2 : float
        Parameter of equation [m³/(kg·K²)].
    B0 : float
        Parameter of equation [Pa].
    B1 : float
        Parameter of equation [K⁻¹].
    Tmin : float
        Lower temperature bound [K].
    Tmax : float
        Upper temperature bound [K].
    Pmin : float
        Lower pressure bound [Pa].
    Pmax : float
        Upper pressure bound [Pa].
    name : str
        Name.
    """

    Trange: tuple[float, float]
    Prange: tuple[float, float]
    symbol = r"$\hat{V}$"
    unit = "m³/kg"

    A0: float
    A1: float
    A2: float
    B0: float
    B1: float

    _C = 0.0894

    def __init__(
        self,
        A0: float,
        A1: float,
        A2: float,
        B0: float,
        B1: float,
        Tmin: float = 0.0,
        Tmax: float = np.inf,
        Pmin: float = 0.0,
        Pmax: float = np.inf,
        name: str = "",
    ) -> None:

        # Check bounds
        check_bounds(A0, 1e-4, 2e-3, "A0")
        check_bounds(A1, 1e-7, 2e-6, "A1")
        check_bounds(A2, -2e-9, 1e-8, "A2")
        check_bounds(B0, 1e7, 1e9, "B0")
        check_bounds(B1, 1e-3, 2e-2, "B1")
        check_bounds(Tmin, 0, np.inf, "Tmin")
        check_bounds(Tmax, 0, np.inf, "Tmax")
        check_bounds(Tmax - Tmin, eps, np.inf, "Tmax-Tmin")
        check_bounds(Pmin, 0, np.inf, "Pmin")
        check_bounds(Pmax, 0, np.inf, "Pmax")
        check_bounds(Pmax - Pmin, eps, np.inf, "Pmax-Pmin")

        self.A0 = A0
        self.A1 = A1
        self.A2 = A2
        self.B0 = B0
        self.B1 = B1
        self.Trange = (Tmin, Tmax)
        self.Prange = (Pmin, Pmax)
        self.name = name

    def __repr__(self) -> str:
        """Return a string representation of the Tait equation object."""
        return (
            f"name         : {self.name}\n"
            f"symbol       : {self.symbol}\n"
            f"unit         : {self.unit}\n"
            f"Trange [K]   : {self.Trange}\n"
            f"Prange [Pa]  : {self.Prange}\n"
            f"A0 [m³/kg]   : {self.A0}\n"
            f"A1 [m³/kg·K] : {self.A1}\n"
            f"A2 [m³/kg·K²]: {self.A2}\n"
            f"B0 [Pa]      : {self.B0}\n"
            f"B1 [K⁻¹]     : {self.B1}"
        )

    def eval(
        self,
        T: float | FloatArray,
        P: float | FloatArray,
    ) -> float | FloatArray:
        r"""Evaluate the specific volume, $\hat{v}$, at given SI conditions
        without unit conversions or checks.

        Parameters
        ----------
        T : float | FloatArray
            Temperature [K].
        P : float | FloatArray
            Pressure [Pa].

        Returns
        -------
        float | FloatArray
            Specific volume [m³/kg].
        """
        TC = T - 273.15
        v0 = self.A0 + self.A1 * TC + self.A2 * TC**2
        B = self._B(T)
        v = v0 * (1 - self._C * ln(1 + P / B))
        return v

    def _B(self, T: float | FloatArray) -> float | FloatArray:
        r"""Parameter B(T).

        Parameters
        ----------
        T : float | FloatArray
            Temperature [K].

        Returns
        -------
        float | FloatArray
            B(T) [Pa].
        """
        return self.B0 * exp(-self.B1 * (T - 273.15))

    def alpha(
        self,
        T: float | FloatArray,
        P: float | FloatArray,
    ) -> float | FloatArray:
        r"""Calculate the thermal expansion coefficient, $\alpha$.

        $$ \alpha = \frac{1}{\hat{v}}
                    \left( \frac{\partial \hat{v}}{\partial T} \right)_{P} $$

        Parameters
        ----------
        T : float | FloatArray
            Temperature [K].
        P : float | FloatArray
            Pressure [Pa].

        Returns
        -------
        float | FloatArray
            Thermal expansion coefficient, $\alpha$ [K⁻¹].
        """
        A0 = self.A0
        A1 = self.A1
        A2 = self.A2
        TC = T - 273.15
        alpha0 = (A1 + 2 * A2 * TC) / (A0 + A1 * TC + A2 * TC**2)
        return alpha0 - P * self.B1 * self.beta(T, P)

    def beta(
        self,
        T: float | FloatArray,
        P: float | FloatArray,
    ) -> float | FloatArray:
        r"""Calculate the isothermal compressibility coefficient, $\beta$.

        $$ \beta = -\frac{1}{\hat{v}}
                    \left( \frac{\partial \hat{v}}{\partial P} \right)_{T} $$

        Parameters
        ----------
        T : float | FloatArray
            Temperature [K].
        P : float | FloatArray
            Pressure [Pa].

        Returns
        -------
        float | FloatArray
            Isothermal compressibility coefficient, $\beta$ [Pa⁻¹].
        """
        B = self._B(T)
        return (self._C / (P + B)) / (1 - self._C * ln(1 + P / B))

    def vs(
        self,
        T: float | FloatVectorLike,
        P: float | FloatVectorLike,
        Tunit: Literal["C", "K"] = "K",
        Punit: Literal["bar", "MPa", "Pa"] = "Pa",
    ) -> float | FloatArray:
        r"""Evaluate the specific volume, $\hat{v}$, at given temperature and
        pressure, including unit conversion and range check.

        Parameters
        ----------
        T : float | FloatArrayLike
            Temperature [`Tunit`].
        P : float | FloatArrayLike
            Pressure [`Punit`].
        Tunit : Literal['C', 'K']
            Temperature unit.
        Punit : Literal['bar', 'MPa', 'Pa']
            Pressure unit.

        Returns
        -------
        float | FloatArray
            Specific volume [m³/kg].
        """
        TK = convert_check_temperature(T, Tunit, self.Trange)
        Pa = convert_check_pressure(P, Punit, self.Prange)
        return self.eval(TK, Pa)

    @classmethod
    def from_database(cls, name: str) -> Tait | None:
        r"""Construct `Tait` with parameters from the database.

        Parameters
        ----------
        name : str
            Polymer code name.
        """
        table = load_PVT_parameters(method=cls.__name__)
        try:
            mask = table.index == name
            parameters = table[mask].iloc[0, :].to_dict()
            return cls(**parameters, name=name)
        except IndexError:
            print(
                f"Error: '{name}' does not exist in polymer database.\n"
                f"Valid names are: {table.index.to_list()}"
            )

    @classmethod
    def get_database(cls) -> pd.DataFrame:
        r"""Get database with parameters for the Tait equation.

        Parameters from Table 3B-1 (p. 41) of Danner and High (2010).

        **References**

        *  Danner, Ronald P., and Martin S. High. Handbook of polymer
            solution thermodynamics. John Wiley & Sons, 2010, p. 41.
        """
        return load_PVT_parameters(method=cls.__name__)
