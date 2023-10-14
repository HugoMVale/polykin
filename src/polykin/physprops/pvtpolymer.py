# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from __future__ import annotations

from polykin.utils import check_bounds, convert_check_temperature, \
    convert_check_pressure, vectorize, \
    FloatOrArray, FloatOrArrayLike, eps
from polykin.physprops.property_equation import PropertyEquation

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Literal
from abc import abstractmethod
from scipy.optimize import root_scalar


__all__ = ['Tait', 'Flory', 'HartmannHaque']

# %% Parameter tables

table_Tait_parameters: Optional[pd.DataFrame] = None


def load_Tait_parameters() -> pd.DataFrame:
    "Load table with Tait parameters."
    global table_Tait_parameters
    if table_Tait_parameters is None:
        filepath = (Path(__file__).parent).joinpath('Tait_parameters.tsv')
        table_Tait_parameters = pd.read_csv(filepath, delim_whitespace=True)
        table_Tait_parameters.set_index("Polymer", inplace=True)
    return table_Tait_parameters


# %% PolymerPVTEquation

class PolymerPVTEquation(PropertyEquation):
    r"""_Abstract_ polymer PVT equation, $\hat{V}(T, P)$"""

    Trange: tuple[FloatOrArray, FloatOrArray]
    Prange: tuple[FloatOrArray, FloatOrArray]
    symbol = r"$\hat{V}$"
    unit = "m³/kg"

    def __init__(self,
                 Tmin: float,
                 Tmax: float,
                 Pmin: float,
                 Pmax: float,
                 name: str
                 ) -> None:

        # Check bounds
        check_bounds(Tmin, 0, np.inf, 'Tmin')
        check_bounds(Tmax, 0, np.inf, 'Tmax')
        check_bounds(Tmax-Tmin, eps, np.inf, 'Tmax-Tmin')
        check_bounds(Pmin, 0, np.inf, 'Pmin')
        check_bounds(Pmax, 0, np.inf, 'Pmax')
        check_bounds(Pmax-Pmin, eps, np.inf, 'Pmax-Pmin')

        self.Trange = (Tmin, Tmax)
        self.Prange = (Pmin, Pmax)
        self.name = name

    def V(self,
          T: FloatOrArrayLike,
          P: FloatOrArrayLike,
          Tunit: Literal['C', 'K'] = 'K',
          Punit: Literal['bar', 'MPa', 'Pa'] = 'Pa'
          ) -> FloatOrArray:
        r"""Evaluate the specific volume, $\hat{V}$, at given temperature and
        pressure, including unit conversion and range check.

        Parameters
        ----------
        T : FloatOrArrayLike
            Temperature.
            Unit = `Tunit`.
        P : FloatOrArrayLike
            Pressure.
            Unit = `Punit`.
        Tunit : Literal['C', 'K']
            Temperature unit.
        Punit : Literal['bar', 'MPa', 'Pa']
            Pressure unit.

        Returns
        -------
        FloatOrArray
            Specific volume.
            Unit = m³/kg.
        """
        TK = convert_check_temperature(T, Tunit, self.Trange)
        Pa = convert_check_pressure(P, Punit, self.Prange)
        return self.eval(TK, Pa)

    @abstractmethod
    def eval(self, T: FloatOrArray, P: FloatOrArray) -> FloatOrArray:
        r"""Evaluate specific volume, $\hat{V}$, at given SI conditions without
        unit conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        P : FloatOrArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        FloatOrArray
            Specific volume.
            Unit = m³/kg
        """
        pass

    @abstractmethod
    def alpha(self, T: FloatOrArray, P: FloatOrArray) -> FloatOrArray:
        r"""Calculate thermal expansion coefficient, $\alpha$.

        $$\alpha=\frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_{P}$$

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        P : FloatOrArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        FloatOrArray
            Thermal expansion coefficient, $\alpha$.
        """
        pass

    @abstractmethod
    def beta(self, T: FloatOrArray, P: FloatOrArray) -> FloatOrArray:
        r"""Calculate isothermal compressibility coefficient, $\beta$.

        $$\beta=-\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_{T}$$

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        P : FloatOrArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        FloatOrArray
            Isothermal compressibility coefficient, $\beta$.
        """
        pass

# %% PolymerPVTEoS


class PolymerPVTEoS(PolymerPVTEquation):
    r"""_Abstract_ polymer equation of state in reduced form,
    $V(T, P, V0, T0, P0)$."""

    def __init__(self,
                 V0: float,
                 T0: float,
                 P0: float,
                 Tmin: float = 0.0,
                 Tmax: float = np.inf,
                 Pmin: float = 0.0,
                 Pmax: float = np.inf,
                 name: str = ''
                 ) -> None:
        """Construct `PolymerEoSReduced` with the given parameters."""

        # Check bounds
        check_bounds(V0, 0, np.inf, 'V0')
        check_bounds(T0, 0, np.inf, 'T0')
        check_bounds(P0, 0, np.inf, 'P0')

        self.V0 = V0
        self.T0 = T0
        self.P0 = P0
        super().__init__(Tmin, Tmax, Pmin, Pmax, name)

    @vectorize
    def eval(self,
             T: FloatOrArray,
             P: FloatOrArray
             ) -> FloatOrArray:
        t = T/self.T0
        p = P/self.P0
        solution = root_scalar(f=self.equation,
                               args=(t, p),
                               # bracket=[1.1, 1.5],
                               x0=1.1,
                               method='halley',
                               fprime=True,
                               fprime2=True)

        if solution.converged:
            v = solution.root
            V = v*self.V0
        else:
            print(solution.flag)
            V = -1.
        return V

    def alpha(self,
              T: FloatOrArray,
              P: FloatOrArray
              ) -> FloatOrArray:
        r"""Calculate thermal expansion coefficient, $\alpha$.

        $$\alpha=\frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_{P}$$

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        P : FloatOrArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        FloatOrArray
            Thermal expansion coefficient, $\alpha$.
        """
        dT = 0.5
        V2 = self.eval(T + dT, P)
        V1 = self.eval(T - dT, P)
        return (V2 - V1)/dT/(V1 + V2)

    def beta(self,
             T: FloatOrArray,
             P: FloatOrArray
             ) -> FloatOrArray:
        r"""Calculate isothermal compressibility coefficient, $\beta$.

        $$\beta=-\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_{T}$$

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        P : FloatOrArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        FloatOrArray
            Isothermal compressibility coefficient, $\beta$.
        """
        dP = 1e5
        P2 = P + dP
        P1 = np.max(P - dP, 0)
        V2 = self.eval(T, P2)
        V1 = self.eval(T, P1)
        return -(V2 - V1)/(P2 - P1)/(V1 + V2)*2

    @staticmethod
    @abstractmethod
    def equation(v: float, t: float, p: float) -> tuple[float, ...]:
        """Equation of state and its volume derivatives.

        Parameters
        ----------
        v : float
            Reduced volume.
        t : float
            Reduced temperature.
        p : float
            Reduced pressure.

        Returns
        -------
        tuple[float,...]
            Equation of state, first derivative, second derivative.
        """
        pass
# %% Tait


class Tait(PolymerPVTEquation):
    r"""Tait equation of state for the specific volume of a liquid.

    This EoS implements the following temperature and pressure dependence:

    $$\hat{V}(T,P)=\hat{V}(T,0)\left[1-C\ln\left(\frac{P}{B(T)}\right)\right]$$

    with:

    $$ \hat{V}(T,0) = A_0 + A_1(T - 273.15) + A_2(T - 273.15)^2 $$

    $$ B(T) = B_0\exp\left [-B_1(T - 273.15)\right] $$

    where $A_i$ and $B_i$ are constant parameters, $T$ is the absolute
    temperature, and $P$ is the pressure.

    References:

    *   Danner, Ronald P., and Martin S. High. Handbook of polymer
        solution thermodynamics. John Wiley & Sons, 2010.

    Parameters
    ----------
    A0 : float
        Parameter of equation.
        Unit = m³/kg.
    A1 : float
        Parameter of equation.
        Unit = m³/(kg·K).
    A2 : float
        Parameter of equation.
        Unit = m³/(kg·K²).
    B0 : float
        Parameter of equation.
        Unit = Pa.
    B1 : float
        Parameter of equation.
        Unit = 1/K.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    Pmin : float
        Lower pressure bound.
        Unit = Pa.
    Pmax : float
        Upper pressure bound.
        Unit = Pa.
    name : str
        Name.
    """

    A0: float
    A1: float
    A2: float
    B0: float
    B1: float

    _C = 0.0894

    def __init__(self,
                 A0: float,
                 A1: float,
                 A2: float,
                 B0: float,
                 B1: float,
                 Tmin: float = 0.0,
                 Tmax: float = np.inf,
                 Pmin: float = 0.0,
                 Pmax: float = np.inf,
                 name: str = ''
                 ) -> None:
        """Construct `Tait` with the given parameters."""

        # Check bounds
        check_bounds(A0, 1e-4, 2e-3, 'A0')
        check_bounds(A1, 1e-7, 2e-6, 'A1')
        check_bounds(A2, -2e-9, 1e-8, 'A2')
        check_bounds(B0, 1e7, 1e9, 'B0')
        check_bounds(B1, 1e-3, 2e-2, 'B1')

        self.A0 = A0
        self.A1 = A1
        self.A2 = A2
        self.B0 = B0
        self.B1 = B1
        super().__init__(Tmin, Tmax, Pmin, Pmax, name)

    def __repr__(self) -> str:
        return (
            f"name:          {self.name}\n"
            f"symbol:        {self.symbol}\n"
            f"unit:          {self.unit}\n"
            f"Trange [K]:    {self.Trange}\n"
            f"Prange [Pa]:   {self.Prange}\n"
            f"A0 [m³/kg]:    {self.A0}\n"
            f"A1 [m³/kg.K]:  {self.A1}\n"
            f"A2 [m³/kg.K²]: {self.A2}\n"
            f"B0 [Pa]:       {self.B0}\n"
            f"B1 [1/K]:      {self.B1}"
        )

    def eval(self,
             T: FloatOrArray,
             P: FloatOrArray
             ) -> FloatOrArray:
        r"""Evaluate specific volume, $\hat{V}$, at given SI conditions without
        unit conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        P : FloatOrArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        FloatOrArray
            Specific volume.
            Unit = m³/kg
        """
        TC = T - 273.15
        V0 = self.A0 + self.A1*TC + self.A2*TC**2
        B = self._B(T)
        V = V0*(1 - self._C*np.log(1 + P/B))
        return V

    def _B(self,
           T: FloatOrArray
           ) -> FloatOrArray:
        r"""Parameter B(T).

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            B(T).
            Unit = Pa
        """
        return self.B0*np.exp(-self.B1*(T - 273.15))

    def alpha(self,
              T: FloatOrArray,
              P: FloatOrArray
              ) -> FloatOrArray:
        r"""Calculate thermal expansion coefficient, $\alpha$.

        $$\alpha=\frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_{P}$$

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        P : FloatOrArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        FloatOrArray
            alpha.
        """
        A0 = self.A0
        A1 = self.A1
        A2 = self.A2
        TC = T - 273.15
        alpha0 = (A1 + 2*A2*TC)/(A0 + A1*TC + A2*TC**2)
        return alpha0 - P*self.B1*self.beta(T, P)

    def beta(self,
             T: FloatOrArray,
             P: FloatOrArray
             ) -> FloatOrArray:
        r"""Calculate isothermal compressibility coefficient, $\beta$.

        $$\beta=-\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_{T}$$

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        P : FloatOrArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        FloatOrArray
            beta.
        """
        B = self._B(T)
        return (self._C/(P + B))/(1 - self._C*np.log(1 + P/B))

    @classmethod
    def from_database(cls, name: str) -> Optional[Tait]:
        """Construct `Tait` with parameters from the database.

        The parameters are those reported in Table 3B-1 (p. 41) of the Handbook
        of Polymer Solution Thermodynamics.

        Parameters
        ----------
        name : str
            Polymer code name.
        """
        table = load_Tait_parameters()
        try:
            mask = table.index.to_series().str.contains(name, case=False)
            parameters = table[mask].iloc[0, :].to_dict()
        except IndexError:
            parameters = None
            print(
                f"Error: '{name}' does not exist in polymer database.\n"
                f"Valid names are: {table.index.to_list()}")

        if parameters:
            parameters['Pmin'] *= 1e6
            parameters['Pmax'] *= 1e6
            return cls(**parameters, name=name)

# %% Flory


class Flory(PolymerPVTEoS):
    r"""Flory equation of state for the specific volume of a polymer.

    This EoS implements the following implicit P-V-T dependence:

    $$ \frac{\tilde{P}\tilde{V}}{\tilde{T}} = \\
      \frac{\tilde{V}^{1/3}}{\tilde{V}^{1/3}-1}-\frac{1}{\tilde{V}\tilde{T}}$$

    where $\tilde{V}=V/V^*$, $\tilde{P}=P/P^*$ and $\tilde{T}=T/T^*$ are,
    respectively, the reduced volume, reduced pressure and reduced temperature.
    $V^*$, $P^*$ and $T^*$ are reference quantities that are polymer dependent.

    References:

    *   Caruthers et al. Handbook of Diffusion and Thermal Properties of
        Polymers and Polymer Solutions. AIChE, 1998.

    Parameters
    ----------
    V0 : float
        Reference volume, $V^*$.
    T0 : float
        Reference temperature, $T^*$.
    P0 : float
        Reference pressure, $P^*$.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    Pmin : float
        Lower pressure bound.
        Unit = Pa.
    Pmax : float
        Upper pressure bound.
        Unit = Pa.
    name : str
        Name.
    """

    @staticmethod
    def equation(v: float, t: float, p: float) -> tuple[float, float, float]:
        r"""Flory equation of state and its volume derivatives.

        Parameters
        ----------
        v : float
            Reduced volume, $\tilde{V}$.
        t : float
            Reduced temperature, $\tilde{T}$.
        p : float
            Reduced pressure, $\tilde{P}$.

        Returns
        -------
        tuple[float, float, float]
            Equation of state, first derivative, second derivative.
        """
        f = p*v/t - (v**(1/3)/(v**(1/3) - 1) - 1/(v*t))  # =0
        d1f = p/t - 1/(t*v**2) - 1/(3*(v**(1/3) - 1)*v**(2/3)) + \
            1/(3*(v**(1/3) - 1)**2*v**(1/3))
        d2f = (2*(9/t + (v**(4/3) - 2*v**(5/3))/(-1 + v**(1/3))**3))/(9*v**3)
        return (f, d1f, d2f)


class HartmannHaque(PolymerPVTEoS):
    r"""Hartmann-Haque equation of state for the specific volume of a polymer.

    This EoS implements the following implicit P-V-T dependence:

    $$ \tilde{P}\tilde{V}^5=\tilde{T}^{3/2}-\ln{\tilde{V}} $$

    where $\tilde{V}=V/V^*$, $\tilde{P}=P/P^*$ and $\tilde{T}=T/T^*$ are,
    respectively, the reduced volume, reduced pressure and reduced temperature.
    $V^*$, $P^*$ and $T^*$ are reference quantities that are polymer dependent.

    References:

    *   Caruthers et al. Handbook of Diffusion and Thermal Properties of
        Polymers and Polymer Solutions. AIChE, 1998.

    Parameters
    ----------
    V0 : float
        Reference volume, $V^*$.
    T0 : float
        Reference temperature, $T^*$.
    P0 : float
        Reference pressure, $P^*$.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    Pmin : float
        Lower pressure bound.
        Unit = Pa.
    Pmax : float
        Upper pressure bound.
        Unit = Pa.
    name : str
        Name.
    """

    @staticmethod
    def equation(v: float, t: float, p: float) -> tuple[float, float, float]:
        """Hartmann-Haque equation of state and its volume derivatives.

        Parameters
        ----------
        v : float
            Reduced volume.
        t : float
            Reduced temperature.
        p : float
            Reduced pressure.

        Returns
        -------
        tuple[float, float, float]
            Equation of state, first derivative, second derivative.
        """
        f = p*v**5 - t**(3/2) + np.log(v)  # =0
        d1f = 5*p*v**4 + 1/v
        d2f = 20*p*v**3 - 1/v**2
        return (f, d1f, d2f)
