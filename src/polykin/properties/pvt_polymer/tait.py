# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Union

import numpy as np
from numpy import exp, log

from polykin.utils.tools import check_bounds
from polykin.utils.types import FloatArray

from .base import PolymerPVTEquation

__all__ = ['Tait']


class Tait(PolymerPVTEquation):
    r"""Tait equation of state for the specific volume of a liquid.

    This EoS implements the following explicit PVT dependence:

    $$\hat{V}(T,P)=\hat{V}(T,0)\left[1-C\ln\left(\frac{P}{B(T)}\right)\right]$$

    with:

    $$ \begin{gather*}
    \hat{V}(T,0) = A_0 + A_1(T - 273.15) + A_2(T - 273.15)^2 \\
    B(T) = B_0\exp\left [-B_1(T - 273.15)\right]
    \end{gather*} $$

    where $A_i$ and $B_i$ are constant parameters, $T$ is the absolute
    temperature, and $P$ is the pressure.

    **References**

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
             T: Union[float, FloatArray],
             P: Union[float, FloatArray]
             ) -> Union[float, FloatArray]:
        r"""Evaluate specific volume, $\hat{V}$, at given SI conditions without
        unit conversions or checks.

        Parameters
        ----------
        T : float | FloatArray
            Temperature.
            Unit = K.
        P : float | FloatArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        float | FloatArray
            Specific volume.
            Unit = m³/kg.
        """
        TC = T - 273.15
        V0 = self.A0 + self.A1*TC + self.A2*TC**2
        B = self._B(T)
        V = V0*(1 - self._C*log(1 + P/B))
        return V

    def _B(self,
           T: Union[float, FloatArray]
           ) -> Union[float, FloatArray]:
        r"""Parameter B(T).

        Parameters
        ----------
        T : float | FloatArray
            Temperature.
            Unit = K.

        Returns
        -------
        float | FloatArray
            B(T).
            Unit = Pa.
        """
        return self.B0*exp(-self.B1*(T - 273.15))

    def alpha(self,
              T: Union[float, FloatArray],
              P: Union[float, FloatArray]
              ) -> Union[float, FloatArray]:
        r"""Calculate thermal expansion coefficient, $\alpha$.

        $$\alpha=\frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_{P}$$

        Parameters
        ----------
        T : float | FloatArray
            Temperature.
            Unit = K.
        P : float | FloatArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        float | FloatArray
            Thermal expansion coefficient, $\alpha$.
            Unit = 1/K.
        """
        A0 = self.A0
        A1 = self.A1
        A2 = self.A2
        TC = T - 273.15
        alpha0 = (A1 + 2*A2*TC)/(A0 + A1*TC + A2*TC**2)
        return alpha0 - P*self.B1*self.beta(T, P)

    def beta(self,
             T: Union[float, FloatArray],
             P: Union[float, FloatArray]
             ) -> Union[float, FloatArray]:
        r"""Calculate isothermal compressibility coefficient, $\beta$.

        $$\beta=-\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_{T}$$

        Parameters
        ----------
        T : float | FloatArray
            Temperature.
            Unit = K.
        P : float | FloatArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        float | FloatArray
            Isothermal compressibility coefficient, $\beta$.
            Unit = 1/Pa.
        """
        B = self._B(T)
        return (self._C/(P + B))/(1 - self._C*log(1 + P/B))
