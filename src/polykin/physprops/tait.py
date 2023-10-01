# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.utils import check_bounds, FloatOrArray, eps
from polykin.physprops.propertyequation import PropertyEquationTP

import numpy as np

__all__ = ['Tait']


class Tait(PropertyEquationTP):
    r"""Tait equation of state for the specific volume of a liquid.

    This equation implements the following temperature and pressure dependence:

    $$ \hat{V}(T,P)=\hat{V}(T,0)\left[1-C\ln\left(\frac{P}{B(T)}\right)\right]$$

    with:

    $$ \hat{V}(T,0) = A_0 + A_1(T - 273.15) + A_2(T - 273.15)^2 $$

    $$ B(T) = B_0\exp\left [-B_1(T - 273.15)\right] $$

    where $A_i$ and $B_i$ are constant parameters, $T$ is the absolute
    temperature, and $P$ is the pressure.

    Reference: Handbook of Polymer Solution Thermodynamics.

    Parameters
    ----------
    A0 : float
        Parameter of equation.
        Unit = m³/kg
    A1 : float
        Parameter of equation.
        Unit = m³/(kg.K)
    A2 : float
        Parameter of equation.
        Unit = m³/(kg.K²)
    B0 : float
        Parameter of equation.
        Unit = Pa
    B1 : float
        Parameter of equation.
        Unit = 1/K
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    Pmin : float
        Lower pressure bound.
        Unit = Pa
    Pmax : float
        Upper pressure bound.
        Unit = Pa
    name : str
        Name.
    """

    A0: float
    A1: float
    A2: float
    B0: float
    B1: float
    symbol = r"$\hat{V}$"
    unit = "kg/m³"
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

        # Check bounds
        check_bounds(A0, 1e-4, 2e-3, 'A0')
        check_bounds(A1, 1e-7, 2e-6, 'A1')
        check_bounds(A2, -2e-9, 1e-8, 'A2')
        check_bounds(B0, 1e7, 1e9, 'B0')
        check_bounds(B1, 1e-3, 2e-2, 'B1')
        check_bounds(Tmin, 0, np.inf, 'Tmin')
        check_bounds(Tmax, 0, np.inf, 'Tmax')
        check_bounds(Tmax-Tmin, eps, np.inf, 'Tmax-Tmin')
        check_bounds(Pmin, 0, np.inf, 'Pmin')
        check_bounds(Pmax, 0, np.inf, 'Pmax')
        check_bounds(Pmax-Pmin, eps, np.inf, 'Pmax-Pmin')

        self.A0 = A0
        self.A1 = A1
        self.A2 = A2
        self.B0 = B0
        self.B1 = B1
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.Pmin = Pmin
        self.Pmax = Pmax
        self.name = name

    def __repr__(self) -> str:
        return (
            f"name:          {self.name}\n"
            f"symbol:        {self.symbol}\n"
            f"unit:          {self.unit}\n"
            f"A0 [m³/kg]:    {self.A0}\n"
            f"A1 [m³/kg.K]:  {self.A1}\n"
            f"A2 [m³/kg.K²]: {self.A2}\n"
            f"B0 [Pa]:       {self.B0}\n"
            f"B1 [1/K]:      {self.B1}\n"
            f"Tmin [K]:      {self.Tmin}\n"
            f"Tmax [K]:      {self.Tmax}"
            f"Pmin [Pa]:     {self.Pmin}\n"
            f"Pmax [Pa]:     {self.Pmax}\n"
        )

    def eval(self,
             T: FloatOrArray,
             P: FloatOrArray
             ) -> FloatOrArray:
        r"""Specific volume, $\hat{V}$.

        Direct evaluation at given SI conditions, without unit conversions or
        checks.

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
        r"""Thermal expansion coefficient, $\alpha$.

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
        r"""Isothermal compressibility coefficient, $\beta$.

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
