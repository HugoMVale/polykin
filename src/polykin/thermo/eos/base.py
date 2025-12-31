# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal

import numpy as np
import scipy.integrate as integrate
from numpy import exp, sqrt
from scipy.constants import R

from polykin.math.derivatives import (
    derivative_centered,
    derivative_complex,
    jacobian_forward,
)
from polykin.math.roots import root_newton
from polykin.utils.exceptions import RootSolverError
from polykin.utils.math import eps
from polykin.utils.types import FloatVector, Number

__all__ = ["EoS", "GasEoS"]


class EoS(ABC):
    """Abstract base class for equations of state."""

    _N: int
    name: str

    def __init__(self, N: int, name: str) -> None:
        self._N = N
        self.name = name

    @property
    def N(self) -> int:
        """Number of components."""
        return self._N

    @abstractmethod
    def Z(self, T, P, z) -> float | FloatVector:
        """Calculate the compressibility factor of the fluid."""

    def DA(
        self,
        T: float,
        V: float,
        n: FloatVector,
        v0: float,
    ) -> float:
        r"""Calculate the departure of Helmholtz energy.

        $$ A(T,V,n) - A^{\circ}(T,V,n)$$

        Parameters
        ----------
        T : float
            Temperature [K].
        V : float
            Volume [m³].
        n : FloatVector (N)
            Mole amounts of all components [mol].
        v0 : float
            Molar volume in reference state [m³/mol].

        Returns
        -------
        float
            Helmholtz energy departure, $A - A^{\circ}$ [J].
        """

    def DX(
        self,
        T: float,
        P: float,
        y: FloatVector,
        P0: float = 1e5,
    ) -> dict[str, float]:

        v0 = R * T / P0
        nt = 1.0
        n = nt * y

        Z = self.Z(T, P, y)
        if isinstance(Z, Iterable):
            Z = max(Z)  # temporary fix, get only vapor solution !!!
        V = nt * Z * R * T / P

        dT = 2 * sqrt(eps) * T
        DA_minus = self.DA(T - dT, V, n, v0)
        DA_plus = self.DA(T + dT, V, n, v0)
        DA = (DA_minus + DA_plus) / 2
        DS = -(DA_plus - DA_minus) / (2 * dT)
        DU = DA + T * DS
        DH = DU + R * T * (Z - 1)
        DG = DA + R * T * (Z - 1)
        result = {"A": DA, "G": DG, "H": DH, "S": DS, "U": DU}
        return result


class GasEoS(EoS):
    r"""Abstract base class for gas equations of state.

    A gas EoS is defined in terms of a compressibility factor function having
    pressure and temperature as independent variables:

    $$ Z = Z(T,P) $$

    All other thermodynamic properties are derived from $Z$.

    To implement a specific gas EoS, subclasses must:

    * Implement the `Z` method.
    * Preferably override the `gR`, `P`, and `phi` methods for efficiency.
    """

    @abstractmethod
    def Z(self, T: Number, P: Number, y: FloatVector) -> float:
        r"""Calculate the compressibility factor of the fluid.

        $$ Z = \frac{P v}{R T} $$

        Parameters
        ----------
        T : Number
            Temperature [K].
        P : Number
            Pressure [Pa].
        y : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Compressibility factor.
        """

    def gR(self, T: Number, P: float, y: FloatVector) -> float:
        r"""Calculate the molar residual Gibbs energy of the fluid.

        $$ g^R = R T \int_0^P (Z-1)\frac{d P}{P} $$

        Parameters
        ----------
        T : Number
            Temperature [K].
        P : float
            Pressure [Pa].
        y : FloatVector
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Molar residual Gibbs energy [J/mol].
        """
        res = integrate.quad(lambda P_: (self.Z(T, P_, y) - 1) / P_, eps, P)

        return R * T * res[0]

    def P(self, T: float, v: float, y: FloatVector) -> float:
        r"""Calculate the pressure of the fluid.

        $$ P = \frac{Z(T,P,y) R T}{v} $$

        Parameters
        ----------
        T : float
            Temperature [K].
        v : float
            Molar volume [m³/mol].
        y : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Pressure [Pa].
        """
        sol = root_newton(
            lambda P_: P_ * v / (R * T) - self.Z(T, P_, y),
            x0=R * T / v,
        )

        if not sol.success:
            raise RootSolverError(
                f"Could not solve for pressure in `GasEoS.P` method.\n{sol.message}"
            )

        return sol.x

    def phi(self, T: float, P: float, y: FloatVector) -> FloatVector:
        r"""Calculate the fugacity coefficients of all components.

        $$ \ln \hat{\phi}_i = \frac{1}{RT}
           \left( \frac{\partial (n g^R)}{\partial n_i} \right)_{T,P,n_j} $$

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        y : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        FloatVector (N)
            Fugacity coefficients of all components.
        """

        def GR(n: np.ndarray):
            """Total residual Gibbs energy."""
            nT = n.sum()
            return nT * self.gR(T, P, n / nT)

        dGRdn = jacobian_forward(GR, y)

        return exp(dGRdn / (R * T))

    def sR(self, T: float, P: float, y: FloatVector) -> float:
        r"""Calculate the molar residual entropy of the fluid.

        $$ s^R = -\left( \frac{\partial g^R}{\partial T} \right)_{P,y_i} $$

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        y : FloatVector
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Molar residual entropy [J/(mol·K)].
        """
        return -1 * derivative_centered(lambda T_: self.gR(T_, P, y), T)[0]

    def hR(self, T: float, P: float, y: FloatVector) -> float:
        r"""Calculate the molar residual enthalpy of the fluid.

        $$ h^R = g^R + T s^R $$

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        y : FloatVector
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Molar residual enthalpy [J/mol].
        """
        return self.gR(T, P, y) + T * self.sR(T, P, y)

    def vR(self, T: float, P: float, y: FloatVector) -> float:
        r"""Calculate the molar residual volume of the fluid.

        $$ v^R = \left(Z - 1 \right) \frac{R T}{P} $$

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        y : FloatVector
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Molar residual volume [m³/mol].
        """
        return R * T / P * (self.Z(T, P, y) - 1.0)

    def v(self, T: float, P: float, y: FloatVector) -> float:
        r"""Calculate the molar volume the fluid.

        $$ v = Z \frac{R T}{P} $$

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        y : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Molar volume of the fluid [m³/mol].
        """
        return self.Z(T, P, y) * R * T / P

    def beta(self, T: float, P: float, y: FloatVector) -> float:
        r"""Calculate the thermal expansion coefficient.

        $$ \beta \equiv
         \frac{1}{v} \left( \frac{\partial v}{\partial T} \right)_{P,y_i}
         = \frac{1}{T}
         + \frac{1}{Z} \left( \frac{\partial Z}{\partial T} \right)_{P,y_i} $$

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        y : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Thermal expansion coefficient, $\beta$ [K⁻¹].
        """
        dZdT, Z = derivative_centered(lambda T_: self.Z(T_, P, y), T)

        return 1 / T + dZdT / Z

    def kappa(self, T: float, P: float, y: FloatVector) -> float:
        r"""Calculate the isothermal compressibility coefficient.

        $$ \kappa \equiv
        - \frac{1}{v} \left( \frac{\partial v}{\partial P} \right)_{T,y_i}
        = \frac{1}{P}
        - \frac{1}{Z} \left( \frac{\partial Z}{\partial P} \right)_{T,y_i} $$

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        y : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Isothermal compressibility coefficient, $\kappa$ [Pa⁻¹].
        """
        dZdP, Z = derivative_complex(lambda P_: self.Z(T, P_, y), P)

        return 1 / P - dZdP / Z

    def f(self, T: float, P: float, y: FloatVector) -> FloatVector:
        r"""Calculate the fugacity of all components.

        $$ \hat{f}_i = \hat{\phi}_i y_i P $$

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        y : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        FloatVector (N)
            Fugacities of all components [Pa].
        """
        return self.phi(T, P, y) * y * P


class GasLiquidEoS(EoS):
    """Abstract base class for gas-liquid equations of state."""

    @abstractmethod
    def P(self, T: float, v: float, z: FloatVector) -> float:
        """Calculate the pressure of the fluid."""

    @abstractmethod
    def Z(self, T: float, P: float, z: FloatVector) -> FloatVector:
        """Calculate the compressibility factors for the possible phases of a
        fluid.
        """

    @abstractmethod
    def phi(
        self,
        T: float,
        P: float,
        z: FloatVector,
        phase: Literal["L", "V"],
    ) -> FloatVector:
        """Calculate the fugacity coefficients of all components in a given
        phase.
        """

    def beta(self, T: float, P: float, z: FloatVector) -> FloatVector:
        r"""Calculate the thermal expansion coefficients of the possible phases
        of a fluid.

        $$ \beta \equiv
             \frac{1}{v} \left( \frac{\partial v}{\partial T} \right)_P
           = \frac{1}{T}
             + \frac{1}{Z} \left( \frac{\partial Z}{\partial T} \right)_P $$

        where $P$ is the pressure, $T$ is the temperature, and $Z$ is the
        compressibility factor.

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        z : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        FloatVector
            Thermal expansion coefficients of the possible phases [K⁻¹].
            If two phases are possible, the first result corresponds to the
            liquid.
        """
        dT = 1.0
        Zp = self.Z(T + dT, P, z)
        Zm = self.Z(T - dT, P, z)
        dZdT = (Zp - Zm) / (2 * dT)
        Z = (Zp + Zm) / 2

        return 1 / T + dZdT / Z

    def kappa(self, T: float, P: float, z: FloatVector) -> FloatVector:
        r"""Calculate the isothermal compressibility coefficients of the
        possible phases of a fluid.

        $$ \kappa \equiv
             - \frac{1}{v} \left( \frac{\partial v}{\partial P} \right)_T
           = \frac{1}{P}
             - \frac{1}{Z} \left( \frac{\partial Z}{\partial P} \right)_T $$

        where $P$ is the pressure, $T$ is the temperature, and $Z$ is the
        compressibility factor.

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        z : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Isothermal compressibility coefficients of the possible phases [Pa⁻¹].
            If two phases are possible, the first result corresponds to the
            liquid.
        """
        dP = max(P * 1e-2, 1e1)
        Zp = self.Z(T, P + dP, z)
        Zm = self.Z(T, P - dP, z)
        dZdP = (Zp - Zm) / (2 * dP)
        Z = (Zp + Zm) / 2

        return 1 / P - dZdP / Z

    def v(
        self,
        T: float,
        P: float,
        z: FloatVector,
    ) -> FloatVector:
        r"""Calculate the molar volumes of the possible phases a fluid.

        $$ v = \frac{Z R T}{P} $$

        where $v$ is the molar volume, $Z(T, P, z)$ is the compressibility
        factor, $T$ is the temperature, and $P$ is the pressure, and $z$ is the
        mole fraction vector.

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        z : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        FloatVector
            Molar volumes of the possible phases [m³/mol]. If two phases are
            possible, the first result is the lowest value (liquid).
        """
        return self.Z(T, P, z) * R * T / P

    def f(
        self,
        T: float,
        P: float,
        z: FloatVector,
        phase: Literal["L", "V"],
    ) -> FloatVector:
        r"""Calculate the fugacity of all components in a given phase.

        For each component, the fugacity is given by:

        $$ \hat{f}_i = \hat{\phi}_i z_i P $$

        where $\hat{\phi}_i(T,P,y)$ is the fugacity coefficient, $P$ is the
        pressure, and $z_i$ is the mole fraction.

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        z : FloatVector (N)
            Mole fractions of all components [mol/mol].
        phase : Literal['L', 'V']
            Phase of the fluid. Only relevant for systems where both liquid
            and vapor phases may exist.

        Returns
        -------
        FloatVector (N)
            Fugacities of all components [Pa].
        """
        return self.phi(T, P, z, phase) * z * P

    def K(
        self,
        T: float,
        P: float,
        x: FloatVector,
        y: FloatVector,
    ) -> FloatVector:
        r"""Calculate the K-values of all components.

        $$ K_i = \frac{y_i}{x_i} = \frac{\hat{\phi}_i^L}{\hat{\phi}_i^V} $$

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        x : FloatVector (N)
            Liquid mole fractions of all components [mol/mol].
        y : FloatVector (N)
            Vapor mole fractions of all components [mol/mol].

        Returns
        -------
        FloatVector (N)
            K-values of all components.
        """
        phiV = self.phi(T, P, y, "V")
        phiL = self.phi(T, P, x, "L")

        return phiL / phiV
