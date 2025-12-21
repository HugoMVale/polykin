# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal

# import numpy as np
from numpy import sqrt
from scipy.constants import R

from polykin.math.derivatives import derivative_complex
from polykin.utils.math import eps
from polykin.utils.types import FloatVector, Number

__all__ = ["EoS", "GasEoS"]


class EoS(ABC):
    """Base class for equation of state."""

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
        pass

    @abstractmethod
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
        pass

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
    """Base class for gas equations of state."""

    @abstractmethod
    def P(self, T: float, v: float, y: FloatVector) -> float:
        """Calculate the pressure of the fluid."""
        pass

    @abstractmethod
    def Z(self, T: Number, P: Number, y: FloatVector) -> float:
        """Calculate the compressibility factor of the fluid."""
        pass

    @abstractmethod
    def phi(self, T: float, P: float, y: FloatVector) -> FloatVector:
        """Calculate the fugacity coefficients of all components."""
        pass

    def v(
        self,
        T: float,
        P: float,
        y: FloatVector,
    ) -> float:
        r"""Calculate the molar volume the fluid.

        $$ v = \frac{Z R T}{P} $$

        where $v$ is the molar volume, $Z(T, P, y)$ is the compressibility
        factor, $T$ is the temperature, $P$ is the pressure, and $y$ is the
        mole fraction vector.

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

    def beta(
        self,
        T: float,
        P: float,
        y: FloatVector,
    ) -> float:
        r"""Calculate the thermal expansion coefficient.

        $$ \beta
           = \frac{1}{v} \left( \frac{\partial v}{\partial T} \right)_P
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
        y : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Thermal expansion coefficient, $\beta$ [K⁻¹].
        """
        dZdT, Z = derivative_complex(
            lambda x: self.Z(x, P, y),
            T,
        )
        return 1 / T + dZdT / Z

    def kappa(
        self,
        T: float,
        P: float,
        y: FloatVector,
    ) -> float:
        r"""Calculate the isothermal compressibility coefficient.

        $$ \kappa
           = - \frac{1}{v} \left( \frac{\partial v}{\partial P} \right)_T
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
        y : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Isothermal compressibility coefficient, $\kappa$ [Pa⁻¹].
        """
        dZdP, Z = derivative_complex(
            lambda x: self.Z(T, x, y),
            P,
        )
        return 1 / P - dZdP / Z

    def f(
        self,
        T: float,
        P: float,
        y: FloatVector,
    ) -> FloatVector:
        r"""Calculate the fugacity of all components.

        For each component, the fugacity is given by:

        $$ \hat{f}_i = \hat{\phi}_i y_i P $$

        where $\hat{\phi}_i(T,P,y)$ is the fugacity coefficient, $P$ is the
        pressure, and $y_i$ is the mole fraction.

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
    """Base class for gas-liquid equations of state."""

    @abstractmethod
    def P(self, T: float, v: float, z: FloatVector) -> float:
        """Calculate the pressure of the fluid."""
        pass

    @abstractmethod
    def Z(self, T: float, P: float, z: FloatVector) -> FloatVector:
        """Calculate the compressibility factors for the possible phases of a
        fluid.
        """
        pass

    def beta(
        self,
        T: float,
        P: float,
        z: FloatVector,
    ) -> FloatVector:
        r"""Calculate the thermal expansion coefficients of the possible phases
        of a fluid.

        $$ \beta
           = \frac{1}{v} \left( \frac{\partial v}{\partial T} \right)_P
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

    def kappa(
        self,
        T: float,
        P: float,
        z: FloatVector,
    ) -> FloatVector:
        r"""Calculate the isothermal compressibility coefficients of the
        possible phases of a fluid.

        $$ \kappa
           = - \frac{1}{v} \left( \frac{\partial v}{\partial P} \right)_T
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
        pass

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
