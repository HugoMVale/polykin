# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from scipy.constants import R

from polykin.utils.types import FloatVector, override

from .base import GasEoS

__all__ = ["IdealGas"]


class IdealGas(GasEoS):
    r"""Ideal gas equation of state.

    This EoS is based on the following trivial $Z(T,P)$ relationship:

    $$ Z = 1 $$

    Parameters
    ----------
    N : int
        Number of components.
    name : str
        Name.
    """

    def __init__(self, N: int, name: str = "") -> None:

        super().__init__(N, name)

    def Z(self, T=None, P=None, y=None) -> float:
        r"""Calculate the compressibility factor of the fluid.

        $$ Z = 1 $$

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
            Compressibility factor.
        """
        return 1.0

    @override
    def gR(self, T=None, P=None, y=None) -> float:
        """Calculate the molar residual Gibbs energy of the fluid.

        $$ g^R = 0 $$

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
            Molar residual Gibbs energy [J/mol].
        """
        return 0.0

    @override
    def P(self, T: float, v: float, y=None) -> float:
        r"""Calculate the pressure of the fluid.

        $$ P = \frac{R T}{v} $$

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
        return R * T / v

    @override
    def phi(self, T=None, P=None, y=None) -> FloatVector:
        r"""Calculate the fugacity coefficients of all components.

        $$ \hat{\phi}_i = 1 $$

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
        return np.ones(self.N)

    @override
    def beta(self, T: float, P=None, y=None) -> float:
        r"""Calculate the thermal expansion coefficient.

        $$ \beta \equiv
           \frac{1}{v} \left( \frac{\partial v}{\partial T} \right)_P
           = \frac{1}{T} $$

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
        return 1 / T

    @override
    def kappa(self, T: float, P: float, y=None) -> float:
        r"""Calculate the isothermal compressibility coefficient.

        $$ \kappa \equiv
           - \frac{1}{v} \left( \frac{\partial v}{\partial P} \right)_T
           = \frac{1}{P} $$

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
        return 1 / P
