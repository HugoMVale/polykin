# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from numpy import log
from scipy.constants import R

from polykin.utils.types import FloatVector

from .base import GasEoS

__all__ = ["IdealGas"]


class IdealGas(GasEoS):
    r"""Ideal gas equation of state.

    This EoS is based on the following $P(v,T)$ relationship:

    $$ P = \frac{R T}{v} $$

    where $P$ is the pressure, $T$ is the temperature, and $v$ is the molar
    volume.

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

        Returns
        -------
        float
            Compressibility factor.
        """
        return 1.0

    def P(self, T: float, v: float, y=None) -> float:
        r"""Calculate the pressure of the fluid.

        Parameters
        ----------
        T : float
            Temperature [K].
        v : float
            Molar volume [m³/mol].

        Returns
        -------
        float
            Pressure [Pa].
        """
        return R * T / v

    def phi(self, T=None, P=None, y=None) -> FloatVector:
        r"""Calculate the fugacity coefficients of all components.

        $$ \hat{\phi}_i = 1 $$

        Returns
        -------
        FloatVector (N)
            Fugacity coefficients of all components.
        """
        return np.ones(self.N)

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
        nT = n.sum()
        return -nT * R * T * log(V / (nT * v0))
