# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

import functools

from numpy import dot, exp, log
from scipy.constants import R

from polykin.properties.pvt.mixing_rules import quadratic_mixing
from polykin.properties.pvt.virial import B_mixture
from polykin.utils.math import convert_FloatOrVectorLike_to_FloatVector
from polykin.utils.types import (
    FloatSquareMatrix,
    FloatVector,
    FloatVectorLike,
)

from .base import GasEoS

__all__ = ["Virial"]


class Virial(GasEoS):
    r"""Virial equation of state truncated to the second coefficient.

    This EoS is based on the following $P(v,T)$ relationship:

    $$ P = \frac{R T}{v - B_m} $$

    where $P$ is the pressure, $T$ is the temperature, $v$ is the molar
    volume, $B_m$ is the mixture second virial coefficient. The parameter
    $B_m=B_m(T,y)$ is estimated based on the critical properties of the pure
    components and the mixture composition $y$.

    !!! important

        This equation is an improvement over the ideal gas model, but it
        should only be used up to moderate pressures such that $v/v_c > 2$.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
        liquids 4th edition, 1986, p. 37, 40, 80, 82.

    Parameters
    ----------
    Tc : float | FloatVectorLike (N)
        Critical temperatures of all components [K].
    Pc : float | FloatVectorLike (N)
        Critical pressures of all components [Pa].
    Zc : float | FloatVectorLike (N)
        Critical compressibility factors of all components.
    w : float | FloatVectorLike (N)
        Acentric factors of all components.
    name : str
        Name.
    """

    Tc: FloatVector
    Pc: FloatVector
    Zc: FloatVector
    w: FloatVector

    def __init__(
        self,
        Tc: float | FloatVectorLike,
        Pc: float | FloatVectorLike,
        Zc: float | FloatVectorLike,
        w: float | FloatVectorLike,
        name: str = "",
    ) -> None:

        Tc, Pc, Zc, w = convert_FloatOrVectorLike_to_FloatVector([Tc, Pc, Zc, w])

        N = len(Tc)
        super().__init__(N, name)
        self.Tc = Tc
        self.Pc = Pc
        self.Zc = Zc
        self.w = w

    def Bm(
        self,
        T: float,
        y: FloatVector,
    ) -> float:
        r"""Calculate the second virial coefficient of the mixture.

        $$ B_m = \sum_i \sum_j y_i y_j B_{ij} $$

        **References**

        *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
            liquids 4th edition, 1986, p. 79.

        Parameters
        ----------
        T : float
            Temperature [K].
        y : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Mixture second virial coefficient, $B_m$ [m³/mol].
        """
        return quadratic_mixing(y, self.Bij(T))

    @functools.cache
    def Bij(
        self,
        T: float,
    ) -> FloatSquareMatrix:
        r"""Calculate the matrix of interaction virial coefficients.

        The calculation is handled by [`B_mixture`](../../properties/pvt/B_mixture.md).

        Parameters
        ----------
        T : float
            Temperature [K].

        Returns
        -------
        FloatSquareMatrix (N,N)
            Matrix of interaction virial coefficients, $B_{ij}$ [m³/mol].
        """
        return B_mixture(T, self.Tc, self.Pc, self.Zc, self.w)

    def Z(
        self,
        T: float,
        P: float,
        y: FloatVector,
    ) -> float:
        r"""Calculate the compressibility factor of the fluid.

        $$ Z = 1 + \frac{B_m P}{R T} $$

        where $P$ is the pressure, $T$ is the temperature, and $B_m=B_m(T,y)$
        is the mixture second virial coefficient.

        **References**

        *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
            liquids 4th edition, 1986, p. 37.

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
        Bm = self.Bm(T, y)
        return 1.0 + Bm * P / (R * T)

    def P(
        self,
        T: float,
        v: float,
        y: FloatVector,
    ) -> float:
        r"""Calculate the pressure of the fluid.

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
        Bm = self.Bm(T, y)
        return R * T / (v - Bm)

    def phi(
        self,
        T: float,
        P: float,
        y: FloatVector,
    ) -> FloatVector:
        r"""Calculate the fugacity coefficients of all components.

        For each component, the fugacity coefficient is given by:

        $$
        \ln \hat{\phi}_i = \left(2\sum_j {y_jB_{ij}} -B_m \right)\frac{P}{RT}
        $$

        where $P$ is the pressure, $T$ is the temperature, $B_{ij}$ is the
        matrix of interaction virial coefficients, $B_m$ is the second virial
        coefficient of the mixture, and $y_i$ is the mole fraction.

        **References**

        *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
            liquids 4th edition, 1986, p. 145.

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
        B = self.Bij(T)
        Bm = self.Bm(T, y)
        return exp((2 * dot(B, y) - Bm) * P / (R * T))

    def DA(
        self,
        T: float,
        V: float,
        n: FloatVector,
        v0: float,
    ) -> float:
        nT = n.sum()
        y = n / nT
        Bm = self.Bm(T, y)
        return -nT * R * T * log((V - nT * Bm) / (nT * v0))
