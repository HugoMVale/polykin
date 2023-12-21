# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector, FloatVectorLike, FloatOrArray, \
    FloatSquareMatrix
from .base import GasEoS
from ..mixing_rules import quadratic_mixing

import numpy as np
from numpy import sqrt, exp, log, dot
from scipy.constants import R
import functools

__all__ = ['Virial',
           'B_pure',
           'B_mixture']

# %% Virial equation


class Virial(GasEoS):
    r"""Virial equation of state truncated to the second coefficient.

    This EOS is based on the following $P(v,T)$ relationship:

    $$ P = \frac{R T}{v - B_m} $$

    where $P$ is the pressure, $T$ is the temperature, $v$ is the molar
    volume, $B_m$ is the mixture second virial coefficient. The parameter
    $B_m=B_m(T,y)$ is estimated based on the critical properties of the pure
    components and the mixture composition $y$.

    !!! important

        This equation is an improvement over the ideal gas model, but it
        should only be used up to moderate pressures such that $v/v_c > 2$.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases &
    liquids 4th edition, 1986, p. 37, 40, 80, 82.

    Parameters
    ----------
    Tc : FloatVector
        Critical temperatures of all components. Unit = K.
    Pc : FloatVector
        Critical pressures of all components. Unit = Pa.
    Zc : FloatVector
        Critical compressibility factors of all components.
    w : FloatVector
        Acentric factors of all components.
    """

    Tc: FloatVector
    Pc: FloatVector
    Zc: FloatVector
    w: FloatVector

    def __init__(self,
                 Tc: FloatVectorLike,
                 Pc: FloatVectorLike,
                 Zc: FloatVectorLike,
                 w: FloatVectorLike
                 ) -> None:
        """Construct `Virial` with the given parameters."""

        if isinstance(Tc, (list, tuple)):
            Tc = np.array(Tc, dtype=np.float64)
        if isinstance(Pc, (list, tuple)):
            Pc = np.array(Pc, dtype=np.float64)
        if isinstance(Zc, (list, tuple)):
            Zc = np.array(Zc, dtype=np.float64)
        if isinstance(w, (list, tuple)):
            w = np.array(w, dtype=np.float64)

        self.Tc = Tc
        self.Pc = Pc
        self.Zc = Zc
        self.w = w

    def Z(self, T, P, y):
        r"""Calculate the compressibility factor of the fluid.

        $$ Z = 1 + \frac{B_m P}{R T} $$

        where $Z$ is the compressibility factor, $P$ is the pressure, $T$ is
        the temperature, and $B_m=B_m(T,y)$ is the mixture second virial
        coefficient.

        Reference:

        * RC Reid, JM Prausniz, and BE Poling. The properties of gases &
        liquids 4th edition, 1986, p. 37.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        P : float
            Pressure. Unit = Pa.
        y : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector
            Compressibility factor of the fluid.
        """
        Bm = self.Bm(T, y)
        return 1. + Bm*P/(R*T)

    def P(self, T, v, y):
        r"""Calculate the pressure of the fluid.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        v : float
            Molar volume. Unit = m³/mol.
        y : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector
             Pressure. Unit = Pa.
        """
        Bm = self.Bm(T, y)
        return R*T/(v - Bm)

    def Bm(self,
           T: float,
           y: FloatVector
           ) -> float:
        r"""Calculate the second virial coefficient of the mixture.

        $$ B_m = \sum_i \sum_j y_i y_j B_{ij} $$

        Reference:

        * RC Reid, JM Prausniz, and BE Poling. The properties of gases &
        liquids 4th edition, 1986, p. 82.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        y : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Mixture second virial coefficient, $B_m$. Unit = m³/mol.
        """
        return quadratic_mixing(y, self.Bij(T))

    @functools.cache
    def Bij(self,
            T: float,
            ) -> FloatSquareMatrix:
        r"""Calculate the matrix of interaction virial coefficients.

        The calculation is handled by
        [`B_mixture`](.#polykin.properties.eos.B_mixture).

        Parameters
        ----------
        T : float
            Temperature. Unit = K.

        Returns
        -------
        FloatSquareMatrix
            Matrix of interaction virial coefficients, $B_{ij}$.
            Unit = m³/mol.
        """
        return B_mixture(T, self.Tc, self.Pc, self.Zc, self.w)

    def phi(self,
            T: float,
            P: float,
            y: FloatVector
            ) -> FloatVector:
        r"""Calculate the fugacity coefficients of all components in the gas
        phase.

        $$
        \ln \hat{\phi}_i = \left(2\sum_j {y_jB_{ij}} -B_m \right)\frac{P}{RT}
        $$

        where $\phi_i$ is the fugacity coefficient, $P$ is the pressure, $T$
        is the temperature, $B_{ij}$ is the matrix of interaction virial
        coefficients, $B_m$ is the second virial coefficient of the mixture,
        and $y_i$ is the mole fraction in the gas phase.

        Reference:

        * RC Reid, JM Prausniz, and BE Poling. The properties of gases &
        liquids 4th edition, 1986, p. 145.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        P : float
            Pressure. Unit = Pa.
        y : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector
            Fugacity coefficients of all components.
        """
        Bm = self.Bm(T, y)
        B = self.Bij(T)
        return exp((2*dot(B, y) - Bm)*P/(R*T))

    def DA(self, n, T, V, V0):
        nt = n.sum()
        y = n/nt
        Bm = self.Bm(T, y)
        return nt*R*T*log(V/(V - nt*Bm)) - R*T*log(V/V0)

# %% Second virial coefficient


def B_pure(T: FloatOrArray,
           Tc: float,
           Pc: float,
           w: float) -> FloatOrArray:
    r"""Estimate the second virial coefficient of a nonpolar or slightly polar
    gas.

    $$ \frac{B P_c}{R T_c} = B^{(0)}(T_r) + \omega B^{(1)}(T_r) $$

    where $B$ is the second virial coefficient, $P_c$ is the critical pressure,
    $T_c$ is the critical temperature, $T_r=T/T_c$ is the reduced temperature,
    and $\omega$ is the acentric factor.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 40.

    Parameters
    ----------
    T : FloatOrArray
        Temperature. Unit = K.
    Tc : float
        Critical temperature. Unit = K.
    Pc : float
        Critical pressure. Unit = Pa.
    w : float
        Acentric factor.

    Returns
    -------
    FloatOrArray
        Second virial coefficient, $B$. Unit = m³/mol.
    """
    Tr = T/Tc
    B0 = 0.083 - 0.422/Tr**1.6
    B1 = 0.139 - 0.172/Tr**4.2
    return R*Tc/Pc*(B0 + w*B1)


def B_mixture(T: float,
              Tc: FloatVector,
              Pc: FloatVector,
              Zc: FloatVector,
              w: FloatVector,
              ) -> FloatSquareMatrix:
    r"""Calculate the matrix of interaction virial coefficients using the
    mixing rules of Prausnitz.

    $$ \begin{aligned}
        B_{ij} &= B(T,T_{cij},P_{cij},\omega_{ij}) \\
        v_{cij} &= \frac{(v_{ci}^{1/3}+v_{cj}^{1/3})^3}{8} \\
        k_{ij} &= 1 -\frac{\sqrt{v_{ci}v_{cj}}}{v_{cij}} \\
        T_{cij} &= \sqrt{T_{ci}T_{cj}}(1-k_{ij}) \\
        Z_{cij} &= \frac{Z_{ci}+Z_{cj}}{2} \\
        \omega_{ij} &= \frac{\omega_{i}+\omega_{j}}{2} \\
        P_{cij} &= \frac{Z_{cij} R T_{cij}}{v_{cij}}
    \end{aligned} $$

    The calculation of the individual coefficients is handled by
    [`B_pure`](.#polykin.properties.eos.B_pure).

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 80.

    Parameters
    ----------
    T : float
        Temperature. Unit = K.
    Tc : FloatVector
        Critical temperatures of all components. Unit = K.
    Pc : FloatVector
        Critical pressures of all components. Unit = Pa.
    Zc : FloatVector
        Critical compressibility factors of all components.
    w : FloatVector
        Acentric factors of all components.

    Returns
    -------
    FloatSquareMatrix
        Matrix of interaction virial coefficients $B_{ij}$. Unit = m³/mol.
    """
    N = Tc.size
    vc = Zc*R*Tc/Pc
    B = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            if i == j:
                B[i, j] = B_pure(T, Tc[i],  Pc[i], w[i])
            else:
                vcm = (vc[i]**(1/3) + vc[j]**(1/3))**3 / 8
                km = 1. - sqrt(vc[i]*vc[j])/vcm
                Tcm = sqrt(Tc[i]*Tc[j])*(1. - km)
                Zcm = (Zc[i] + Zc[j])/2
                wm = (w[i] + w[j])/2
                Pcm = Zcm*R*Tcm/vcm
                B[i, j] = B_pure(T, Tcm, Pcm, wm)
                B[j, i] = B[i, j]
    return B
