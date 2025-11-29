# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

import functools
from typing import Union

import numpy as np
from numpy import dot, exp, log, sqrt
from scipy.constants import R

from polykin.properties.pvt.mixing_rules import quadratic_mixing
from polykin.utils.math import convert_FloatOrVectorLike_to_FloatVector
from polykin.utils.types import (FloatArray, FloatSquareMatrix, FloatVector,
                                 FloatVectorLike)

from .base import GasEoS

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

    def __init__(self,
                 Tc: Union[float, FloatVectorLike],
                 Pc: Union[float, FloatVectorLike],
                 Zc: Union[float, FloatVectorLike],
                 w: Union[float, FloatVectorLike],
                 name: str = ''
                 ) -> None:

        Tc, Pc, Zc, w = \
            convert_FloatOrVectorLike_to_FloatVector([Tc, Pc, Zc, w])

        N = len(Tc)
        super().__init__(N, name)
        self.Tc = Tc
        self.Pc = Pc
        self.Zc = Zc
        self.w = w

    def Bm(self,
           T: float,
           y: FloatVector
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
    def Bij(self,
            T: float,
            ) -> FloatSquareMatrix:
        r"""Calculate the matrix of interaction virial coefficients.

        The calculation is handled by [`B_mixture`](B_mixture.md).

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

    def Z(self,
          T: float,
          P: float,
          y: FloatVector
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
        return 1.0 + Bm*P/(R*T)

    def P(self,
          T: float,
          v: float,
          y: FloatVector
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
        return R*T/(v - Bm)

    def phi(self,
            T: float,
            P: float,
            y: FloatVector
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
        return exp((2*dot(B, y) - Bm)*P/(R*T))

    def DA(self,
           T: float,
           V: float,
           n: FloatVector,
           v0: float
           ) -> float:
        nT = n.sum()
        y = n/nT
        Bm = self.Bm(T, y)
        return -nT*R*T*log((V - nT*Bm)/(nT*v0))

# %% Second virial coefficient


def B_pure(T: Union[float, FloatArray],
           Tc: float,
           Pc: float,
           w: float
           ) -> Union[float, FloatArray]:
    r"""Estimate the second virial coefficient of a nonpolar or slightly polar
    gas.

    $$ \frac{B P_c}{R T_c} = B^{(0)}(T_r) + \omega B^{(1)}(T_r) $$

    where $B$ is the second virial coefficient, $P_c$ is the critical pressure,
    $T_c$ is the critical temperature, $T_r=T/T_c$ is the reduced temperature,
    and $\omega$ is the acentric factor.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 40.

    Parameters
    ----------
    T : float | FloatArray
        Temperature [K].
    Tc : float
        Critical temperature [K].
    Pc : float
        Critical pressure [Pa].
    w : float
        Acentric factor.

    Returns
    -------
    float | FloatArray
        Second virial coefficient, $B$ [m³/mol].
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

    \begin{aligned}
        B_{ij} &= B(T,T_{cij},P_{cij},\omega_{ij}) \\
        v_{cij} &= \frac{(v_{ci}^{1/3}+v_{cj}^{1/3})^3}{8} \\
        k_{ij} &= 1 -\frac{\sqrt{v_{ci}v_{cj}}}{v_{cij}} \\
        T_{cij} &= \sqrt{T_{ci}T_{cj}}(1-k_{ij}) \\
        Z_{cij} &= \frac{Z_{ci}+Z_{cj}}{2} \\
        \omega_{ij} &= \frac{\omega_{i}+\omega_{j}}{2} \\
        P_{cij} &= \frac{Z_{cij} R T_{cij}}{v_{cij}}
    \end{aligned}

    The calculation of the individual coefficients is handled by
    [`B_pure`](B_pure.md).

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 80.

    Parameters
    ----------
    T : float
        Temperature [K].
    Tc : FloatVector (N)
        Critical temperatures of all components [K].
    Pc : FloatVector (N)
        Critical pressures of all components [Pa].
    Zc : FloatVector (N)
        Critical compressibility factors of all components.
    w : FloatVector (N)
        Acentric factors of all components.

    Returns
    -------
    FloatSquareMatrix (N,N)
        Matrix of interaction virial coefficients $B_{ij}$ [m³/mol].
    """
    vc = Zc*R*Tc/Pc
    N = Tc.size
    B = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            if i == j:
                B[i, j] = B_pure(T, Tc[i],  Pc[i], w[i])
            else:
                vcm = (vc[i]**(1/3) + vc[j]**(1/3))**3 / 8
                km = 1 - sqrt(vc[i]*vc[j])/vcm
                Tcm = sqrt(Tc[i]*Tc[j])*(1 - km)
                Zcm = (Zc[i] + Zc[j])/2
                wm = (w[i] + w[j])/2
                Pcm = Zcm*R*Tcm/vcm
                B[i, j] = B_pure(T, Tcm, Pcm, wm)
                B[j, i] = B[i, j]
    return B
