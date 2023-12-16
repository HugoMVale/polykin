# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector, FloatVectorLike, FloatOrArray, \
    FloatSquareMatrix
from .base import EoS
from ..mixing_rules import quadratic_mixing_rule

import numpy as np
from numpy import sqrt, exp
from scipy.constants import R
import functools

__all__ = ['Virial',
           'B_vanness_abbott',
           'B_matrix']

# %% Virial equation


class Virial(EoS):

    def __init__(self,
                 Tc: FloatVectorLike,
                 Pc: FloatVectorLike,
                 Zc: FloatVectorLike,
                 w: FloatVectorLike
                 ) -> None:
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
        Bm = self.Bm(T, y)
        return 1. + Bm*P/(R*T)

    def P(self, T, V, y):
        Bm = self.Bm(T, y)
        return R*T/(V - Bm)

    def Bm(self,
           T: float,
           y: FloatVector
           ) -> float:
        r"""Calculate the second virial coefficient for the mixture.

        $$ B_m = sum_i sum_j y_i y_j B_{i,j} $$

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
        return quadratic_mixing_rule(y, self.Bij(T))

    @functools.cache
    def Bij(self,
            T: float,
            ) -> FloatSquareMatrix:
        return B_matrix(T, self.Tc, self.Pc, self.Zc, self.w)

    def phi(self,
            T: float,
            P: float,
            y: FloatVector
            ) -> FloatVector:
        Bm = self.Bm(T, y)
        B = self.Bij(T)
        return exp((2*np.dot(B, y) - Bm)*P/(R*T))

    def DAS(self, T, P, y):
        pass


# %% Second virial coefficient


def B_vanness_abbott(T: FloatOrArray,
                     Tc: float,
                     Pc: float,
                     w: float) -> FloatOrArray:
    r"""Estimate the second virial coefficient for a nonpolar or slightly polar
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


def B_matrix(T: float,
             Tc: FloatVector,
             Pc: FloatVector,
             Zc: FloatVector,
             w: FloatVector,
             ) -> FloatSquareMatrix:
    r"""Calculate matrix of interaction virial coefficients using the mixing
    rules of Prausnitz.

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
    Vc = Zc*R*Tc/Pc
    B = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                Tcm = Tc[i]
                Pcm = Pc[i]
                Zcm = Zc[i]
                Vcm = Vc[i]
                wm = w[i]
            else:
                Vcm = (Vc[i]**(1/3) + Vc[j]**(1/3))**3 / 8
                km = 1. - sqrt(Vc[i]*Vc[j])/Vcm
                Tcm = sqrt(Tc[i]*Tc[j])*(1. - km)
                Zcm = (Zc[i] + Zc[j])/2
                wm = (w[i] + w[j])/2
                Pcm = Zcm*R*Tcm/Vcm
            B[i, j] = B_vanness_abbott(T, Tcm, Pcm, wm)
    return B
