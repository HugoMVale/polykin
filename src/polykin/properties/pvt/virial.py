# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025


import numpy as np
from numpy import sqrt
from scipy.constants import R

from polykin.utils.types import (
    FloatArray,
    FloatSquareMatrix,
    FloatVector,
)

__all__ = ["B_pure", "B_mixture"]


def B_pure(
    T: float | FloatArray,
    Tc: float,
    Pc: float,
    w: float,
) -> float | FloatArray:
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
    Tr = T / Tc
    B0 = 0.083 - 0.422 / Tr**1.6
    B1 = 0.139 - 0.172 / Tr**4.2
    return R * Tc / Pc * (B0 + w * B1)


def B_mixture(
    T: float,
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
    vc = Zc * R * Tc / Pc
    N = Tc.size
    B = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            if i == j:
                B[i, j] = B_pure(T, Tc[i], Pc[i], w[i])
            else:
                vcm = (vc[i] ** (1 / 3) + vc[j] ** (1 / 3)) ** 3 / 8
                km = 1 - sqrt(vc[i] * vc[j]) / vcm
                Tcm = sqrt(Tc[i] * Tc[j]) * (1 - km)
                Zcm = (Zc[i] + Zc[j]) / 2
                wm = (w[i] + w[j]) / 2
                Pcm = Zcm * R * Tcm / vcm
                B[i, j] = B_pure(T, Tcm, Pcm, wm)
                B[j, i] = B[i, j]
    return B
