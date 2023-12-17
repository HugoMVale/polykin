# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector, FloatSquareMatrix
from polykin.utils import eps
from .base import EoS

import numpy as np
from numpy import exp, sqrt, log
from scipy.constants import R
from typing import Optional, Literal

__all__ = ['Cubic']


class Cubic(EoS):

    def Z(self, T, P, y):
        return 1.

    def P(self, T, V, y):
        return 0  # R*T/(V - bm) - am/(V**2 + u*V*bm + w*bm**2)

#     def DA(self) -> FloatOrArray:
#         a = 1
#         b = 1
#         u = self.u
#         w = self.w
#         B = 1
#         d = sqrt(u**2 - 4*w)
#         Z = self.Z()
#         T = self.T
#         V = 1
#         V0 = 1
#         out = a/(b*d)*np.log((2*Z + B*(u - d))/(2*Z + B*(u + d))) - \
#             R*T*np.log(1 - B/Z) - R*T*np.log(V/V0)
#         return


# class RedlichKwong(CubicEoS):
#     pass


# class Soave(CubicEoS):
#     pass


def Z_cubic_root(zu: float,
                 zw: float,
                 A: float,
                 B: float
                 ) -> FloatVector:
    c3 = 1.
    c2 = -(1. + B - zu*B)
    c1 = (A + zw*B**2 - zu*B - zu*B**2)
    c0 = -(A*B + zw*B**2 + zw*B**3)

    coeffs = (c3, c2, c1, c0)
    roots = np.roots(coeffs)
    roots = [x.real for x in roots if abs(x.imag) < eps]

    Z = []
    Z.append(min(roots))
    if len(roots) > 1:
        Z.append(max(roots))

    return np.array(Z)


def Z_cubic(T: float,
            P: float,
            y: FloatVector,
            Tc: FloatVector,
            Pc: FloatVector,
            w: Optional[FloatVector] = None,
            k: Optional[FloatSquareMatrix] = None,
            method: Literal['RK', 'SRK', 'PR'] = 'SRK'):

    Tr = T/Tc
    if method == "RK":
        zu = 1.
        zw = 0.
        b = 0.08664
        a = 0.42748/sqrt(Tr)
        fw = 1
    elif method == "SRK" and w is not None:
        zu = 1.
        zw = 0.
        b = 0.08664
        fw = 0.48 + 1.574*w - 0.176*w**2
        a = 0.42748*(1 + fw*(1 - sqrt(Tr)))**2
    elif method == "PR" and w is not None:
        zu = 2.
        zw = -1.
        b = 0.07780
        fw = 0.37464 + 1.54226*w - 0.26992*w**2
        a = 0.45724*(1 + fw*(1 - sqrt(Tr)))**2
    else:
        raise ValueError(f"Invalid method: `{method}`.")

    b = b*R*Tc/Pc
    a *= (R*Tc)**2 / Pc

    # Mixing rules
    a_m, b_m = cubic_mixing_rules(y, a, b, k)

    # Z
    A = a_m*P/(R*T)**2
    B = b_m*P/(R*T)
    Z = Z_cubic_root(zu, zw, A, B)

    # Departures
    dadT = 1.  # todo
    P0 = 1  # fix!!!!
    d = sqrt(zu**2 - 4*zw)
    t1 = b_m*d
    t2 = log((2*Z + B*(zu - d))/(2*Z + B*(zu + d)))
    t3 = R*log((Z - B)*P0/P)
    DA = a_m/t1*t2 - T*t3
    DS = t3 - dadT/t1*t2
    (DU, DH, DG) = departures(T, DA, DS, Z)

    return Z


def cubic_mixing_rules(y: FloatVector,
                       a: FloatVector,
                       b: FloatVector,
                       k: Optional[FloatSquareMatrix] = None
                       ) -> tuple[float, float]:
    r"""Mixing rules for Redlich-Kwong-type equations of state.

    The mixing rules for all two-parameter cubic equations of state are:

    $$ \begin{aligned}
        a_{m} &= \sum_i \sum_j y_i y_j (a_i a_j)^{1/2} (1 - \bar{k}_{ij}) \\
        b_{m} &= \sum_i y_i b_i
    \end{aligned} $$

    where $a_i$ and $b_i$ are the pure-component parameters, and $\bar{k}_{ij}$
    are the optional binary interaction coefficients.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 82.

    Parameters
    ----------
    y : FloatVector
        Mole fractions of all components. Unit = mol/mol.
    a : FloatVector
        Interaction parameter for all components. Unit = J·m³.
    b : FloatVector
        Excluded volume parameter for all components. Unit = m³.
    k : FloatSquareMatrix | None
        Binary interaction parameter matrix.

    Returns
    -------
    tuple[float, float]
        Tuple with mixture parameters, ($a_m, b_m$). Unit = (J·m³, m³).
    """

    if k is None:
        a_m = (np.dot(y, sqrt(a)))**2
    else:
        # k += k.T
        # np.fill_diagonal(k, 1.)
        # a_m = np.sum(np.outer(y, y) * np.sqrt(np.outer(a, a))
        #              * (1. - k), dtype=np.float64)
        a_m = 0.
        N = len(y)
        for i in range(N):
            for j in range(N):
                a_m += y[i]*y[j]*sqrt(a[i]*a[j])*(1. - k[i, j])

    b_m = np.dot(y, b)

    return (a_m, b_m)


# # %%

# T = 398.15
# P = 100e5
# Tc = np.array([364.9])
# Pc = np.array([46.0e5])
# w = np.array([0.144])
# y = np.array([1.])
# Z = Z_cubic(T, P, y, Tc, Pc, w)
