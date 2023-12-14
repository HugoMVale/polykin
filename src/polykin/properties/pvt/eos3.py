# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector, FloatSquareMatrix
from polykin.utils import eps

import numpy as np
from scipy.constants import R
from abc import ABC, abstractmethod
from typing import Optional, Literal
from math import sqrt

__all__ = ['Z_cubic']


# class EoS(ABC):

#     T: float
#     P: float
#     x: FloatVector

#     @abstractmethod
#     def DA(self) -> FloatOrArray:
#         """Helmholtz energy departure."""
#         pass

#     @abstractmethod
#     def DS(self) -> FloatOrArray:
#         """Entropy departure."""
#         pass

#     @abstractmethod
#     def Z(self) -> FloatOrArray:
#         """Compressibility factor."""
#         pass

#     def DG(self) -> FloatOrArray:
#         """Gibbs energy departure."""
#         DA = self.DA()
#         Z = self.Z()
#         T = self.T
#         return DA + R*T*(Z - 1)

#     def DH(self) -> FloatOrArray:
#         """Enthalpy departure."""
#         DA = self.DA()
#         DS = self.DS()
#         Z = self.Z()
#         T = self.T
#         return DA + T*DS + R*T*(Z - 1)

#     def DU(self) -> FloatOrArray:
#         """Internal energy departure."""
#         DA = self.DA()
#         DS = self.DS()
#         T = self.T
#         return DA + T*DS


# class CubicEoS(EoS):
#     u: float
#     w: float

#     def fw(w: float) -> float:
#         return 1.

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


def Z_cubic_root(U: float,
                 W: float,
                 A: float,
                 B: float
                 ) -> tuple[float, float]:
    c3 = 1.
    c2 = -(1. + B - U*B)
    c1 = (A + W*B**2 - U*B - U*B**2)
    c0 = -(A*B + W*B**2 + W*B**3)

    coeffs = (c3, c2, c1, c0)
    roots = np.roots(coeffs)
    roots = [x.real for x in roots if abs(x.imag) < eps]

    Z1 = min(roots)
    if len(roots) > 1:
        Z2 = max(roots)
    else:
        Z2 = np.nan

    return (Z1, Z2)


def Z_cubic(T: float,
            P: float,
            y: FloatVector,
            Tc: FloatVector,
            Pc: FloatVector,
            w: Optional[FloatVector] = None,
            k=None,
            method: Literal['RK', 'SRK', 'PR'] = 'SRK'):

    Tr = T/Tc
    if method == "RK":
        U = 1.
        W = 0.
        b = 0.08664
        a = 0.42748/np.sqrt(Tr)
        fw = 1
    elif method == "SRK" and w is not None:
        U = 1.
        W = 0.
        b = 0.08664
        fw = 0.48 + 1.574*w - 0.176*w**2
        a = 0.42748*(1 + fw*(1 - np.sqrt(Tr)))**2
    elif method == "PR" and w is not None:
        U = 2.
        W = -1.
        b = 0.07780
        fw = 0.37464 + 1.54226*w - 0.26992*w**2
        a = 0.45724*(1 + fw*(1 - np.sqrt(Tr)))**2
    else:
        raise ValueError(f"Invalid method: `{method}`.")

    b = b*R*Tc/Pc
    a *= (R*Tc)**2 / Pc

    # Mixing rules
    a_m, b_m = mixing_rules_cubic(y, a, b, k)

    # Z
    A = a_m*P/(R*T)**2
    B = b_m*P/(R*T)
    Z = Z_cubic_root(U, W, A, B)

    return Z


def mixing_rules_cubic(y: FloatVector,
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
        a_m = (np.dot(y, np.sqrt(a)))**2
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
