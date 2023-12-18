# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector, FloatVectorLike, FloatSquareMatrix
from polykin.utils import eps
from ..mixing_rules import geometric_interaction_mixing
from .base import EoS

import numpy as np
from numpy import sqrt, log, dot
from scipy.constants import R
from typing import Optional, Literal
from abc import abstractmethod

__all__ = ['Cubic',
           'RedlichKwong',
           'Soave',
           'PengRobinson']


class Cubic(EoS):

    Tc: FloatVector
    Pc: FloatVector
    w: FloatVector
    k: Optional[FloatSquareMatrix]
    _u: float
    _w: float
    _a: float
    _b: float

    def __init__(self,
                 Tc: FloatVectorLike,
                 Pc: FloatVectorLike,
                 w: FloatVectorLike,
                 k: Optional[FloatSquareMatrix] = None
                 ) -> None:

        if isinstance(Tc, (list, tuple)):
            Tc = np.array(Tc, dtype=np.float64)
        if isinstance(Pc, (list, tuple)):
            Pc = np.array(Pc, dtype=np.float64)
        if isinstance(w, (list, tuple)):
            w = np.array(w, dtype=np.float64)

        self.Tc = Tc
        self.Pc = Pc
        self.w = w
        self.k = k
        self.b = self._b*R*Tc/Pc

    def Z(self, T, P, y):
        A = self.am(T, y)*P/(R*T)**2
        B = self.bm(y)*P/(R*T)
        return Z_cubic_root(self._u, self._w, A, B)

    def P(self, T, V, y):
        r"""Calculate the equilibrium pressure of the fluid.

        $$ P = \frac{RT}{V - b_m} -\frac{a_m}{V^2 + u V b_m + w b_m^2} $$

        where $P$ is the pressure, $T$ is the temperature, $V$ is the molar
        volume, $a_m(T,y)$ and $b_m(y)$ are the mixture EOS parameters, and
        $y$ is the vector of mole fractions.

        | Equation      | $u$ | $w$ |
        |---------------|:---:|:---:|
        | Redlich-Kwong |  1  |  0  |
        | Soave         |  1  |  0  |
        | Peng-Robinson | 2   | -1  |

        Reference:

        * RC Reid, JM Prausniz, and BE Poling. The properties of gases &
        liquids 4th edition, 1986, p. 42-43.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        V : float
            Molar volume. Unit = m³/mol.
        y : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector
            Pressure. Unit = Pa.
        """
        am = 1
        bm = 1
        u = self._u
        w = self._w
        return R*T/(V - bm) - am/(V**2 + u*V*bm + w*bm**2)

    def am(self,
           T: float,
           y: FloatVector) -> float:
        r"""Calculate mixture attractive parameter from the corresponding
        pure-component parameters.

        $$ a_m = \sum_i \sum_j y_i y_j (a_i a_j)^{1/2} (1 - \bar{k}_{ij}) $$

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
            Mixture attractive parameter, $a_m$. Unit = J·m³.
        """
        return geometric_interaction_mixing(y, self.a(T), self.k)

    def bm(self,
           y: FloatVector
           ) -> float:
        r"""Calculate mixture repulsive parameter from the corresponding
        pure-component parameters.

        $$ b_m = \sum_i y_i b_i $$

        Reference:

        * RC Reid, JM Prausniz, and BE Poling. The properties of gases &
        liquids 4th edition, 1986, p. 82.

        Parameters
        ----------
        y : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Mixture repulsive parameter, $b_m$. Unit = m³.
        """
        return dot(y, self.b)

    @abstractmethod
    def a(self, T: float) -> FloatVector:
        pass


class RedlichKwong(Cubic):

    _u = 1.
    _w = 0.
    _b = 0.08664

    def a(self, T: float) -> FloatVector:
        Tc = self.Tc
        Pc = self.Pc
        Tr = T/Tc
        a = (R*Tc)**2 / Pc
        a *= 0.42748/sqrt(Tr)
        return a


class Soave(Cubic):

    _u = 1.
    _w = 0.
    _b = 0.08664

    def a(self, T: float) -> FloatVector:
        Tc = self.Tc
        Pc = self.Pc
        w = self.w
        Tr = T/Tc
        fw = 0.48 + 1.574*w - 0.176*w**2
        a = (R*Tc)**2 / Pc
        a *= 0.42748*(1 + fw*(1 - sqrt(Tr)))**2
        return a


class PengRobinson(Cubic):

    _u = 2.
    _w = -1.
    _b = 0.07780

    def a(self, T: float) -> FloatVector:
        Tc = self.Tc
        Pc = self.Pc
        w = self.w
        Tr = T/Tc
        fw = 0.37464 + 1.54226*w - 0.26992*w**2
        a = (R*Tc)**2 / Pc
        a *= 0.45724*(1 + fw*(1 - sqrt(Tr)))**2
        return a


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

# # %%

# T = 398.15
# P = 100e5
# Tc = np.array([364.9])
# Pc = np.array([46.0e5])
# w = np.array([0.144])
# y = np.array([1.])
# Z = Z_cubic(T, P, y, Tc, Pc, w)
