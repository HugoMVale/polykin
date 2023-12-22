# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector, FloatVectorLike, FloatSquareMatrix
from polykin.utils import eps
from ..mixing_rules import geometric_interaction_mixing
from .base import GasAndLiquidEoS

import numpy as np
from numpy import sqrt, dot
from scipy.constants import R
from typing import Optional
from abc import abstractmethod
import functools

__all__ = ['RedlichKwong',
           'Soave',
           'PengRobinson',
           'Z_cubic_roots']


class Cubic(GasAndLiquidEoS):

    Tc: FloatVector
    Pc: FloatVector
    k: Optional[FloatSquareMatrix]
    _u: float
    _w: float
    _Ωa: float
    _Ωb: float

    def __init__(self,
                 Tc: FloatVectorLike,
                 Pc: FloatVectorLike,
                 k: Optional[FloatSquareMatrix] = None
                 ) -> None:
        """Construct `Cubic` with the given parameters."""

        if isinstance(Tc, (list, tuple)):
            Tc = np.array(Tc, dtype=np.float64)
        if isinstance(Pc, (list, tuple)):
            Pc = np.array(Pc, dtype=np.float64)

        self.Tc = Tc
        self.Pc = Pc
        self.k = k

    @functools.cache
    def a(self, T: float) -> FloatVector:
        r"""Calculate the attractive parameters of the pure-components that
        make up the mixture.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.

        Returns
        -------
        FloatVector
            Attractive parameters of all components, $a_i$. Unit = J·m³.
        """
        return self._Ωa * (R*self.Tc)**2 / self.Pc * self._alpha(T)

    @functools.cached_property
    def b(self) -> FloatVector:
        r"""Calculate the repulsive parameters of the pure-components that
        make up the mixture.

        Returns
        -------
        FloatVector
            Repulsive parameters of all components, $b_i$. Unit = m³/mol.
        """
        return self._Ωb*R*self.Tc/self.Pc

    def am(self,
           T: float,
           y: FloatVector) -> float:
        r"""Calculate the mixture attractive parameter from the corresponding
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
        r"""Calculate the mixture repulsive parameter from the corresponding
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
            Mixture repulsive parameter, $b_m$. Unit = m³/mol.
        """
        return dot(y, self.b)

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
        am = self.am(T, y)
        bm = self.bm(y)
        u = self._u
        w = self._w
        return R*T/(v - bm) - am/(v**2 + u*v*bm + w*bm**2)

    def Z(self, T, P, y):
        r"""Calculate the compressibility factors of the coexisting phases a
        fluid.

        The calculation is handled by
        [`Z_cubic_roots`](RedlichKwong.md#polykin.properties.eos.Z_cubic_roots).

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
            Compressibility factor of the gas and/or liquid phases.
        """
        A = self.am(T, y)*P/(R*T)**2
        B = self.bm(y)*P/(R*T)
        return Z_cubic_roots(self._u, self._w, A, B)

    def Bm(self,
           T: float,
           y: FloatVector
           ) -> float:
        r"""Calculate the second virial coefficient of the mixture.

        $$ B_m = b_m - \frac{a_m}{R T} $$

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
        return self.bm(y) - self.am(T, y)/(R*T)

    @abstractmethod
    def _alpha(self, T: float) -> FloatVector:
        pass

    def DA(self):
        pass

    def phi(self):
        pass


class RedlichKwong(Cubic):
    r"""[Redlich-Kwong](https://en.wikipedia.org/wiki/Redlich%E2%80%93Kwong_equation_of_state)
    equation of state.

    This EOS is based on the following $P(v,T)$ relationship:

    $$ P = \frac{RT}{v - b_m} -\frac{a_m}{v (v + b_m)} $$

    where $P$ is the pressure, $T$ is the temperature, $v$ is the molar
    volume, $a_m(T,y)$ and $b_m(y)$ are the mixture EOS parameters, and
    $y$ is the vector of mole fractions.

    For a single component, the parameters $a$ and $b$ are given by:

    \begin{aligned}
    a &= 0.42748 \frac{R^2 T_{c}^2}{P_{c}} T_{r}^{-1/2} \\
    b &= 0.08664\frac{R T_{c}}{P_{c}}
    \end{aligned}

    where $T_c$ is the critical temperature, $P_c$ is the critical pressure,
    and $T_r = T/T_c$ is the reduced temperature.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases &
    liquids 4th edition, 1986, p. 37, 40, 80, 82.

    Parameters
    ----------
    Tc : FloatVector
        Critical temperatures of all components. Unit = K.
    Pc : FloatVector
        Critical pressures of all components. Unit = Pa.
    k : FloatSquareMatrix | None
        Binary interaction parameter matrix.
    """

    _u = 1.
    _w = 0.
    _Ωa = 0.42748
    _Ωb = 0.08664

    def __init__(self,
                 Tc: FloatVectorLike,
                 Pc: FloatVectorLike,
                 k: Optional[FloatSquareMatrix] = None
                 ) -> None:
        """Construct `RedlichKwong` with the given parameters."""

        super().__init__(Tc, Pc, k)

    def _alpha(self, T: float) -> FloatVector:
        return sqrt(self.Tc/T)


class Soave(Cubic):
    r"""[Soave](https://en.wikipedia.org/wiki/Cubic_equations_of_state#Soave_modification_of_Redlich%E2%80%93Kwong)
    equation of state.

    This EOS is based on the following $P(v,T)$ relationship:

    $$ P = \frac{RT}{v - b_m} -\frac{a_m}{v (v + b_m)} $$

    where $P$ is the pressure, $T$ is the temperature, $v$ is the molar
    volume, $a_m(T,y)$ and $b_m(y)$ are the mixture EOS parameters, and
    $y$ is the vector of mole fractions.

    For a single component, the parameters $a$ and $b$ are given by:

    \begin{aligned}
    a &= 0.42748 \frac{R^2 T_{c}^2}{P_{c}} [1 + f_\omega(1 - T_{r}^{1/2})]^2 \\
    f_\omega &= 0.48 + 1.574\omega - 0.176\omega^2 \\
    b &= 0.08664\frac{R T_{c}}{P_{c}}
    \end{aligned}

    where $T_c$ is the critical temperature, $P_c$ is the critical pressure,
    and $T_r = T/T_c$ is the reduced temperature.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases &
    liquids 4th edition, 1986, p. 37, 40, 80, 82.

    Parameters
    ----------
    Tc : FloatVector
        Critical temperatures of all components. Unit = K.
    Pc : FloatVector
        Critical pressures of all components. Unit = Pa.
    w : FloatVector
        Acentric factors of all components.
    k : FloatSquareMatrix | None
        Binary interaction parameter matrix.
    """
    w: FloatVector
    _u = 1.
    _w = 0.
    _Ωa = 0.42748
    _Ωb = 0.08664

    def __init__(self,
                 Tc: FloatVectorLike,
                 Pc: FloatVectorLike,
                 w: FloatVectorLike,
                 k: Optional[FloatSquareMatrix] = None
                 ) -> None:
        """Construct `Soave` with the given parameters."""

        if isinstance(w, (list, tuple)):
            w = np.array(w, dtype=np.float64)

        self.w = w
        super().__init__(Tc, Pc, k)

    def _alpha(self, T: float) -> FloatVector:
        w = self.w
        Tr = T/self.Tc
        fw = 0.48 + 1.574*w - 0.176*w**2
        return (1 + fw*(1 - sqrt(Tr)))**2


class PengRobinson(Cubic):
    r"""[Peng-Robinson](https://en.wikipedia.org/wiki/Cubic_equations_of_state#Peng%E2%80%93Robinson_equation_of_state)
    equation of state.

    This EOS is based on the following $P(v,T)$ relationship:

    $$ P = \frac{RT}{v - b_m} -\frac{a_m}{v^2 + 2 v b_m - b_m^2} $$

    where $P$ is the pressure, $T$ is the temperature, $v$ is the molar
    volume, $a_m(T,y)$ and $b_m(y)$ are the mixture EOS parameters, and
    $y$ is the vector of mole fractions.

    For a single component, the parameters $a$ and $b$ are given by:

    \begin{aligned}
    a &= 0.45724 \frac{R^2 T_{c}^2}{P_{c}} [1 + f_\omega(1 - T_{r}^{1/2})]^2 \\
    f_\omega &= 0.37464 + 1.54226\omega - 0.26992\omega^2 \\
    b &= 0.07780\frac{R T_{c}}{P_{c}}
    \end{aligned}

    where $T_c$ is the critical temperature, $P_c$ is the critical pressure,
    and $T_r = T/T_c$ is the reduced temperature.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases &
    liquids 4th edition, 1986, p. 37, 40, 80, 82.

    Parameters
    ----------
    Tc : FloatVector
        Critical temperatures of all components. Unit = K.
    Pc : FloatVector
        Critical pressures of all components. Unit = Pa.
    w : FloatVector
        Acentric factors of all components.
    k : FloatSquareMatrix | None
        Binary interaction parameter matrix.
    """
    w: FloatVector
    _u = 2.
    _w = -1.
    _Ωa = 0.45724
    _Ωb = 0.07780

    def __init__(self,
                 Tc: FloatVectorLike,
                 Pc: FloatVectorLike,
                 w: FloatVectorLike,
                 k: Optional[FloatSquareMatrix] = None
                 ) -> None:
        """Construct `PengRobinson` with the given parameters."""

        if isinstance(w, (list, tuple)):
            w = np.array(w, dtype=np.float64)

        self.w = w
        super().__init__(Tc, Pc, k)

    def _alpha(self, T: float) -> FloatVector:
        w = self.w
        Tr = T/self.Tc
        fw = 0.37464 + 1.54226*w - 0.26992*w**2
        return (1 + fw*(1 - sqrt(Tr)))**2

# %%

    # Departures
    # dadT = 1.  # todo
    # P0 = 1  # fix!!!!
    # d = sqrt(zu**2 - 4*zw)
    # t1 = b_m*d
    # t2 = log((2*Z + B*(zu - d))/(2*Z + B*(zu + d)))
    # t3 = R*log((Z - B)*P0/P)
    # DA = a_m/t1*t2 - T*t3
    # DS = t3 - dadT/t1*t2
    # (DU, DH, DG) = departures(T, DA, DS, Z)


def Z_cubic_roots(u: float,
                  w: float,
                  A: float,
                  B: float
                  ) -> FloatVector:
    r"""Find the compressibility factor roots of a cubic EOS.

    \begin{gathered}
        Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0 \\
        c_2 = -(1 + B - u B) \\
        c_1 = A + w B^2 - u B - u B^2 \\
        c_0 = -(A B + w B^2 + w B^3) \\
        A = \frac{a_m P}{R^2 T^2} \\
        B = \frac{b_m P}{R T}
    \end{gathered}

    | Equation      | $u$ | $w$ |
    |---------------|:---:|:---:|
    | Redlich-Kwong |  1  |  0  |
    | Soave         |  1  |  0  |
    | Peng-Robinson |  2  | -1  |

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases &
    liquids 4th edition, 1986, p. 42.

    Parameters
    ----------
    u : float
        Parameter of polynomial equation.
    w : float
        Parameter of polynomial equation.
    A : float
        Parameter of polynomial equation.
    B : float
        Parameter of polynomial equation.

    Returns
    -------
    FloatVector
        Compressibility factor(s) of the coexisting phases. If there are two
        phases, the first result is the lowest value (liquid).
    """
    c3 = 1.
    c2 = -(1. + B - u*B)
    c1 = (A + w*B**2 - u*B - u*B**2)
    c0 = -(A*B + w*B**2 + w*B**3)

    coeffs = (c3, c2, c1, c0)
    roots = np.roots(coeffs)
    roots = [x.real for x in roots if abs(x.imag) < eps]

    Z = []
    Z.append(min(roots))
    if len(roots) > 1:
        Z.append(max(roots))

    return np.array(Z)
