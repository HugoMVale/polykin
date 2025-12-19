# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import functools
from abc import abstractmethod
from collections.abc import Iterable
from typing import Literal

# import matplotlib.pyplot as plt
import numpy as np

# from matplotlib.axes._axes import Axes
# from matplotlib.figure import Figure
from numpy import dot, exp, log, sqrt
from scipy.constants import R

from polykin.math import fixpoint_wegstein
from polykin.properties.pvt.mixing_rules import geometric_interaction_mixing
from polykin.properties.vaporization import PL_Wilson
from polykin.utils.exceptions import ConvergenceError
from polykin.utils.math import convert_FloatOrVectorLike_to_FloatVector, eps
from polykin.utils.types import FloatSquareMatrix, FloatVector, FloatVectorLike

from .base import GasLiquidEoS

__all__ = [
    "CubicEoS",
    "PengRobinson",
    "RedlichKwong",
    "SoaveRedlichKwong",
    "Z_cubic_roots",
]


class CubicEoS(GasLiquidEoS):
    r"""Base class for cubic equations of state.

    This abstract class represents a general two-parameter cubic EoS of the
    form:

    $$ P = \frac{R T}{v - b_m} - \frac{a_m}{v^2 + u b_m v + w b_m^2} $$

    where $P$ is the pressure, $T$ is the temperature, $v$ is the molar
    volume, $a_m(T,z)$ and $b_m(z)$ are the mixture EoS parameters, $z$ is the
    vector of mole fractions, and $u$ and $w$ are numerical constants that
    determine the specific EoS.

    For a pure component, the parameters $a$ and $b$ are given by:

    \begin{aligned}
    a &= \Omega_a \left( \frac{R^2 T_{c}^2}{P_{c}} \right) \alpha(T) \\
    b &= \Omega_b \left( \frac{R T_{c}}{P_{c}} \right)
    \end{aligned}

    where $T_c$ is the critical temperature, $P_c$ is the critical pressure,
    $\Omega_a$ and $\Omega_b$ are numerical constants that determine the
    specific EoS, and $\alpha(T)$ is a dimensionless temperature-dependent
    function.

    To implement a specific cubic EoS, subclasses must:

    * Define the class-level constants `_Ωa`, `_Ωb`, `_u` and `_w`.
    * Implement the temperature-dependent function `_alpha(T)`.
    * Optionally override the mixing-rule functions `am(T, z)` and `bm(z)`.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
        liquids 4th edition, 1986, p. 42.
    """

    Tc: FloatVector
    Pc: FloatVector
    w: FloatVector
    k: FloatSquareMatrix | None
    _u: float
    _w: float
    _Ωa: float
    _Ωb: float

    def __init__(
        self,
        Tc: float | FloatVectorLike,
        Pc: float | FloatVectorLike,
        w: float | FloatVectorLike,
        k: FloatSquareMatrix | None,
        name: str = "",
    ) -> None:

        Tc, Pc, w = convert_FloatOrVectorLike_to_FloatVector([Tc, Pc, w])

        N = len(Tc)
        super().__init__(N, name)
        self.Tc = Tc
        self.Pc = Pc
        self.w = w
        self.k = k

    @abstractmethod
    def _alpha(self, T: float) -> FloatVector:
        pass

    @functools.cache
    def a(self, T: float) -> FloatVector:
        r"""Calculate the attractive parameters of the pure-components that
        make up the mixture.

        Parameters
        ----------
        T : float
            Temperature [K].

        Returns
        -------
        FloatVector (N)
            Attractive parameters of all components, $a_i$ [J·m³].
        """
        return self._Ωa * (R * self.Tc) ** 2 / self.Pc * self._alpha(T)

    @functools.cached_property
    def b(self) -> FloatVector:
        r"""Calculate the repulsive parameters of the pure-components that
        make up the mixture.

        Returns
        -------
        FloatVector (N)
            Repulsive parameters of all components, $b_i$ [m³/mol].
        """
        return self._Ωb * R * self.Tc / self.Pc

    def am(self, T: float, z: FloatVector) -> float:
        r"""Calculate the mixture attractive parameter from the corresponding
        pure-component parameters.

        $$ a_m = \sum_i \sum_j z_i z_j (a_i a_j)^{1/2} (1 - \bar{k}_{ij}) $$

        **References**

        *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
            liquids 4th edition, 1986, p. 82.

        Parameters
        ----------
        T : float
            Temperature [K].
        z : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Mixture attractive parameter, $a_m$ [J·m³].
        """
        return geometric_interaction_mixing(z, self.a(T), self.k)

    def bm(self, z: FloatVector) -> float:
        r"""Calculate the mixture repulsive parameter from the corresponding
        pure-component parameters.

        $$ b_m = \sum_i z_i b_i $$

        **References**

        *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
            liquids 4th edition, 1986, p. 82.

        Parameters
        ----------
        z : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Mixture repulsive parameter, $b_m$ [m³/mol].
        """
        return dot(z, self.b)

    def Bm(self, T: float, z: FloatVector) -> float:
        r"""Calculate the second virial coefficient of the mixture.

        $$ B_m = b_m - \frac{a_m}{R T} $$

        **References**

        *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
            liquids 4th edition, 1986, p. 82.

        Parameters
        ----------
        T : float
            Temperature [K].
        z : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Mixture second virial coefficient, $B_m$ [m³/mol].
        """
        return self.bm(z) - self.am(T, z) / (R * T)

    def P(
        self,
        T: float,
        v: float,
        z: FloatVector,
    ) -> float:
        r"""Calculate the pressure of the fluid.

        Parameters
        ----------
        T : float
            Temperature [K].
        v : float
            Molar volume [m³/mol].
        z : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        float
            Pressure [Pa].
        """
        u = self._u
        w = self._w
        am = self.am(T, z)
        bm = self.bm(z)
        return R * T / (v - bm) - am / (v**2 + u * v * bm + w * bm**2)

    def Z(
        self,
        T: float,
        P: float,
        z: FloatVector,
    ) -> FloatVector:
        r"""Calculate the compressibility factors for the possible phases of a
        fluid.

        The calculation is handled by [`Z_cubic_roots`](Z_cubic_roots.md).

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        z : FloatVector (N)
            Mole fractions of all components [mol/mol].

        Returns
        -------
        FloatVector
            Compressibility factors of the possible phases. If two phases
            are possible, the first result is the lowest value (liquid).
        """
        A = self.am(T, z) * P / (R * T) ** 2
        B = self.bm(z) * P / (R * T)
        Z = Z_cubic_roots(self._u, self._w, A, B)
        return Z

    def phi(
        self,
        T: float,
        P: float,
        z: FloatVector,
        phase: Literal["L", "V"],
    ) -> FloatVector:
        r"""Calculate the fugacity coefficients of all components in a given
        phase.

        For each component, the fugacity coefficient is given by:

        \begin{aligned}
        \ln \hat{\phi}_i &= \frac{b_i}{b_m}(Z-1)-\ln(Z-B^*)
        +\frac{A^*}{B^*\sqrt{u^2-4w}}\left(\frac{b_i}{b_m}
        -\delta_i\right)\ln{\frac{2Z+B^*(u+\sqrt{u^2-4w})}{2Z+B^*(u-\sqrt{u^2-4w})}} \\
        \delta_i &= \frac{2a_i^{1/2}}{a_m}\sum_j z_j a_j^{1/2}(1-\bar{k}_{ij})
        \end{aligned}

        **References**

        *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
            liquids 4th edition, 1986, p. 145.

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].
        z : FloatVector (N)
            Mole fractions of all components [mol/mol].
        phase : Literal['L', 'V']
            Phase of the fluid. Only relevant for systems where both liquid
            and vapor phases may exist.

        Returns
        -------
        FloatVector (N)
            Fugacity coefficients of all components.
        """
        u = self._u
        w = self._w
        d = sqrt(u**2 - 4 * w)
        a = self.a(T)
        am = self.am(T, z)
        b = self.b
        bm = self.bm(z)
        k = self.k
        A = am * P / (R * T) ** 2
        B = bm * P / (R * T)
        b_bm = b / bm
        Z = self.Z(T, P, z)

        if k is None:
            δ = 2 * sqrt(a / am)
        else:
            δ = np.sum(z * sqrt(a) * (1 - k), axis=1)

        if phase == "L":
            Zi = Z[0]
        elif phase == "V":
            Zi = Z[-1]
        else:
            raise ValueError(f"Invalid phase: {phase}.")

        ln_phi = (
            b_bm * (Zi - 1)
            - log(Zi - B)
            + A
            / (B * d)
            * (b_bm - δ)
            * log((2 * Zi + B * (u + d)) / (2 * Zi + B * (u - d)))
        )

        return exp(ln_phi)

    def Psat(
        self,
        T: float,
        Psat0: float | None = None,
    ) -> float:
        r"""Calculate the saturation pressure of the fluid.

        !!! note

            The saturation pressure is only defined for single-component
            systems. For multicomponent systems, a specific flash solver
            must be used: [polykin.thermo.flash](../flash/index.md).

        Parameters
        ----------
        T : float
            Temperature [K].
        Psat0 : float | None
            Initial guess for the saturation pressure [Pa]. By default, the
            value is estimated using the Wilson equation.

        Returns
        -------
        float
            Saturation pressure [Pa].
        """
        if self.N != 1:
            raise ValueError("Psat is only defined for single-component systems.")

        if T >= self.Tc[0]:
            raise ValueError(f"T >= Tc = {self.Tc[0]} K.")

        if Psat0 is None:
            Psat0 = PL_Wilson(T, self.Tc[0], self.Pc[0], self.w[0])

        # Solve as fixed-point problem (Newton fails when T is close to Tc)
        def g(x, T=T, z=np.array([1.0])):
            return x * self.K(T, x[0], z, z)

        sol = fixpoint_wegstein(g, np.array([Psat0]))

        if sol.success:
            return sol.x[0]
        else:
            raise ConvergenceError(f"Psat failed to converge. Solution: {sol}.")

    def DA(self, T, V, n, v0):
        nT = n.sum()
        z = n / nT
        u = self._u
        w = self._w
        d = sqrt(u**2 - 4 * w)
        am = self.am(T, z)
        bm = self.bm(z)

        return nT * am / (bm * d) * log(
            (2 * V + nT * bm * (u - d)) / (2 * V + nT * bm * (u + d))
        ) - nT * R * T * log((V - nT * bm) / (nT * v0))


class RedlichKwong(CubicEoS):
    r"""[Redlich-Kwong](https://en.wikipedia.org/wiki/Redlich%E2%80%93Kwong_equation_of_state)
    equation of state.

    This EoS is based on the following $P(v,T)$ relationship:

    $$ P = \frac{RT}{v - b_m} -\frac{a_m}{v (v + b_m)} $$

    where $P$ is the pressure, $T$ is the temperature, $v$ is the molar
    volume, $a_m(T,z)$ and $b_m(z)$ are the mixture EoS parameters, and
    $z$ is the vector of mole fractions.

    For a single component, the parameters $a$ and $b$ are given by:

    \begin{aligned}
        a &= 0.42748 \frac{R^2 T_{c}^2}{P_{c}} T_{r}^{-1/2} \\
        b &= 0.08664\frac{R T_{c}}{P_{c}}
    \end{aligned}

    where $T_c$ is the critical temperature, $P_c$ is the critical pressure,
    and $T_r = T/T_c$ is the reduced temperature.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
        liquids 4th edition, 1986, p. 37, 40, 80, 82.

    Parameters
    ----------
    Tc : float | FloatVectorLike (N)
        Critical temperatures of all components [K].
    Pc : float | FloatVectorLike (N)
        Critical pressures of all components [Pa].
    k : FloatSquareMatrix (N,N) | None
        Binary interaction parameter matrix.
    name : str
        Name.
    """

    _u = 1.0
    _w = 0.0
    _Ωa = 0.42748
    _Ωb = 0.08664

    def __init__(
        self,
        Tc: float | FloatVectorLike,
        Pc: float | FloatVectorLike,
        k: FloatSquareMatrix | None = None,
        name: str = "",
    ) -> None:

        w = np.zeros_like(Tc) if isinstance(Tc, Iterable) else 0.0
        super().__init__(Tc, Pc, w, k, name)

    def _alpha(self, T: float) -> FloatVector:
        return sqrt(self.Tc / T)


class SoaveRedlichKwong(CubicEoS):
    r"""[Soave-Redlich-Kwong](https://en.wikipedia.org/wiki/Cubic_equations_of_state#Soave_modification_of_Redlich%E2%80%93Kwong)
    equation of state.

    This EoS is based on the following $P(v,T)$ relationship:

    $$ P = \frac{RT}{v - b_m} -\frac{a_m}{v (v + b_m)} $$

    where $P$ is the pressure, $T$ is the temperature, $v$ is the molar
    volume, $a_m(T,z)$ and $b_m(z)$ are the mixture EoS parameters, and
    $z$ is the vector of mole fractions.

    For a single component, the parameters $a$ and $b$ are given by:

    \begin{aligned}
    a &= 0.42748 \frac{R^2 T_{c}^2}{P_{c}} [1 + f_\omega(1 - T_{r}^{1/2})]^2 \\
    f_\omega &= 0.48 + 1.574\omega -0.176\omega^2 \\
    b &= 0.08664\frac{R T_{c}}{P_{c}}
    \end{aligned}

    where $T_c$ is the critical temperature, $P_c$ is the critical pressure,
    and $T_r = T/T_c$ is the reduced temperature.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
        liquids 4th edition, 1986, p. 37, 40, 80, 82.

    Parameters
    ----------
    Tc : float | FloatVectorLike (N)
        Critical temperatures of all components [K].
    Pc : float | FloatVectorLike (N)
        Critical pressures of all components [Pa].
    w : float | FloatVectorLike (N)
        Acentric factors of all components.
    k : FloatSquareMatrix (N,N) | None
        Binary interaction parameter matrix.
    use_graboski : bool
        If `True`, use Graboski & Daubert's improved $f_\omega$ expression.
        Otherwise, use Soave's original expression.
    name : str
        Name.
    """

    _u = 1.0
    _w = 0.0
    _Ωa = 0.42748
    _Ωb = 0.08664
    use_graboski: bool

    def __init__(
        self,
        Tc: float | FloatVectorLike,
        Pc: float | FloatVectorLike,
        w: float | FloatVectorLike,
        k: FloatSquareMatrix | None = None,
        use_graboski: bool = True,
        name: str = "",
    ) -> None:

        super().__init__(Tc, Pc, w, k, name)
        self.use_graboski = use_graboski

    def _alpha(self, T: float) -> FloatVector:
        w = self.w
        Tr = T / self.Tc

        if self.use_graboski:
            fw = 0.48508 + 1.55171 * w - 0.1561 * w**2
        else:
            fw = 0.48 + 1.574 * w - 0.176 * w**2

        return (1.0 + fw * (1.0 - sqrt(Tr))) ** 2


class PengRobinson(CubicEoS):
    r"""[Peng-Robinson](https://en.wikipedia.org/wiki/Cubic_equations_of_state#Peng%E2%80%93Robinson_equation_of_state)
    equation of state.

    This EoS is based on the following $P(v,T)$ relationship:

    $$ P = \frac{RT}{v - b_m} -\frac{a_m}{v^2 + 2 v b_m - b_m^2} $$

    where $P$ is the pressure, $T$ is the temperature, $v$ is the molar
    volume, $a_m(T,z)$ and $b_m(z)$ are the mixture EoS parameters, and
    $z$ is the vector of mole fractions.

    For a single component, the parameters $a$ and $b$ are given by:

    \begin{aligned}
    a &= 0.45724 \frac{R^2 T_{c}^2}{P_{c}} [1 + f_\omega(1 - T_{r}^{1/2})]^2 \\
    f_\omega &= 0.37464 + 1.54226\omega - 0.26992\omega^2 \\
    b &= 0.07780\frac{R T_{c}}{P_{c}}
    \end{aligned}

    where $T_c$ is the critical temperature, $P_c$ is the critical pressure,
    and $T_r = T/T_c$ is the reduced temperature.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
        liquids 4th edition, 1986, p. 37, 40, 80, 82.

    Parameters
    ----------
    Tc : float | FloatVectorLike (N)
        Critical temperatures of all components [K].
    Pc : float | FloatVectorLike (N)
        Critical pressures of all components [Pa].
    w : float | FloatVectorLike (N)
        Acentric factors of all components.
    k : FloatSquareMatrix (N,N) | None
        Binary interaction parameter matrix.
    name : str
        Name.
    """

    _u = 2.0
    _w = -1.0
    _Ωa = 0.45724
    _Ωb = 0.07780

    def __init__(
        self,
        Tc: float | FloatVectorLike,
        Pc: float | FloatVectorLike,
        w: float | FloatVectorLike,
        k: FloatSquareMatrix | None = None,
        name: str = "",
    ) -> None:

        super().__init__(Tc, Pc, w, k, name)

    def _alpha(self, T: float) -> FloatVector:
        w = self.w
        Tr = T / self.Tc
        fw = 0.37464 + 1.54226 * w - 0.26992 * w**2
        return (1.0 + fw * (1.0 - sqrt(Tr))) ** 2


def Z_cubic_roots(
    u: float,
    w: float,
    A: float,
    B: float,
) -> FloatVector:
    r"""Find the compressibility factor roots of a cubic EoS.

    \begin{gathered}
        Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0 \\
        c_2 = -(1 + B - u B) \\
        c_1 = A + w B^2 - u B - u B^2 \\
        c_0 = -(A B + w B^2 + w B^3) \\
        A = \frac{a_m P}{R^2 T^2} \\
        B = \frac{b_m P}{R T}
    \end{gathered}

    | Equation            | $u$ | $w$ |
    |---------------------|:---:|:---:|
    | Redlich-Kwong       |  1  |  0  |
    | Soave-Redlich-Kwong |  1  |  0  |
    | Peng-Robinson       |  2  | -1  |

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases &
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
    c3 = 1.0
    c2 = -(1 + B - u * B)
    c1 = A + w * B**2 - u * B - u * B**2
    c0 = -(A * B + w * B**2 + w * B**3)

    roots = np.roots((c3, c2, c1, c0))
    roots = [x.real for x in roots if (abs(x.imag) < eps and x.real > B)]

    Z = [min(roots)]
    if len(roots) > 1:
        Z.append(max(roots))

    return np.array(Z)
