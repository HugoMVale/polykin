# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

# This modules implements the most commonly used DIPPR equations.

from typing import Union

import numpy as np
from numpy import cosh, exp, log, sinh

from polykin.utils.types import FloatArray

from .base import PropertyEquationT

__all__ = ['DIPPR100',
           'DIPPR101',
           'DIPPR102',
           'DIPPR104',
           'DIPPR105',
           'DIPPR106',
           'DIPPR107']


class DIPPR(PropertyEquationT):
    """_Abstract_ class for all
    [DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)
    temperature-dependent equations."""
    pass


class DIPPRP4(DIPPR):
    """_Abstract_ class for DIPPR equations with 4 parameters (A-D)."""

    def __init__(self,
                 A: float,
                 B: float,
                 C: float,
                 D: float,
                 Tmin: float,
                 Tmax: float,
                 unit,
                 symbol,
                 name
                 ) -> None:

        self.p = {'A': A, 'B': B, 'C': C, 'D': D}
        super().__init__((Tmin, Tmax), unit, symbol, name)


class DIPPRP5(DIPPR):
    """_Abstract_ class for DIPPR equations with 5 parameters (A-E)."""

    def __init__(self,
                 A: float,
                 B: float,
                 C: float,
                 D: float,
                 E: float,
                 Tmin: float,
                 Tmax: float,
                 unit,
                 symbol,
                 name
                 ) -> None:

        self.p = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E}
        super().__init__((Tmin, Tmax), unit, symbol, name)


class DIPPR100(DIPPRP5):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-100 equation.

    This equation implements the following temperature dependence:

    $$ Y = A + B T + C T^2 + D T^3 + E T^4 $$

    where $A$ to $E$ are component-specific constants and $T$ is the absolute
    temperature.

    Parameters
    ----------
    A : float
        Parameter of equation.
    B : float
        Parameter of equation.
    C : float
        Parameter of equation.
    D : float
        Parameter of equation.
    E : float
        Parameter of equation.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of output variable $Y$.
    symbol : str
        Symbol of output variable $Y$.
    name : str
        Name.
    """

    _pinfo = {'A': ('#', True), 'B': ('#/K', True), 'C': ('#/K²', True),
              'D': ('#/K³', True), 'E': ('#/K⁴', True)}

    def __init__(self,
                 A: float = 0.,
                 B: float = 0.,
                 C: float = 0.,
                 D: float = 0.,
                 E: float = 0.,
                 Tmin: float = 0.0,
                 Tmax: float = np.inf,
                 unit: str = '-',
                 symbol: str = 'Y',
                 name: str = ''
                 ) -> None:

        super().__init__(A, B, C, D, E, Tmin, Tmax, unit, symbol, name)

    @staticmethod
    def equation(T: Union[float, FloatArray],
                 A: float,
                 B: float,
                 C: float,
                 D: float,
                 E: float
                 ) -> Union[float, FloatArray]:
        r"""DIPPR-100 equation."""
        return A + B*T + C*T**2 + D*T**3 + E*T**4


class DIPPR101(DIPPRP5):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-101 equation.

    This equation implements the following temperature dependence:

    $$ Y = \exp{\left(A + B / T + C \ln(T) + D T^E\right)} $$

    where $A$ to $E$ are component-specific constants and $T$ is the absolute
    temperature.

    Parameters
    ----------
    A : float
        Parameter of equation.
    B : float
        Parameter of equation.
    C : float
        Parameter of equation.
    D : float
        Parameter of equation.
    E : float
        Parameter of equation.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of output variable $Y$.
    symbol : str
        Symbol of output variable $Y$.
    name : str
        Name.
    """
    _pinfo = {'A': ('', True), 'B': ('K', True), 'C': ('', True),
              'D': ('', True), 'E': ('', True)}

    def __init__(self,
                 A: float,
                 B: float,
                 C: float = 0.,
                 D: float = 0.,
                 E: float = 0.,
                 Tmin: float = 0.0,
                 Tmax: float = np.inf,
                 unit: str = '-',
                 symbol: str = 'Y',
                 name: str = ''
                 ) -> None:

        super().__init__(A, B, C, D, E, Tmin, Tmax, unit, symbol, name)

    @staticmethod
    def equation(T: Union[float, FloatArray],
                 A: float,
                 B: float,
                 C: float,
                 D: float,
                 E: float
                 ) -> Union[float, FloatArray]:
        r"""DIPPR-101 equation."""
        return exp(A + B/T + C*log(T) + D*T**E)


class DIPPR102(DIPPRP4):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-102 equation.

    This equation implements the following temperature dependence:

    $$ Y = \frac{A T^B}{ 1 + C/T + D/T^2} $$

    where $A$ to $D$ are component-specific constants and $T$ is the absolute
    temperature.

    Parameters
    ----------
    A : float
        Parameter of equation.
    B : float
        Parameter of equation.
    C : float
        Parameter of equation.
    D : float
        Parameter of equation.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of output variable $Y$.
    symbol : str
        Symbol of output variable $Y$.
    name : str
        Name.
    """

    _pinfo = {'A': ('#', True), 'B': ('', True), 'C': ('K', True),
              'D': ('K²', True)}

    def __init__(self,
                 A: float,
                 B: float,
                 C: float = 0.,
                 D: float = 0.,
                 Tmin: float = 0.0,
                 Tmax: float = np.inf,
                 unit: str = '-',
                 symbol: str = 'Y',
                 name: str = ''
                 ) -> None:

        super().__init__(A, B, C, D, Tmin, Tmax, unit, symbol, name)

    @staticmethod
    def equation(T: Union[float, FloatArray],
                 A: float,
                 B: float,
                 C: float,
                 D: float
                 ) -> Union[float, FloatArray]:
        r"""DIPPR-102 equation."""
        return (A * T**B) / (1 + C/T + D/T**2)


class DIPPR104(DIPPRP5):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-104 equation.

    This equation implements the following temperature dependence:

    $$ Y = A + B/T + C/T^3 + D/T^8 + E/T^9 $$

    where $A$ to $E$ are component-specific constants and $T$ is the absolute
    temperature.

    Parameters
    ----------
    A : float
        Parameter of equation.
    B : float
        Parameter of equation.
    C : float
        Parameter of equation.
    D : float
        Parameter of equation.
    E : float
        Parameter of equation.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of output variable $Y$.
    symbol : str
        Symbol of output variable $Y$.
    name : str
        Name.
    """

    _pinfo = {'A': ('#', True), 'B': ('#·K', True), 'C': ('#·K³', True),
              'D': ('#·K⁸', True), 'E': ('#·K⁹', True)}

    def __init__(self,
                 A: float,
                 B: float,
                 C: float = 0.,
                 D: float = 0.,
                 E: float = 0.,
                 Tmin: float = 0.0,
                 Tmax: float = np.inf,
                 unit: str = '-',
                 symbol: str = 'Y',
                 name: str = ''
                 ) -> None:

        super().__init__(A, B, C, D, E, Tmin, Tmax, unit, symbol, name)

    @staticmethod
    def equation(T: Union[float, FloatArray],
                 A: float,
                 B: float,
                 C: float,
                 D: float,
                 E: float
                 ) -> Union[float, FloatArray]:
        r"""DIPPR-104 equation."""
        return A + B/T + C/T**3 + D/T**8 + E/T**9


class DIPPR105(DIPPRP4):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-105 equation.

    This equation implements the following temperature dependence:

    $$ Y = \frac{A}{B^{ \left( 1 + (1 - T / C)^D \right) }} $$

    where $A$ to $D$ are component-specific constants and $T$ is the absolute
    temperature.

    Parameters
    ----------
    A : float
        Parameter of equation.
    B : float
        Parameter of equation.
    C : float
        Parameter of equation.
    D : float
        Parameter of equation.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of output variable $Y$.
    symbol : str
        Symbol of output variable $Y$.
    name : str
        Name.
    """

    _pinfo = {'A': ('#', True), 'B': ('', True), 'C': ('K', True),
              'D': ('', True)}

    def __init__(self,
                 A: float,
                 B: float,
                 C: float,
                 D: float,
                 Tmin: float = 0.0,
                 Tmax: float = np.inf,
                 unit: str = '-',
                 symbol: str = 'Y',
                 name: str = ''
                 ) -> None:

        super().__init__(A, B, C, D, Tmin, Tmax, unit, symbol, name)

    @staticmethod
    def equation(T: Union[float, FloatArray],
                 A: float,
                 B: float,
                 C: float,
                 D: float
                 ) -> Union[float, FloatArray]:
        r"""DIPPR-105 equation."""
        return A / B**(1 + (1 - T / C)**D)


class DIPPR106(DIPPR):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-106 equation.

    This equation implements the following temperature dependence:

    $$ Y = A (1 - T_r)^{B + C T_r + D T_r^2 + E T_r^3} $$

    where $A$ to $E$ are component-specific constants, $T$ is the absolute
    temperature, $T_c$ is the critical temperature and $T_r = T/T_c$ is the
    reduced temperature.

    Parameters
    ----------
    Tc : float
        Critical temperature.
        Unit = K.
    A : float
        Parameter of equation.
    B : float
        Parameter of equation.
    C : float
        Parameter of equation.
    D : float
        Parameter of equation.
    E : float
        Parameter of equation.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of output variable $Y$.
    symbol : str
        Symbol of output variable $Y$.
    name : str
        Name.
    """

    _pinfo = {'A': ('#', True), 'B': ('', True), 'C': ('', True),
              'D': ('', True), 'E': ('', True), 'Tc': ('K', False)}

    def __init__(self,
                 Tc: float,
                 A: float,
                 B: float,
                 C: float = 0.,
                 D: float = 0.,
                 E: float = 0.,
                 Tmin: float = 0.0,
                 Tmax: float = np.inf,
                 unit: str = '-',
                 symbol: str = 'Y',
                 name: str = ''
                 ) -> None:

        self.p = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'Tc': Tc}
        super().__init__((Tmin, Tmax), unit, symbol, name)

    @staticmethod
    def equation(T: Union[float, FloatArray],
                 A: float,
                 B: float,
                 C: float,
                 D: float,
                 E: float,
                 Tc: float,
                 ) -> Union[float, FloatArray]:
        r"""DIPPR-106 equation."""
        Tr = T/Tc
        return A*(1-Tr)**(B + Tr*(C + Tr*(D + E*Tr)))


class DIPPR107(DIPPRP5):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-107 equation.

    This equation implements the following temperature dependence:

    $$ Y = A + B\left[{\frac {C/T}{\sinh \left(C/T\right)}}\right]^2 +
        D\left[{\frac {E/T}{\cosh \left(E/T\right)}}\right]^2 $$

    where $A$ to $E$ are component-specific constants and $T$ is the absolute
    temperature.

    Parameters
    ----------
    A : float
        Parameter of equation.
    B : float
        Parameter of equation.
    C : float
        Parameter of equation.
    D : float
        Parameter of equation.
    E : float
        Parameter of equation.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of output variable $Y$.
    symbol : str
        Symbol of output variable $Y$.
    name : str
        Name.
    """

    _pinfo = {'A': ('#', True), 'B': ('#', True), 'C': ('K', True),
              'D': ('#', True), 'E': ('K', True)}

    def __init__(self,
                 A: float,
                 B: float,
                 C: float,
                 D: float,
                 E: float,
                 Tmin: float = 0.0,
                 Tmax: float = np.inf,
                 unit: str = '-',
                 symbol: str = 'Y',
                 name: str = ''
                 ) -> None:

        super().__init__(A, B, C, D, E, Tmin, Tmax, unit, symbol, name)

    @staticmethod
    def equation(T: Union[float, FloatArray],
                 A: float,
                 B: float,
                 C: float,
                 D: float,
                 E: float
                 ) -> Union[float, FloatArray]:
        r"""DIPPR-107 equation."""
        return A + B*(C/T/sinh(C/T))**2 + D*(E/T/cosh(E/T))**2
