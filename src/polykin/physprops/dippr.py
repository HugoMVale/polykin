# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.utils import FloatOrArray
from polykin.physprops.property_equation import PropertyEquationT

import numpy as np

__all__ = ['DIPPR100',
           'DIPPR101',
           'DIPPR102',
           'DIPPR104',
           'DIPPR105',
           'DIPPR106']


class DIPPR(PropertyEquationT):
    """_Abstract_ class for all
    [DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)
    temperature-dependent equations."""
    pass


class DIPPRP4(DIPPR):
    """_Abstract_ class for DIPPR equations with 4 parameters (A-D)."""

    A: float
    B: float
    C: float
    D: float

    _params = (('A', 'B', 'C', 'D'), ())

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

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        super().__init__((Tmin, Tmax), unit, symbol, name)


class DIPPRP5(DIPPR):
    """_Abstract_ class for DIPPR equations with 5 parameters (A-E)."""

    A: float
    B: float
    C: float
    D: float
    E: float

    _params = (('A', 'B', 'C', 'D', 'E'), ())

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

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
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

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate correlation at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Property value, $Y$.
        """
        return self.A + self.B*T + self.C*T**2 + self.D*T**3 + self.E*T**4


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

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate correlation at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Property value, $Y$.
        """
        return np.exp(self.A + self.B/T + self.C*np.log(T) + self.D*T**self.E)


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

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate correlation at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Property value, $Y$.
        """
        return (self.A * T**self.B) / (1 + self.C/T + self.D/T**2)


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

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate correlation at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Property value, $Y$.
        """
        return self.A + self.B/T + self.C/T**3 + self.D/T**8 + self.E/T**9


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

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate correlation at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Property value, $Y$.
        """
        return self.A / self.B**(1 + (1 - T / self.C)**self.D)


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

    Tc: float
    A: float
    B: float
    C: float
    D: float
    E: float

    _params = (('A', 'B', 'C', 'D', 'E'), ('Tc',))

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

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.Tc = Tc
        super().__init__((Tmin, Tmax), unit, symbol, name)

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate correlation at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Property value, $Y$.
        """
        Tr = T/self.Tc
        return self.A*(1-Tr)**(self.B + Tr*(self.C + Tr*(self.D + self.E*Tr)))
