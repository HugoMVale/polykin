# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.utils import FloatOrArray
from polykin.physprops.property_equation import PropertyEquationT

import numpy as np

__all__ = ['Antoine', 'Wagner']


class Antoine(PropertyEquationT):
    r"""[Antoine](https://en.wikipedia.org/wiki/Antoine_equation) equation for
    vapor pressure.

    This equation implements the following temperature dependence:

    $$ \log_{base} P^* = A - \frac{B}{T + C} $$

    where $A$, $B$ and $C$ are component-specific constants, $P^*$ is the vapor
    pressure and $T$ is the temperature. When $C=0$, this equation reverts to
    the Clapeyron equation.

    !!! note
        There is no consensus on the value of $base$, the unit of temperature,
        or the unit of pressure. The function is flexible enough to accomodate
        most cases, but care should be taken to ensure the parameters match the
        intended use.

    !!! hint
        The Antoine equation is limited in terms of temperature range. Wider
        ranges can be achieved with [DIPPR101](DIPPR.md) or
        [Wagner](Wagner.md).

    Parameters
    ----------
    A : float
        Parameter of equation.
    B : float
        Parameter of equation.
        Unit = K.
    C : float
        Parameter of equation.
        Unit = K.
    base : float
        Base of logarithm, usually equal to $10$ or $e$.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of vapor pressure.
    symbol : str
        Symbol of vapor pressure.
    name : str
        Name.
    """

    _params = (('A', 'B', 'C'), ('base',))

    def __init__(self,
                 A: float,
                 B: float,
                 C: float = 0.,
                 base: float = 10.,
                 Tmin: float = 0.0,
                 Tmax: float = np.inf,
                 unit: str = 'Pa',
                 symbol: str = 'P^*',
                 name: str = ''
                 ) -> None:
        """Construct `Antoine` with the given parameters."""

        self.A = A
        self.B = B
        self.C = C
        self.base = base
        super().__init__((Tmin, Tmax), unit, symbol, name)

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate property equation at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Vapor pressure, $P^*$.
        """
        A = self.A
        B = self.B
        C = self.C
        base = self.base
        return base**(A - B/(T + C))


# %% Wagner


class Wagner(PropertyEquationT):

    r"""[Wagner](https://de.wikipedia.org/wiki/Wagner-Gleichung) equation for
    vapor pressure.

    This equation implements the following temperature dependence:

    $$ \ln(P^*/P_c) = \frac{a\tau + b\tau^{1.5} + c\tau^{2.5} + d\tau^5}{T_r}$$

    with:

    $$ \tau = 1 - T_r$$

    where $a$ to $d$ are component-specific constants, $P^*$ is the vapor
    pressure, $P_c$ is the critical pressure, $T$ is the absolute temperature,
    $T_c$ is the critical temperature, and $T_r=T/T_c$ is the reduced
    temperature.

    !!! note

        There are several versions of the Wagner equation with different
        exponents. This is the so-called 25 version also used in the
        [ThermoData Engine](https://trc.nist.gov/tde.html).

    Parameters
    ----------
    Tc : float
        Critical temperature.
        Unit = K.
    Pc : float
        Critical pressure.
        Unit = Any.
    a : float
        Parameter of equation.
    b : float
        Parameter of equation.
    c : float
        Parameter of equation.
    d : float
        Parameter of equation.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of vapor pressure.
    symbol : str
        Symbol of vapor pressure.
    name : str
        Name.
    """

    _params = (('a', 'b', 'c', 'd', 'Pc'), ('Tc',))

    def __init__(self,
                 Pc: float,
                 Tc: float,
                 a: float,
                 b: float,
                 c: float,
                 d: float,
                 Tmin: float = 0.0,
                 Tmax: float = np.inf,
                 unit: str = 'Pa',
                 symbol: str = 'P^*',
                 name: str = ''
                 ) -> None:
        """Construct `Wagner` with the given parameters."""

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.Pc = Pc
        self.Tc = Tc
        super().__init__((Tmin, Tmax), unit, symbol, name)

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate property equation at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Vapor pressure, $P^*$.
        """
        Tc = self.Tc
        Pc = self.Pc
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        Tr = T/Tc
        t = 1 - Tr
        return Pc*np.exp((a*t + b*t**1.5 + c*t**2.5 + d*t**5)/Tr)
