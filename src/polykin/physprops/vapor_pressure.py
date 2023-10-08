# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.utils import FloatOrArray
from polykin.physprops.property_equation import PropertyEquationT

import numpy as np

__all__ = ['Antoine', 'antoine']


class Antoine(PropertyEquationT):
    r"""[Antoine](https://en.wikipedia.org/wiki/Antoine_equation) equation.

    This equation implements the following temperature dependence:

    $$ \log_{base} P^* = A - \frac{B}{T + C} $$

    where $A$, $B$ and $C$ are constant parameters, $T$ is the temperature,
    and $P^*$ is the vapor pressure.

    !!! note
        There is no consensus on the value of $base$, the unit of temperature,
        or the unit of pressure. The function is flexible enough to accomodate
        most cases, but care should be taken to ensure the parameters match the
        intended use.

    !!! hint
        The Antoine equation is limited in terms of temperature range. Wider
        ranges can be achieved with
        [DIPPR101](DIPPR.md).

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

    A: float
    B: float
    C: float
    base: float

    def __init__(self,
                 A: float,
                 B: float,
                 C: float,
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
        self.Trange = (Tmin, Tmax)
        self.unit = unit
        self.symbol = symbol
        self.name = name

    def __repr__(self) -> str:
        return (
            f"name:        {self.name}\n"
            f"symbol:      {self.symbol}\n"
            f"unit:        {self.unit}\n"
            f"A:           {self.A}\n"
            f"B:           {self.B}\n"
            f"C:           {self.C}\n"
            f"base:        {self.base}\n"
            f"Trange [K]:  {self.Trange}"
        )

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
        return antoine(T, self.A, self.B, self.C, self.base)


def antoine(T: FloatOrArray,
            A: float,
            B: float,
            C: float,
            base=10.
            ) -> FloatOrArray:
    r"""[Antoine](https://en.wikipedia.org/wiki/Antoine_equation) equation.

    This equation implements the following temperature dependence:

    $$ \log_{base} P^* = A - \frac{B}{T + C} $$

    where $A$, $B$ and $C$ are constant parameters, $T$ is the temperature,
    and $P^*$ is the vapor pressure.

    !!! note
        There is no consensus on the value of $base$, the unit of temperature,
        or the unit of pressure. The function is flexible enough to accomodate
        most cases, but care should be taken to ensure the parameters match the
        intended use.

    !!! hint
        The Antoine equation is limited in terms of temperature range. Wider
        ranges can be achieved with
        [DIPPR101](DIPPR.md).

    Parameters
    ----------
    T : FloatOrArray
        Temperature.
        Unit = (K, °C).
    A : float
        Parameter of equation.
    B : float
        Parameter of equation.
        Unit = K.
    C : float
        Parameter of equation.
        Unit = (K, °C).
    base : float
        Base of logarithm, usually equal to $10$ or $e$.

    Returns
    -------
    FloatOrArray
        Vapor pressure, $P^*$.
    """
    return base**(A - B/(T + C))
