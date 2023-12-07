# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

# This module implements commonly used equations to evaluate the viscosity of
# pure components.

from polykin.types import FloatOrArray
from .base import PropertyEquationT

import numpy as np

__all__ = ['Yaws']


class Yaws(PropertyEquationT):
    r"""Yaws equation for saturated liquid viscosity.

    This equation implements the following temperature dependence:

    $$ \log_{base} \mu = A + \frac{B}{T} + C T + D T^2 $$

    where $A$ to $D$ are component-specific constants, $\mu$ is the liquid
    viscosity, and $T$ is the temperature. When $C=D=0$, this equation reverts
    to the Andrade equation.

    Parameters
    ----------
    A : float
        Parameter of equation.
    B : float
        Parameter of equation.
        Unit = K.
    C : float
        Parameter of equation.
        Unit = K⁻¹.
    D : float
        Parameter of equation.
        Unit = K⁻².
    base : float
        Base of logarithm, usually equal to $10$ or $e$.
    Tmin : float
        Lower temperature bound.
        Unit = K.
    Tmax : float
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of viscosity.
    symbol : str
        Symbol of viscosity.
    name : str
        Name.
    """

    _pnames = (('A', 'B', 'C', 'D'), ('base',))
    _punits = ('', 'K', 'K⁻¹', 'K⁻²', '')

    def __init__(self,
                 A: float,
                 B: float,
                 C: float = 0.,
                 D: float = 0.,
                 base: float = 10.,
                 Tmin: float = 0.,
                 Tmax: float = np.inf,
                 unit: str = 'Pa·s',
                 symbol: str = r'\mu',
                 name: str = ''
                 ) -> None:
        """Construct `Yaws` with the given parameters."""

        self.pvalues = (A, B, C, D, base)
        super().__init__((Tmin, Tmax), unit, symbol, name)

    @staticmethod
    def equation(T: FloatOrArray,
                 A: float,
                 B: float,
                 C: float,
                 D: float,
                 base: float
                 ) -> FloatOrArray:
        r"""Yaws equation.

        Parameters
        ----------
        T : FloatOrArray
            Temperature. Unit = K.
        A : float
            Parameter of equation.
        B : float
            Parameter of equation.
            Unit = K.
        C : float
            Parameter of equation.
            Unit = K⁻¹.
        D : float
            Parameter of equation.
            Unit = K⁻².
        base : float
            Base of logarithm, usually equal to $10$ or $e$.

        Returns
        -------
        FloatOrArray
            Viscosity. Unit = Any.
        """
        return base**(A + B/T + C*T + D*T**2)
