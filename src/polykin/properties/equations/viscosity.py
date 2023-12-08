# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

# This module implements commonly used equations to evaluate the viscosity of
# pure components.

from polykin.types import FloatOrArray
from .base import PropertyEquationT

import numpy as np
from typing import Literal

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
        Base of logarithm, either $10$ or $e$.
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

    _pinfo = {'A': ('', True), 'B': ('K', True), 'C': ('K⁻¹', True),
              'D': ('K⁻²', True), 'base': ('', False)}

    def __init__(self,
                 A: float,
                 B: float,
                 C: float = 0.,
                 D: float = 0.,
                 base: Literal['e', '10'] = '10',
                 Tmin: float = 0.,
                 Tmax: float = np.inf,
                 unit: str = 'Pa·s',
                 symbol: str = r'\mu',
                 name: str = ''
                 ) -> None:
        """Construct `Yaws` with the given parameters."""

        self.p = {'A': A, 'B': B, 'C': C, 'D': D, 'base': base}
        super().__init__((Tmin, Tmax), unit, symbol, name)

    @staticmethod
    def equation(T: FloatOrArray,
                 A: float,
                 B: float,
                 C: float,
                 D: float,
                 base: Literal['e', '10']
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
        x = A + B/T + C*T + D*T**2
        if base == '10':
            return 10**x
        elif base == 'e':
            return np.exp(x)
        else:
            raise ValueError("Invalid `base`.")