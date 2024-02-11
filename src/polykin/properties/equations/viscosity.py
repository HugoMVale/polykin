# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

# This module implements commonly used equations to evaluate the viscosity of
# pure components.

from typing import Union

import numpy as np
from numpy import exp

from polykin.utils.types import FloatArray

from .base import PropertyEquationT

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
    base10 : bool
        If `True` base of logarithm is `10`, otherwise it is $e$.
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
              'D': ('K⁻²', True), 'base10': ('', False)}

    def __init__(self,
                 A: float,
                 B: float,
                 C: float = 0.,
                 D: float = 0.,
                 base10: bool = True,
                 Tmin: float = 0.,
                 Tmax: float = np.inf,
                 unit: str = 'Pa·s',
                 symbol: str = r'\mu',
                 name: str = ''
                 ) -> None:
        """Construct `Yaws` with the given parameters."""

        self.p = {'A': A, 'B': B, 'C': C, 'D': D, 'base10': base10}
        super().__init__((Tmin, Tmax), unit, symbol, name)

    @staticmethod
    def equation(T: Union[float, FloatArray],
                 A: float,
                 B: float,
                 C: float,
                 D: float,
                 base10: bool
                 ) -> Union[float, FloatArray]:
        r"""Yaws equation.

        Parameters
        ----------
        T : float | FloatArray
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
        base10 : bool
            If `True` base of logarithm is `10`, otherwise it is $e$.

        Returns
        -------
        float | FloatArray
            Viscosity. Unit = Any.
        """
        x = A + B/T + C*T + D*T**2
        if base10:
            return 10**x
        else:
            return exp(x)
