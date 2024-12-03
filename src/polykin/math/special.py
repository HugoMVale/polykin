# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import functools

import numpy as np
from numpy import arctan, exp, inf, pi, sqrt, tan
from scipy.special import erfc

from polykin.math import fzero_newton
from polykin.utils.types import FloatVector

__all__ = ['ierfc',
           'i2erfc',
           'roots_xtanx',
           'roots_xcotx'
           ]


def ierfc(x: float) -> float:
    r"""Integral of the complementary error function.

    $$ \mathrm{ierfc}(x) = \int_x^{\infty}\mathrm{erfc}(\xi)d\xi $$

    Parameters
    ----------
    x : float
        Argument.

    Returns
    -------
    float
        Value of the integral.

    Examples
    --------
    >>> ierfc(0.5)
    0.1996412283742457
    """
    if x > 30.:
        return 0.
    else:
        return exp(-x**2)/sqrt(pi) - x*erfc(x)


def i2erfc(x: float) -> float:
    r"""Integral of the integral of the complementary error function.

    $$ \mathrm{i^2erfc}(x) = \int_x^{\infty}\mathrm{ierfc}(\xi)d\xi $$

    Parameters
    ----------
    x : float
        Argument.

    Returns
    -------
    float
        Value of the integral.

    Examples
    --------
    >>> i2erfc(0.5)
    0.06996472345317695
    """
    if x > 30.:
        return 0.
    else:
        return (erfc(x) - 2*x*ierfc(x))/4


@functools.cache
def roots_xtanx(a: float, N: int, xtol: float = 1e-6) -> FloatVector:
    r"""Determine the first `N` roots of the transcendental equation 
    `x_n*tan(x_n) = a`.

    Parameters
    ----------
    a : float
        Parameter.
    N : int
        Number of roots.
    xtol : float
        Tolerance.

    Returns
    -------
    FloatVector
        Roots of the equation.

    Examples
    --------
    Determine the first 4 roots for a=1.
    >>> from polykin.math import roots_xtanx
    >>> roots_xtanx(1.0, 4)
    array([0.86033359, 3.42561846, 6.43729818, 9.52933441])
    """
    result = np.zeros(N)

    if a == 0:
        for i in range(N):
            result[i] = i*pi
    elif a == inf:
        for i in range(N):
            result[i] = (i + 1/2)*pi
    elif a < 1e2:
        for i in range(N):
            sol = fzero_newton(f=lambda x: x*tan(x) - a,
                               x0=pi*(i + 0.5*a/(a+1)),
                               xtol=xtol,
                               ftol=1e-10)
            result[i] = sol.x
    else:
        for i in range(N):
            x0 = i*pi
            e = 1.0
            for _ in range(0, 10):
                e_new = arctan(a/(x0 + e))
                if abs(e_new - e) < xtol:
                    break
                e = e_new
            result[i] = x0 + e

    return result


@functools.cache
def roots_xcotx(a: float, N: int, xtol: float = 1e-6) -> FloatVector:
    r"""Determine the first `N` roots of the transcendental equation 
    `1 - x_n*cot(x_n) = a`.

    Parameters
    ----------
    a : float
        Parameter.
    N : int
        Number of roots.
    xtol : float
        Tolerance.

    Returns
    -------
    FloatVector
        Roots of the equation.

    Examples
    --------
    Determine the first 4 roots for a=1.
    >>> from polykin.math import roots_xcotx
    >>> roots_xcotx(2.0, 4)
    array([ 2.0287575 ,  4.91318033,  7.97866578, 11.08553772])
    """
    result = np.zeros(N)

    if a == 1:
        for i in range(N):
            result[i] = (i + 1/2)*pi
    elif a == inf:
        for i in range(N):
            result[i] = (i + 1)*pi
    elif a < 2:
        for i in range(N):
            sol = fzero_newton(f=lambda x: x/tan(x) + a - 1,
                               x0=(i + 1/2)*pi,
                               xtol=xtol,
                               ftol=1e-10)
            result[i] = sol.x
    else:
        for i in range(N):
            x0 = (i+1)*pi
            e = 0.0
            for _ in range(0, 30):
                e_new = np.arctan((x0 + e)/(1 - a))
                if abs(e_new - e) < xtol:
                    break
                e = e_new
            result[i] = x0 + e

    return result
