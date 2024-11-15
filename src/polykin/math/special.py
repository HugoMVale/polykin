# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import exp, pi, sqrt
from scipy.special import erfc

__all__ = ['ierfc',
           'i2erfc'
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
