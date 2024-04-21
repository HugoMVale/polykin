# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from typing import Callable

from numpy import cbrt

from polykin.utils.math import eps

__all__ = ['derivative_complex',
           'derivative_centered']


def derivative_complex(f: Callable[[complex], complex],
                       x: float
                       ) -> tuple[float, float]:
    r"""Calculate the numerical derivative of a scalar function using the
    complex differentiation method.

    $$ f'(x) = \frac{\text{Im}\left(f(x + i h) \right)}{h} + O(h^2) $$

    !!! note

        This method is efficient, very accurate, and not ill-conditioned.
        However, its application is restricted to real functions that can be
        evaluated with complex inputs, but which per se do not implement
        complex arithmetic.

    **References**

    *   J. Martins and A. Ning. Engineering Design Optimization. Cambridge
    University Press, 2021.
    *   [boost/math/differentiation/finite_difference.hpp](https://www.boost.org/doc/libs/1_80_0/boost/math/differentiation/finite_difference.hpp).

    Parameters
    ----------
    f : Callable[[complex], complex]
        Function to be diferentiated.
    x : float
        Diferentiation point.

    Returns
    -------
    tuple[float, float]
        Tuple with derivative and function value, $(f'(x), f(x))$.

    Examples
    --------
    Evaluate the numerical derivative of f(x)=x**3 at x=1.
    >>> from polykin.math import derivative_complex
    >>> def f(x): return x**3
    >>> derivative_complex(f, 1.)
    (3.0, 1.0)
    """
    fz = f(complex(x, eps))
    return (fz.imag/eps, fz.real)


def derivative_centered(f: Callable[[float], float],
                        x: float
                        ) -> tuple[float, float]:
    r"""Calculate the numerical derivative of a scalar function using the
    centered finite-difference scheme.

    $$ f'(x) = \frac{f(x + h) - f(x - h)}{2 h} + O(h^2) $$

    **References**

    *   J. Martins and A. Ning. Engineering Design Optimization. Cambridge
    University Press, 2021.
    *   [boost/math/differentiation/finite_difference.hpp](https://www.boost.org/doc/libs/1_80_0/boost/math/differentiation/finite_difference.hpp).

    Parameters
    ----------
    f : Callable[[float], float]
        Function to be diferentiated.
    x : float
        Differentiation point.

    Returns
    -------
    tuple[float, float]
        Tuple with derivative and mean function value, $(f'(x), f(x))$.

    Examples
    --------
    Evaluate the numerical derivative of f(x)=x**3 at x=1.
    >>> from polykin.math import derivative_centered
    >>> def f(x): return x**3
    >>> derivative_centered(f, 1.)
    (3.000000000069477, 1.0000000002288205)
    """
    h = cbrt(3*eps)
    fp = f(x + h)
    fm = f(x - h)
    df = (fp - fm)/(2*h)
    fx = (fp + fm)/2
    return (df, fx)
