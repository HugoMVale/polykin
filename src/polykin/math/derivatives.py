# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from typing import Callable, Optional

import numpy as np
from numpy import cbrt

from polykin.utils.math import eps
from polykin.utils.types import Float2x2Matrix

__all__ = ['derivative_complex',
           'derivative_centered',
           'hessian2']


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
                        x: float,
                        h: Optional[float] = None
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
    h : float | None
        Finite-difference step. If `None`, it will be set to the theoretical
        optimum value $h = \sqrt[3]{3\epsilon}$.

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
    (3.0000000003141882, 1.0000000009152836)
    """

    if h is None:
        h = cbrt(3*eps)  # ~ 1e-5
        h *= (1 + abs(x))

    fp = f(x + h)
    fm = f(x - h)
    df = (fp - fm)/(2*h)
    fx = (fp + fm)/2

    return (df, fx)


def hessian2(f: Callable[[tuple[float, float]], float],
             x: tuple[float, float],
             h: Optional[float] = None
             ) -> Float2x2Matrix:
    r"""Calculate the numerical Hessian of a scalar function $f(x,y)$ using the
    centered finite-difference scheme.

    $$
    H(x,y)=\begin{bmatrix}
    \frac{\partial^2f}{\partial x^2} & \frac{\partial^2f}{\partial x \partial y} \\ 
    \frac{\partial^2f}{\partial y \partial x} & \frac{\partial^2f}{\partial y^2} 
    \end{bmatrix}
    $$

    where the partial derivatives are computed using the centered
    finite-difference schemes:

    \begin{aligned}
    \frac{\partial^2f(x,y)}{\partial x^2} &= 
            \frac{f(x+2h,y)-f(x,y)+f(x-2h,y)}{4 h^2} + O(h^2) \\
    \frac{\partial^2f(x,y)}{\partial x \partial y} &=
            \frac{f(x+h,y+h)-f(x+h,y-h)-f(x-h,y+h)+f(x-h,y-h)}{4 h^2} + O(h^2)
    \end{aligned}

    Although the matrix only contains 4 elements and is symmetric, a total of
    9 function evaluations are performed.

    **References**

    *   J. Martins and A. Ning. Engineering Design Optimization. Cambridge
    University Press, 2021.

    Parameters
    ----------
    f : Callable[[tuple[float, float]], float]
        Function to be diferentiated.
    x : tuple[float, float]
        Differentiation point.
    h : float | None
        Finite-difference step. If `None`, it will be set to the theoretical
        optimum value $h = \sqrt[3]{3\epsilon}$.

    Returns
    -------
    Float2x2Matrix
        Hessian matrix.

    Examples
    --------
    Evaluate the numerical Hessian of f(x,y)=(x**2)*(y**3) at (2., -2.).
    >>> from polykin.math import hessian2
    >>> def fnc(x): return x[0]**2 * x[1]**3
    >>> hessian2(fnc, (2., -2.))
    array([[-15.99999951,  47.99999983],
           [ 47.99999983, -47.99999983]])
    """

    x0, x1 = x

    if h is None:
        h = cbrt(3*eps)  # ~ 1e-5

    h0 = h*(1. + abs(x0))
    h1 = h*(1. + abs(x1))

    H = np.empty((2, 2))
    f0 = f(x)
    H[0, 0] = (f((x0 + 2*h0, x1)) - 2*f0 + f((x0 - 2*h0, x1)))/(4*h0**2)
    H[1, 1] = (f((x0, x1 + 2*h1)) - 2*f0 + f((x0, x1 - 2*h1)))/(4*h1**2)
    H[0, 1] = (f((x0 + h0, x1 + h1)) - f((x0 + h0, x1 - h1)) - f((x0 - h0, x1 + h1))
               + f((x0 - h0, x1 - h1)))/(4*h0*h1)
    H[1, 0] = H[0, 1]

    return H
