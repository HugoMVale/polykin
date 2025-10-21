# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from typing import Callable

import numpy as np
from numpy import cbrt

from polykin.utils.math import eps
from polykin.utils.types import (Float2x2Matrix, FloatArray, FloatMatrix,
                                 FloatVector)

__all__ = [
    'derivative_complex',
    'derivative_centered',
    'hessian2',
    'jacobian',
    'scalex'
]


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
                        h: float = 0.
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
    h : float
        Finite-difference step. If `0`, it will be set to the theoretical
        optimum value $h=\sqrt[3]{3\epsilon}$.

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
    (np.float64(3.0000000003141882), np.float64(1.0000000009152836))
    """

    if h == 0:
        h = cbrt(3*eps)    # ~ 1e-5

    h *= (1 + abs(x))

    fp = f(x + h)
    fm = f(x - h)
    df = (fp - fm)/(2*h)
    fx = (fp + fm)/2

    return (df, fx)


def hessian2(f: Callable[[tuple[float, float]], float],
             x: tuple[float, float],
             h: float = 0.
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
    h : float
        Finite-difference step. If `0`, it will be set to the theoretical
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

    if h == 0:
        h = cbrt(3*eps)  # ~ 1e-5

    h0 = h*(1 + abs(x0))
    h1 = h*(1 + abs(x1))

    H = np.empty((2, 2))
    f0 = f(x)
    H[0, 0] = (f((x0 + 2*h0, x1)) - 2*f0 + f((x0 - 2*h0, x1)))/(4*h0**2)
    H[1, 1] = (f((x0, x1 + 2*h1)) - 2*f0 + f((x0, x1 - 2*h1)))/(4*h1**2)
    H[0, 1] = (f((x0 + h0, x1 + h1)) - f((x0 + h0, x1 - h1)) - f((x0 - h0, x1 + h1))
               + f((x0 - h0, x1 - h1)))/(4*h0*h1)
    H[1, 0] = H[0, 1]

    return H


def jacobian(
    f: Callable[[FloatVector], FloatVector],
    x: FloatVector,
    fx: FloatVector | None = None,
    sx: FloatVector | None = None
) -> FloatMatrix:
    r"""Calculate the numerical Jacobian of a vector function 
    $\mathbf{f}(\mathbf{x})$ using the forward finite-difference scheme.

    $$
    \mathbf{J} = \begin{pmatrix}
    \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
    \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
    \end{pmatrix}
    $$

    Parameters
    ----------
    f : Callable[[FloatVector], FloatVector]
        Function to be diferentiated.
    x : FloatVector
        Differentiation point.
    fx : FloatVector | None
        Function values at `x`, if available.
    sx : FloatVector | None
        Scaling factors for `x`. Ideally, `x[i]*sx[i]` is close to 1.

    Returns
    -------
    FloatMatrix
        Jacobian matrix.

    Examples
    --------
    Evaluate the numerical jacobian of f(x)=(x1**2)*(x2**3) at (2.0, -2.0).
    >>> from polykin.math import jacobian
    >>> import numpy as np
    >>> def fnc(x): return x[0]**2 * x[1]**3
    >>> jacobian(fnc, np.array([2.0, -2.0]))
    array([[-32.00000024,  47.99999928]])
    """

    fx = fx if fx is not None else f(x)
    sx = sx if sx is not None else scalex(x)

    jac = np.empty((fx.size, x.size))
    h0 = np.sqrt(eps)
    xp = x.copy()

    for i in range(x.size):
        h = h0*max(abs(x[i]), 1/sx[i])
        xtemp = xp[i]
        xp[i] += h
        jac[:, i] = (f(xp) - fx)/h
        xp[i] = xtemp

    return jac


def scalex(x: FloatArray) -> FloatArray:
    r"""Calculate a scaling factors for a given array.

    The scaling factors are computed according to the heuristic procedure 
    implemented in ODRPACK95.

    Parameters
    ----------
    x : FloatArray
        Array to be scaled.

    Returns
    -------
    FloatArray
        Scaling array.
    """

    sx = np.ones_like(x)

    iszero = x == 0.0

    if len(x[~iszero]) == 0:
        return sx

    xmax = np.max(np.abs(x))
    xmin = np.min(np.abs(x[~iszero]))

    sx[iszero] = 1e1/xmin

    if np.log10(xmax/xmin) >= 1.0:
        sx[~iszero] = 1/np.abs(x[~iszero])
    else:
        sx[~iszero] = 1/xmax

    return sx
