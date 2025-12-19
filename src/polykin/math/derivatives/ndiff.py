# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from collections.abc import Callable

import numpy as np
from numpy import cbrt

from polykin.utils.math import eps
from polykin.utils.types import (
    Float2x2Matrix,
    FloatArray,
    FloatMatrix,
    FloatSquareMatrix,
    FloatVector,
)

__all__ = [
    "derivative_complex",
    "derivative_centered",
    "jacobian_forward",
    "hessian_forward",
    "hessian2_centered",
    "scalex",
]


def derivative_complex(
    f: Callable[[complex], complex],
    x: float,
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
    >>> derivative_complex(f, 1.0)
    (3.0, 1.0)
    """
    fz = f(complex(x, eps))
    return (fz.imag / eps, fz.real)  # type: ignore


def derivative_centered(
    f: Callable[[float], float],
    x: float,
    *,
    h: float = 0.0,
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
    >>> derivative_centered(f, 1.0)
    (np.float64(3.0000000003141882), np.float64(1.0000000009152836))
    """
    if h == 0:
        h = cbrt(3 * eps)  # ~ 1e-5

    h *= 1 + abs(x)

    fp = f(x + h)
    fm = f(x - h)
    df = (fp - fm) / (2 * h)
    fx = (fp + fm) / 2

    return (df, fx)


def jacobian_forward(
    f: Callable[[FloatVector], FloatVector],
    x: FloatVector,
    *,
    fx: FloatVector | None = None,
    sclx: FloatVector | None = None,
    ndigit: int | None = None,
) -> FloatMatrix:
    r"""Calculate the numerical Jacobian of a vector function
    $\mathbf{f}(\mathbf{x})$ using the forward finite-difference scheme.

    $$ J_{ij} = \frac{\partial f_i}{\partial x_j}
              = \frac{f_i(\mathbf{x} + \mathbf{e}_j h_j) - f_i(\mathbf{x})}{h_j} $$

    The step size $h_j$ is optimally determined according to the number of
    reliable digits of $\mathbf{f}$ and the magnitude and scale of each
    $\mathbf{x}$ component.

    Typically, the first `ndigit/2` digits of the Jacobian are accurate.

    **References**

    * J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
      Optimization and Nonlinear Equations", SIAM, 1996, p. 314.

    Parameters
    ----------
    f : Callable[[FloatVector], FloatVector]
        Function to be diferentiated.
    x : FloatVector
        Differentiation point.
    fx : FloatVector | None
        Function values at `x`, if available.
    sclx : FloatVector | None
        Scaling factors for `x`. Ideally, `x[i]*sclx[i]` is close to 1. By
        default, the factors are set internally based on the magnitudes of `x`.
    ndigit : int | None
        Number of reliable base-10 digits in the values returned by `f`. This
        parameter is optional when the function is evaluated analytically,
        but is essential when the function involves numerical procedures (such
        as root finding or ODE integration). If `None`, machine precision is
        assumed.

    Returns
    -------
    FloatMatrix
        Jacobian matrix.

    Examples
    --------
    Evaluate the numerical jacobian of f(x) = x1**2 * x2**3 at (2, -2).
    >>> from polykin.math import jacobian_forward
    >>> import numpy as np
    >>> def f(x): return x[0]**2 * x[1]**3
    >>> jacobian_forward(f, np.array([2.0, -2.0]))
    array([[-32.00000024,  47.99999928]])
    """
    fx = fx if fx is not None else f(x)
    sclx = sclx if sclx is not None else scalex(x)

    η = eps if ndigit is None else 10 ** (-ndigit)
    h0 = np.sqrt(η)

    J = np.empty((fx.size, x.size))
    xh = x.copy()

    for i in range(x.size):
        h = h0 * max(abs(x[i]), abs(1 / sclx[i]))
        xtemp = xh[i]
        xh[i] += h
        h = xh[i] - xtemp
        J[:, i] = (f(xh) - fx) / h
        xh[i] = xtemp

    return J


def hessian_forward(
    f: Callable[[FloatVector], float],
    x: FloatVector,
    *,
    fx: float | None = None,
    sclx: FloatVector | None = None,
    ndigit: int | None = None,
) -> FloatSquareMatrix:
    r"""Calculate the numerical Hessian of a scalar function $f(\mathbf{x})$
    using the forward finite-difference scheme.

    $$ H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
        = \frac{
          f(\mathbf{x} + \mathbf{e}_i h_i + \mathbf{e}_j h_j)
        - f(\mathbf{x} + \mathbf{e}_i h_i) - f(\mathbf{x} + \mathbf{e}_j h_j)
        + f(\mathbf{x})}{h_i h_j} $$

    The step size $h_j$ is optimally determined according to the number of
    reliable digits of $f$ and the magnitude and scale of each $\mathbf{x}$
    component.

    Typically, the first `ndigit/3` digits of the Hessian are accurate.

    **References**

    * J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
      Optimization and Nonlinear Equations", SIAM, 1996, p. 321.

    Parameters
    ----------
    f : Callable[[FloatVector], float]
        Function to be diferentiated.
    x : FloatVector
        Differentiation point.
    fx : float | None
        Function values at `x`, if available.
    sclx : FloatVector | None
        Scaling factors for `x`. Ideally, `x[i]*sclx[i]` is close to 1. By
        default, the factors are set internally based on the magnitudes of `x`.
    ndigit : int | None
        Number of reliable base-10 digits in the values returned by `f`. This
        parameter is optional when the function is evaluated analytically,
        but is essential when the function involves numerical procedures (such
        as root finding or ODE integration). If `None`, machine precision is
        assumed.

    Returns
    -------
    FloatSquareMatrix
        Hessian matrix.

    Examples
    --------
    Evaluate the numerical hessian of f(x) = x1**2 * x2**3 at (2, -2).
    >>> from polykin.math import hessian_forward
    >>> import numpy as np
    >>> def f(x): return x[0]**2 * x[1]**3
    >>> hessian_forward(f, np.array([2.0, -2.0]))
    array([[-16.0001093 ,  47.99984347],
           [ 47.99984347, -47.99979503]])
    """
    fx = fx if fx is not None else f(x)
    sclx = sclx if sclx is not None else scalex(x)

    η = eps if ndigit is None else 10 ** (-ndigit)
    h0 = np.cbrt(η)

    N = x.size
    H = np.empty((N, N))
    fh = np.empty(N)
    h = np.empty(N)
    xh = x.copy()

    for i in range(N):
        h[i] = h0 * max(abs(x[i]), abs(1 / sclx[i]))
        xtemp1 = xh[i]
        xh[i] += h[i]
        h[i] = xh[i] - xtemp1
        fh[i] = f(xh)
        xh[i] = xtemp1

    for i in range(N):
        xtemp1 = xh[i]
        xh[i] += 2 * h[i]
        H[i, i] = ((fx - fh[i]) + (f(xh) - fh[i])) / (h[i] ** 2)
        xh[i] = xtemp1 + h[i]
        for j in range(i + 1, N):
            xtemp2 = xh[j]
            xh[j] += h[j]
            H[i, j] = ((fx - fh[i]) + (f(xh) - fh[j])) / (h[i] * h[j])
            H[j, i] = H[i, j]
            xh[j] = xtemp2
        xh[i] = xtemp1

    return H


def hessian2_centered(
    f: Callable[[tuple[float, float]], float],
    x: tuple[float, float],
    h: float = 0.0,
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
    Evaluate the numerical Hessian of f(x,y)=(x**2)*(y**3) at (2, -2).
    >>> from polykin.math import hessian2_centered
    >>> def f(x): return x[0]**2 * x[1]**3
    >>> hessian2_centered(f, (2.0, -2.0))
    array([[-15.99999951,  47.99999983],
           [ 47.99999983, -47.99999983]])
    """
    x0, x1 = x

    if h == 0:
        h = cbrt(3 * eps)  # ~ 1e-5

    h0 = h * (1 + abs(x0))
    h1 = h * (1 + abs(x1))

    H = np.empty((2, 2))
    f0 = f(x)
    H[0, 0] = (f((x0 + 2 * h0, x1)) - 2 * f0 + f((x0 - 2 * h0, x1))) / (4 * h0**2)
    H[1, 1] = (f((x0, x1 + 2 * h1)) - 2 * f0 + f((x0, x1 - 2 * h1))) / (4 * h1**2)
    H[0, 1] = (
        f((x0 + h0, x1 + h1))
        - f((x0 + h0, x1 - h1))
        - f((x0 - h0, x1 + h1))
        + f((x0 - h0, x1 - h1))
    ) / (4 * h0 * h1)
    H[1, 0] = H[0, 1]

    return H


def scalex(x: FloatArray) -> FloatArray:
    r"""Calculate scaling factors for a given array.

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

    Examples
    --------
    Scale the following array.
    >>> from polykin.math import scalex
    >>> import numpy as np
    >>> scalex(np.array([1e-2, 0.0, 1.0, 1e3]))
    array([1.e+02, 1.e+03, 1.e+00, 1.e-03])
    """
    sclx = np.ones_like(x)

    iszero = x == 0.0

    if len(x[~iszero]) == 0:
        return sclx

    xmax = np.max(np.abs(x))
    xmin = np.min(np.abs(x[~iszero]))

    sclx[iszero] = 1e1 / xmin

    if np.log10(xmax / xmin) >= 1.0:
        sclx[~iszero] = 1 / np.abs(x[~iszero])
    else:
        sclx[~iszero] = 1 / xmax

    return sclx
