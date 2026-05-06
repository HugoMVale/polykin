# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import math
from collections.abc import Callable

import numpy as np

from polykin.math.machine import eps
from polykin.utils.typing import (
    Float2x2Matrix,
    FloatArray,
    FloatMatrix,
    FloatSquareMatrix,
    FloatVector,
)

__all__ = [
    "derivative_complex",
    "derivative_centered",
    "gradient_forward",
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

    *   J. Martins and A. Ning. Engineering Design Optimization. Cambridge University
        Press, 2021.
    *   [boost/math/differentiation/finite_difference.hpp](https://www.boost.org/doc/libs/1_80_0/boost/math/differentiation/finite_difference.hpp).

    Parameters
    ----------
    f : Callable[[complex], complex]
        Function to be differentiated.
    x : float
        Differentiation point.

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
    epsf: float | None = None,
    h: float = 0.0,
) -> tuple[float, float]:
    r"""Calculate the numerical derivative of a scalar function using the
    centered finite-difference scheme.

    $$ f'(x) = \frac{f(x + h) - f(x - h)}{2 h} + O(h^2) $$

    **References**

    *   J. Martins and A. Ning. Engineering Design Optimization. Cambridge University
        Press, 2021.
    *   [boost/math/differentiation/finite_difference.hpp](https://www.boost.org/doc/libs/1_80_0/boost/math/differentiation/finite_difference.hpp).

    Parameters
    ----------
    f : Callable[[float], float]
        Function to be differentiated.
    x : float
        Differentiation point.
    epsf : float | None
        Machine precision of the function values. If `None`, machine precision of 64-bit
        floating-point type is assumed. If the number of reliable base-10 digits in the
        results returned by the function is $n$, then `epsf` is approximately $10^{-n}$.
    h : float
        Finite-difference step. If `0`, it will be set to the theoretical optimum
        value $h=\sqrt[3]{3\epsilon_f}$.

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
    (3.0000000003141882, 1.0000000009152836)
    """
    epsf = max(epsf, eps) if epsf is not None else eps
    _h = math.cbrt(3 * epsf)

    h = max(h, _h) if h != 0 else _h
    h *= max(1.0, abs(x))

    xp = x + h
    xm = x - h
    fp = f(xp)
    fm = f(xm)
    df = (fp - fm) / (xp - xm)
    fx = (fp + fm) / 2.0

    return (df, fx)


def gradient_forward(
    f: Callable[[FloatVector], float],
    x: FloatVector,
    *,
    fx: float | None = None,
    sclx: FloatVector | None = None,
    epsf: float | None = None,
) -> FloatVector:
    r"""Calculate the numerical gradient of a scalar function $f(\mathbf{x})$ using the
    forward finite-difference scheme.

    $$ \frac{\partial f}{\partial x_i} =
       \frac{f(\mathbf{x} + \mathbf{e}_i h_i) - f(\mathbf{x})}{h_i} $$

    The step size $h_i$ is optimally determined according to the machine precision of the
    function values. Typically, the gradient is accurate to about half the number of
    reliable digits returned by the function.

    If the function value at $\mathbf{x}$ is provided, $N$ function evaluations are
    required to compute the gradient, where $N$ is the dimension of $\mathbf{x}$.

    **References**

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained Optimization
        and Nonlinear Equations", SIAM, 1996, p. 323.

    Parameters
    ----------
    f : Callable[[FloatVector], float]
        Function to be differentiated.
    x : FloatVector
        Differentiation point.
    fx : float | None
        Function value at `x`, if available.
    sclx : FloatVector | None
        Scaling factors for `x`. Ideally, `x[i]*sclx[i]` is close to 1. By
        default, the factors are set internally based on the magnitudes of `x`.
    epsf : float | None
        Machine precision of the function values. If `None`, machine precision of 64-bit
        floating-point type is assumed. If the number of reliable base-10 digits in the
        results returned by the function is $n$, then `epsf` is approximately $10^{-n}$.

    Returns
    -------
    FloatVector
        Gradient vector.

    Examples
    --------
    Evaluate the numerical gradient of f(x) = x1**2 * x2**3 at (2, -2).
    >>> from polykin.math import gradient_forward
    >>> import numpy as np
    >>> def f(x): return x[0]**2 * x[1]**3
    >>> gradient_forward(f, np.array([2.0, -2.0]))
    array([-32.00000024,  47.99999928])
    """
    fx = fx if fx is not None else f(x)
    sclx = np.abs(sclx) if sclx is not None else scalex(x)

    epsf = max(epsf, eps) if epsf is not None else eps
    h0 = math.sqrt(epsf)

    g = np.empty_like(x, dtype=float)
    xh = x.copy()

    for i in range(x.size):
        h = h0 * max(abs(x[i]), abs(1 / sclx[i]))
        xtemp = xh[i]
        xh[i] += h
        h = xh[i] - xtemp
        g[i] = (f(xh) - fx) / h
        xh[i] = xtemp

    return g


def jacobian_forward(
    f: Callable[[FloatVector], FloatVector],
    x: FloatVector,
    *,
    fx: FloatVector | None = None,
    sclx: FloatVector | None = None,
    epsf: float | None = None,
) -> FloatMatrix:
    r"""Calculate the numerical Jacobian of a vector function
    $\mathbf{f}(\mathbf{x})$ using the forward finite-difference scheme.

    $$ J_{ij} = \frac{\partial f_i}{\partial x_j}
              = \frac{f_i(\mathbf{x} + \mathbf{e}_j h_j) - f_i(\mathbf{x})}{h_j} $$

    The step size $h_j$ is optimally determined according to the machine precision of the
    function values. Typically, the Jacobian is accurate to about half the number of
    reliable digits returned by the function.

    If the function value at $\mathbf{x}$ is provided, $N$ function evaluations are
    required to compute the Jacobian, where $N$ is the dimension of $\mathbf{x}$.

    **References**

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained Optimization
        and Nonlinear Equations", SIAM, 1996, p. 314.

    Parameters
    ----------
    f : Callable[[FloatVector], FloatVector]
        Function to be differentiated.
    x : FloatVector
        Differentiation point.
    fx : FloatVector | None
        Function values at `x`, if available.
    sclx : FloatVector | None
        Scaling factors for `x`. Ideally, `x[i]*sclx[i]` is close to 1. By
        default, the factors are set internally based on the magnitudes of `x`.
    epsf : float | None
        Machine precision of the function values. If `None`, machine precision of 64-bit
        floating-point type is assumed. If the number of reliable base-10 digits in the
        results returned by the function is $n$, then `epsf` is approximately $10^{-n}$.

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
    sclx = np.abs(sclx) if sclx is not None else scalex(x)

    epsf = max(epsf, eps) if epsf is not None else eps
    h0 = math.sqrt(epsf)

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
    epsf: float | None = None,
) -> FloatSquareMatrix:
    r"""Calculate the numerical Hessian of a scalar function $f(\mathbf{x})$
    using the forward finite-difference scheme.

    $$ H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
        = \frac{
          f(\mathbf{x} + \mathbf{e}_i h_i + \mathbf{e}_j h_j)
        - f(\mathbf{x} + \mathbf{e}_i h_i) - f(\mathbf{x} + \mathbf{e}_j h_j)
        + f(\mathbf{x})}{h_i h_j} $$

    The step size $h_j$ is optimally determined according to the machine precision of the
    function values. Typically, the Hessian is accurate to about a third of the number of
    reliable digits returned by the function.

    **References**

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained Optimization
        and Nonlinear Equations", SIAM, 1996, p. 321.

    Parameters
    ----------
    f : Callable[[FloatVector], float]
        Function to be differentiated.
    x : FloatVector
        Differentiation point.
    fx : float | None
        Function value at `x`, if available.
    sclx : FloatVector | None
        Scaling factors for `x`. Ideally, `x[i]*sclx[i]` is close to 1. By
        default, the factors are set internally based on the magnitudes of `x`.
    epsf : float | None
        Machine precision of the function values. If `None`, machine precision of 64-bit
        floating-point type is assumed. If the number of reliable base-10 digits in the
        results returned by the function is $n$, then `epsf` is approximately $10^{-n}$.

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
    array([[-16.00001242,  47.99984347],
           [ 47.99984347, -47.99972236]])
    """
    fx = fx if fx is not None else f(x)
    sclx = np.abs(sclx) if sclx is not None else scalex(x)

    epsf = max(epsf, eps) if epsf is not None else eps
    h0 = math.cbrt(epsf)

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
    epsf: float | None = None,
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

    Although the matrix only contains 4 elements and is symmetric, a total of 9 function
    evaluations are performed.

    **References**

    *   J. Martins and A. Ning. Engineering Design Optimization. Cambridge University
        Press, 2021.

    Parameters
    ----------
    f : Callable[[tuple[float, float]], float]
        Function to be differentiated.
    x : tuple[float, float]
        Differentiation point.
    epsf : float | None
        Machine precision of the function values. If `None`, machine precision of 64-bit
        floating-point type is assumed. If the number of reliable base-10 digits in the
        results returned by the function is $n$, then `epsf` is approximately $10^{-n}$.

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

    epsf = max(epsf, eps) if epsf is not None else eps
    h = math.cbrt(3 * epsf)
    h0 = h * max(1.0, abs(x0))
    h1 = h * max(1.0, abs(x1))

    H = np.empty((2, 2))
    fx = f(x)
    H[0, 0] = (f((x0 + 2 * h0, x1)) - 2 * fx + f((x0 - 2 * h0, x1))) / (4 * h0**2)
    H[1, 1] = (f((x0, x1 + 2 * h1)) - 2 * fx + f((x0, x1 - 2 * h1))) / (4 * h1**2)
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
