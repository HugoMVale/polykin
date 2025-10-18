# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from dataclasses import dataclass
from typing import Callable

import numpy as np

from polykin.math.derivatives import derivative_complex
from polykin.utils.math import eps

__all__ = [
    'fzero_newton',
    'fzero_secant',
    'fzero_brent',
    'RootResult',
]


@dataclass
class RootResult():
    """Dataclass with root solution results.

    Attributes
    ----------
    success: bool
        If `True`, the root was found.
    message: str
        Description of the exit status.
    nfeval: int
        Number of function evaluations.
    niter: int
        Number of iterations.
    x: float
        Root value.
    f: float
        Function (residual) value at root.
    """
    success: bool
    message: str
    nfeval: int
    niter: int
    x: float
    f: float


def fzero_newton(f: Callable[[complex], complex],
                 x0: float,
                 xtol: float = 1e-6,
                 ftol: float = 1e-6,
                 maxiter: int = 50
                 ) -> RootResult:
    r"""Find the root of a scalar function using the Newton-Raphson method.

    The Newton-Raphson method uses the first derivative of the function to
    iteratively find the root according to the formula:

    $$ x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)} $$   

    Unlike the equivalent method in [scipy](https://docs.scipy.org/doc/scipy/reference/optimize.root_scalar-newton.html),
    this method uses complex step differentiation to estimate the derivative of
    $f(x)$ without loss of precision. Therefore, there is no need to provide
    $f'(x)$. Its application is restricted to real functions that can be
    evaluated with complex inputs, but which per se do not implement complex
    arithmetic.

    !!! note

        The functions from the `math` module (e.g., `math.log`) do not support
        complex arguments. Use the equivalent functions from `numpy` instead.

    Parameters
    ----------
    f : Callable[[complex], complex]
        Function whose root is to be found.
    x0 : float
        Inital guess.
    xtol : float
        Absolute tolerance for `x` value. The algorithm will terminate when the
        change in `x` between two iterations is less or equal than `xtol`.
    ftol : float
        Absolute tolerance for function value. The algorithm will terminate
        when `|f(x)|<=ftol`.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    RootResult
        Dataclass with root solution results.

    Examples
    --------
    Find a root of the Flory-Huggins equation.
    >>> from polykin.math import fzero_newton
    >>> from numpy import log
    >>> def f(x, a=0.6, chi=0.4):
    ...     return log(x) + (1 - x) + chi*(1 - x)**2 - log(a)
    >>> sol = fzero_newton(f, 0.3)
    >>> print(f"x= {sol.x:.3f}")
    x= 0.213
    """

    nfeval = 0
    message = ""
    success = False

    x = x0

    for k in range(maxiter):

        dfdx, fx = derivative_complex(f, x)
        nfeval += 1

        if abs(fx) <= ftol:
            message = "|f(x)| <= ftol"
            success = True
            break

        if abs(dfdx) <= eps:
            message = f"Nearly zero derivative at x={x} (df/dx={dfdx})."
            break

        Δx = - fx / dfdx

        if (abs(Δx) <= xtol):
            message = "|Δx| <= xtol"
            success = True
            break

        if k + 1 < maxiter:
            x += Δx

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return RootResult(success, message, nfeval, k+1, x, fx)


def fzero_secant(f: Callable[[float], float],
                 x0: float,
                 x1: float,
                 xtol: float = 1e-6,
                 ftol: float = 1e-6,
                 maxiter: int = 50
                 ) -> RootResult:
    r"""Find the root of a scalar function using the secant method.

    The secant method uses two initial guesses and approximates the derivative
    of the function to iteratively find the root according to the formula:

    $$ x_{k+1} = x_k - f(x_k) \frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})} $$

    Unlike the equivalent method in [scipy](https://docs.scipy.org/doc/scipy/reference/optimize.root_scalar-secant.html),
    this method also terminates based on the function value. This is sometimes
    a more meaningful stop criterion.

    Parameters
    ----------
    f : Callable[[float], float]
        Function whose root is to be found.
    x0 : float
        First initial guess.
    x1 : float
        Second initial guess.
    xtol : float
        Absolute tolerance for `x` value. The algorithm will terminate when the
        change in `x` between two iterations is less or equal than `xtol`.
    ftol : float
        Absolute tolerance for function value. The algorithm will terminate
        when `|f(x)|<=ftol`.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    RootResult
        Dataclass with root solution results.

    Examples
    --------
    Find a root of the Flory-Huggins equation.
    >>> from polykin.math import fzero_secant
    >>> from numpy import log
    >>> def f(x, a=0.6, chi=0.4):
    ...     return log(x) + (1 - x) + chi*(1 - x)**2 - log(a)
    >>> sol = fzero_secant(f, 0.3, 0.31)
    >>> print(f"x= {sol.x:.3f}")
    x= 0.213
    """

    nfeval = 0
    message = ""
    success = False

    f0 = f(x0)
    nfeval += 1
    if abs(f0) <= ftol:
        message = "|f(x0)| <= ftol"
        return RootResult(True, message, nfeval, 0, x0, f0)

    f1 = f(x1)
    nfeval += 1
    if abs(f1) <= ftol:
        message = "|f(x1)| <= ftol"
        return RootResult(True, message, nfeval, 0, x1, f1)

    x2, f2 = np.nan, np.nan

    for k in range(maxiter):

        Δf = f1 - f0
        if abs(Δf) <= eps * max(abs(f0), abs(f1), 1.0):
            message = f"Nearly zero slope between x[k-1]={x0} and x[k]={x1} (Δf={Δf})."
            break

        x2 = x1 - f1*(x1 - x0) / Δf
        f2 = f(x2)
        nfeval += 1

        if (abs(x2 - x1) <= xtol):
            message = "|Δx| <= xtol"
            success = True
            break

        if (abs(f2) <= ftol):
            message = "|f(x)| <= ftol"
            success = True
            break

        x0, f0 = x1, f1
        x1, f1 = x2, f2

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return RootResult(success, message, nfeval, k+1, x2, f2)


def fzero_brent(f: Callable[[float], float],
                xa: float,
                xb: float,
                xtol: float = 1e-6,
                ftol: float = 1e-6,
                maxiter: int = 50
                ) -> RootResult:
    r"""Find the root of a scalar function using Brent's method.

    Brent's method is a root-finding algorithm combining bisection, secant,
    and inverse quadratic interpolation. It is robust like the bisection method
    and fast like the secant method.

    Unlike the equivalent method in [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html),
    this method also terminates based on the function value. This is sometimes
    a more meaningful stop criterion.

    **References**

    * William H. Press, Saul A. Teukolsky, William T. Vetterling, and 
      Brian P. Flannery. 2007. "Numerical Recipes 3rd Edition: The Art of 
      Scientific Computing" (3rd. ed.). Cambridge University Press, USA.

    Parameters
    ----------
    f : Callable[[float], float]
        Function whose root is to be found.
    xa : float
        Lower bound of the bracketing interval.
    xb : float
        Upper bound of the bracketing interval.
    xtol : float
        Absolute tolerance for `x` value. The algorithm will terminate when the
        change in `x` between two iterations is less or equal than `xtol`.
    ftol : float
        Absolute tolerance for function value. The algorithm will terminate
        when `|f(x)|<=ftol`.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    RootResult
        Dataclass with root solution results.

    Examples
    --------
    Find a root of the Flory-Huggins equation.
    >>> from polykin.math import fzero_brent
    >>> from numpy import log
    >>> def f(x, a=0.6, chi=0.4):
    ...     return log(x) + (1 - x) + chi*(1 - x)**2 - log(a)
    >>> sol = fzero_brent(f, 0.1, 0.9)
    >>> print(f"x= {sol.x:.3f}")
    x= 0.213
    """

    nfeval = 0
    message = ""
    success = False

    fa = f(xa)
    nfeval += 1
    if abs(fa) <= ftol:
        message = "|f(xa)| <= ftol"
        return RootResult(True, message, nfeval, 0, xa, fa)

    fb = f(xb)
    nfeval += 1
    if abs(fb) <= ftol:
        message = "|f(xb)| <= ftol"
        return RootResult(True, message, nfeval, 0, xb, fb)

    if (fa*fb) > 0.0:
        raise ValueError("Root is not bracketed.")

    xc, fc = xb, fb

    for k in range(maxiter):

        if (fb*fc > 0.0):
            xc, fc = xa, fa
            d = xb - xa
            e = d

        if abs(fc) < abs(fb):
            xa, fa = xb, fb
            xb, fb = xc, fc
            xc, fc = xa, fa

        tol1 = 2*eps*abs(xb) + 0.5*xtol
        m = 0.5*(xc - xb)
        if abs(m) <= tol1:
            message = "|Δx| <= xtol"
            success = True
            break

        if abs(fb) <= ftol:
            message = "|f(x)| <= ftol"
            success = True
            break

        if (abs(e) >= tol1) and (abs(fa) > abs(fb)):
            s = fb/fa
            if xa == xc:
                p = 2*m*s
                q = 1 - s
            else:
                q = fa/fc
                r = fb/fc
                p = s*(2*m*q*(q - r) - (xb - xa)*(r - 1))
                q = (q - 1)*(r - 1)*(s - 1)
            if p > 0:
                q = -q
            p = abs(p)
            min1 = 3*m*q - abs(tol1*q)
            min2 = abs(e*q)
            if 2*p < min(min1, min2):
                e = d
                d = p/q
            else:
                d = m
                e = d
        else:
            d = m
            e = d

        xa, fa = xb, fb
        if abs(d) > tol1:
            xb += d
        else:
            xb += np.copysign(tol1, m)

        fb = f(xb)
        nfeval += 1

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return RootResult(success, message, nfeval, k+1, xb, fb)
