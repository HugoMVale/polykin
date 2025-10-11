# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import math
from dataclasses import dataclass
from typing import Callable

from polykin.math.derivatives import derivative_complex
from polykin.utils.math import eps
from polykin.utils.types import FloatVector

__all__ = [
    'fzero_newton',
    'fzero_secant',
    'fzero_brent'
]


@dataclass
class RootResult():
    """Dataclass with root solution results.

    Attributes
    ----------
    success: bool
        If `True`, the root was found.
    niter: int
        Number of iterations.
    x: float
        Root value.
    f: float
        Function value at root.
    """
    success: bool
    niter: int
    x: float
    f: float


def fzero_newton(f: Callable[[complex], complex],
                 x0: float,
                 xtol: float = 1e-6,
                 ftol: float = 1e-6,
                 maxiter: int = 50
                 ) -> RootResult:
    r"""Find the root of a scalar function using the newton method.

    Unlike the equivalent method in [scipy](https://docs.scipy.org/doc/scipy/reference/optimize.root_scalar-newton.html),
    this method uses complex step differentiation to estimate the derivative of
    $f(x)$ without loss of precision. Therefore, there is no need to provide
    $f'(x)$. It's application is restricted to real functions that can be
    evaluated with complex inputs, but which per se do not implement complex
    arithmetic.

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
    success = False
    niter = 0
    while niter < maxiter:
        dfdx, f0 = derivative_complex(f, x0)
        if (abs(f0) <= ftol):
            success = True
            break
        x1 = x0 - f0 / dfdx
        niter += 1
        if (abs(x1 - x0) <= xtol):
            success = True
            x0 = x1
            f0 = f(x0).real
            break
        x0 = x1

    return RootResult(success, niter, x0, f0)


def fzero_secant(f: Callable[[float], float],
                 x0: float,
                 x1: float,
                 xtol: float = 1e-6,
                 ftol: float = 1e-6,
                 maxiter: int = 50
                 ) -> RootResult:
    r"""Find the root of a scalar function using the secant method.

    Unlike the equivalent method in [scipy](https://docs.scipy.org/doc/scipy/reference/optimize.root_scalar-secant.html),
    this method also terminates based on the function value. This is sometimes
    a more meaningful stop criterion.

    Parameters
    ----------
    f : Callable[[float], float]
        Function whose root is to be found.
    x0 : float
        Inital guess.
    x1 : float
        Second guess.
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
    >>> from math import log
    >>> def f(x, a=0.6, chi=0.4):
    ...     return log(x) + (1 - x) + chi*(1 - x)**2 - log(a)
    >>> sol = fzero_secant(f, 0.3, 0.31)
    >>> print(f"x= {sol.x:.3f}")
    x= 0.213
    """

    f0 = f(x0)
    if (abs(f0) <= ftol):
        return RootResult(True, 0, x0, f0)
    f1 = f(x1)
    if (abs(f1) <= ftol):
        return RootResult(True, 0, x1, f1)

    success = False
    niter = 0
    while niter < maxiter:
        x2 = x1 - f1*(x1 - x0)/(f1 - f0)
        f2 = f(x2)
        niter += 1
        if (abs(x2 - x1) <= xtol) or (abs(f2) <= ftol):
            success = True
            break
        x0, f0 = x1, f1
        x1, f1 = x2, f2

    return RootResult(success, niter, x2, f2)


def fzero_brent(f: Callable[[float], float],
                xa: float,
                xb: float,
                xtol: float = 1e-6,
                ftol: float = 1e-6,
                maxiter: int = 50
                ) -> RootResult:
    r"""Find the root of a scalar function using Brent's method.

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
    >>> from math import log
    >>> def f(x, a=0.6, chi=0.4):
    ...     return log(x) + (1 - x) + chi*(1 - x)**2 - log(a)
    >>> sol = fzero_brent(f, 0.1, 0.9)
    >>> print(f"x= {sol.x:.3f}")
    x= 0.213
    """

    fa = f(xa)
    if (abs(fa) <= ftol):
        return RootResult(True, 0, xa, fa)
    fb = f(xb)
    if (abs(fb) <= ftol):
        return RootResult(True, 0, xb, fb)

    if (fa*fb) > 0.0:
        raise ValueError("Root is not bracketed.")

    xc, fc = xb, fb
    success = False
    for iter in range(maxiter):
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
        if (abs(m) <= tol1) or (abs(fb) <= ftol):
            success = True
            # return xb
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
            xb += math.copysign(tol1, m)
        fb = f(xb)

    return RootResult(success, iter+1, xb, fb)
