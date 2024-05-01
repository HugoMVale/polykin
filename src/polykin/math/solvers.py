# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from dataclasses import dataclass
from typing import Callable

from polykin.math.derivatives import derivative_complex

__all__ = ['root_newton', 'root_secant']


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


def root_newton(f: Callable[[complex], complex],
                x0: float,
                xtol: float = 1e-6,
                ftol: float = 1e-6,
                maxiter: int = 50
                ) -> RootResult:
    r"""Find the root of a scalar function using the newton method.

    Unlike the equivalent method in scipy, this method uses complex step
    differentiation to estimate the derivative of $f(x)$ without loss of
    precision. Therefore, there is no need to provide $f'(x)$. It's application
    is restricted to real functions that can be evaluated with complex inputs,
    but which per se do not implement complex arithmetic.

    Parameters
    ----------
    f : Callable[[complex], compley]
        Function whose root is to be found.
    x0 : float
        Inital guess.
    xtol : float, optional
        Absolute tolerance for `x` value. The algorithm will terminate when the
        change in `x` between two iterations is smaller than `xtol`.
    ftol : float, optional
        Absolute tolerance for function value. The algorithm will terminate
        when `|f(x)|<ftol`.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    RootSolverResult
        Dataclass with root solution results.

    Examples
    --------
    Find a root of the Flory-Huggings equation.
    >>> from polykin.math import root_secant
    >>> from math import log
    >>> def f(x, a=0.6, chi=0.4):
    ...     return log(x) + (1 - x) + chi*(1 - x)**2 - log(a)
    >>> sol = root_secant(f, 0.3, 0.31)
    >>> print(f"x= {sol.x:.2f}")
    x= 0.21
    """
    success = False
    niter = 0
    while niter < maxiter:
        dfdx, f0 = derivative_complex(f, x0)
        if (abs(f0) < ftol):
            success = True
            break
        x1 = x0 - f0 / dfdx
        niter += 1
        if (abs(x1 - x0) < xtol):
            success = True
            x0 = x1
            f0 = f(x0).real
            break
        x0 = x1

    return RootResult(success, niter, x0, f0)


def root_secant(f: Callable[[float], float],
                x0: float,
                x1: float,
                xtol: float = 1e-6,
                ftol: float = 1e-6,
                maxiter: int = 50
                ) -> RootResult:
    r"""Find the root of a scalar function using the secant method.

    Unlike the equivalent method in scipy, this method also terminates based
    on the function value. This is sometimes a more meaningful stop criterion.

    Parameters
    ----------
    f : Callable[[float], float]
        Function whose root is to be found.
    x0 : float
        Inital guess.
    x1 : float
        Second guess.
    xtol : float, optional
        Absolute tolerance for `x` value. The algorithm will terminate when the
        change in `x` between two iterations is smaller than `xtol`.
    ftol : float, optional
        Absolute tolerance for function value. The algorithm will terminate
        when `|f(x)|<ftol`.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    RootSolverResult
        Dataclass with root solution results.

    Examples
    --------
    Find a root of the Flory-Huggings equation.
    >>> from polykin.math import root_secant
    >>> from math import log
    >>> def f(x, a=0.6, chi=0.4):
    ...     return log(x) + (1 - x) + chi*(1 - x)**2 - log(a)
    >>> sol = root_secant(f, 0.3, 0.31)
    >>> print(f"x= {sol.x:.2f}")
    x= 0.21
    """

    f0 = f(x0)
    if (abs(f0) < ftol):
        return RootResult(True, 0, x0, f0)
    f1 = f(x1)
    if (abs(f1) < ftol):
        return RootResult(True, 0, x1, f1)

    success = False
    niter = 0
    while niter < maxiter:
        x2 = x1 - f1*(x1 - x0)/(f1 - f0)
        f2 = f(x2)
        niter += 1
        if (abs(x2 - x1) < xtol) or (abs(f2) < ftol):
            success = True
            break
        x0 = x1
        x1 = x2
        f0 = f1
        f1 = f2

    return RootResult(success, niter, x2, f2)
