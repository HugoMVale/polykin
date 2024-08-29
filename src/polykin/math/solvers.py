# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import math
from dataclasses import dataclass
from typing import Callable, Literal

from numba import njit

from polykin.math.derivatives import derivative_complex

__all__ = ['root_newton', 'root_secant', 'ode_rk']


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
    RootResult
        Dataclass with root solution results.

    Examples
    --------
    Find a root of the Flory-Huggins equation.
    >>> from polykin.math import root_newton
    >>> from numpy import log
    >>> def f(x, a=0.6, chi=0.4):
    ...     return log(x) + (1 - x) + chi*(1 - x)**2 - log(a)
    >>> sol = root_newton(f, 0.3)
    >>> print(f"x= {sol.x:.3f}")
    x= 0.213
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
    RootResult
        Dataclass with root solution results.

    Examples
    --------
    Find a root of the Flory-Huggins equation.
    >>> from polykin.math import root_secant
    >>> from math import log
    >>> def f(x, a=0.6, chi=0.4):
    ...     return log(x) + (1 - x) + chi*(1 - x)**2 - log(a)
    >>> sol = root_secant(f, 0.3, 0.31)
    >>> print(f"x= {sol.x:.3f}")
    x= 0.213
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


@njit
def ode_rk(f: Callable[[float, float], float],
           t0: float,
           tf: float,
           y0: float,
           h: float,
           order: Literal[1, 2, 4] = 1
           ) -> float:
    r"""Integrate an ODE using a fixed-step Runge-Kutta scheme.

    This method is intentionally simple, so that it can be used inside a
    gradient-based optimizer without creating numerical noise and overhead.

    !!! important

        This method is jitted with Numba and, thus, requires a JIT-compiled
        function.

    Parameters
    ----------
    f : Callable[[float, float], float]
        Function to be integrated. Takes two arguments, `t` and `y`, and returns
        the derivative of `y` with respect to `t`.
    t0 : float
        Initial value of `t`.
    tf : float
        Final value of `t`.
    y0 : float
        Initial value of `y`, i.e., `y(t0)`.
    h : float
        Step size.
    order : Literal[1, 2, 4]
        Order of the method. Defaults to 1 (i.e., Euler).

    Returns
    -------
    float
        Final value of `y(tf)`.

    Examples
    --------
    Find the solution of the differential equation $y(t)'=y+t$ with initial
    condition $y(0)=1$ at $t=2$.
    >>> from polykin.math import ode_rk
    >>> from numba import njit
    >>> def ydot(t, y):
    ...     return y + t
    >>> ode_rk(njit(ydot), 0., 2., 1., 1e-3, order=1)
    11.763351307112204
    >>> ode_rk(njit(ydot), 0., 2., 1., 1e-3, order=2)
    11.778107275517668
    >>> ode_rk(njit(ydot), 0., 2., 1., 1e-3, order=4)
    11.778112197860988
    """

    def step_rk1(f, t, y, h):
        "Explicit 1st-order (Euler) step."
        y += f(t, y) * h
        return y

    def step_rk2(f, t, y, h):
        "Explicit 2nd-order mid-point step."
        y += f(t + h / 2, y + f(t, y) * h / 2) * h
        return y

    def step_rk4(f, t, y, h):
        "Explicit 4th-order Runge-Kutta step."
        k1 = f(t, y)
        k2 = f(t + h / 2, y + k1 * h / 2)
        k3 = f(t + h / 2, y + k2 * h / 2)
        k4 = f(t + h, y + k3 * h)
        y += (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
        return y

    def integrate(step, f, t0, tf, y0, h):
        "Integrate using the given step function."
        nsteps = math.floor((tf - t0)/h)
        hf = (tf - t0) - nsteps*h
        t = t0
        y = y0
        for _ in range(nsteps):
            y = step(f, t, y, h)
            t += h
        y = step(f, t, y, hf)
        t += hf
        return y

    # Needs to be done in this explict/verbose form, because function pointers
    # are not supported in numba.
    if order == 1:
        return integrate(step_rk1, f, t0, tf, y0, h)
    elif order == 2:
        return integrate(step_rk2, f, t0, tf, y0, h)
    elif order == 4:
        return integrate(step_rk4, f, t0, tf, y0, h)
    else:
        raise ValueError("Order must be 1, 2, or 4.")
