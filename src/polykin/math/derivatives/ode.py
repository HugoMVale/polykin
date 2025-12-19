# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import math
from collections.abc import Callable
from typing import Literal

from numba import njit

__all__ = ["ode_rk"]


@njit
def ode_rk(
    f: Callable[[float, float], float],
    t0: float,
    tf: float,
    y0: float,
    h: float,
    *,
    order: Literal[1, 2, 4] = 1,
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
    >>> ode_rk(njit(ydot), 0.0, 2.0, 1.0, 1e-3, order=1)
    11.763351307112204
    >>> ode_rk(njit(ydot), 0.0, 2.0, 1.0, 1e-3, order=2)
    11.778107275517668
    >>> ode_rk(njit(ydot), 0.0, 2.0, 1.0, 1e-3, order=4)
    11.778112197860988
    """

    def step_rk1(f, t, y, h):
        """Explicit 1st-order (Euler) step."""
        y += f(t, y) * h
        return y

    def step_rk2(f, t, y, h):
        """Explicit 2nd-order mid-point step."""
        y += f(t + h / 2, y + f(t, y) * h / 2) * h
        return y

    def step_rk4(f, t, y, h):
        """Explicit 4th-order Runge-Kutta step."""
        k1 = f(t, y)
        k2 = f(t + h / 2, y + k1 * h / 2)
        k3 = f(t + h / 2, y + k2 * h / 2)
        k4 = f(t + h, y + k3 * h)
        y += (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
        return y

    def integrate(step, f, t0, tf, y0, h):
        """Integrate using the given step function."""
        nsteps = math.floor((tf - t0) / h)
        hf = (tf - t0) - nsteps * h
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
