# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

import math
from collections.abc import Callable

from polykin.math.machine import eps
from polykin.math.optimization.results import OptimumResult

__all__ = ["fmin_brent"]


def fmin_brent(
    f: Callable[[float], float],
    xa: float,
    xb: float,
    *,
    tolx: float = 1e-6,
    maxiter: int = 50,
    callback: Callable[[int, float, float], tuple[bool, bool]] | None = None,
) -> OptimumResult:
    r"""Find the minimum of a scalar function using Brent's method.

    Brent's method is a derivative-free optimization algorithm that combines
    golden-section search with inverse parabolic interpolation. It maintains a
    bracketing interval that contains a local minimum and iteratively refines
    this interval. When the function behaves smoothly, the method attempts a
    fast parabolic step; otherwise, it falls back to the more robust golden-
    section step.

    **References**

    *   Brent, R. P. Algorithms for Minimization without Derivatives; Prentice-Hall:
        Englewood Cliffs, NJ, 1973.

    Parameters
    ----------
    f : Callable[[float], float]
        Objective function to be minimized.
    xa : float
        Lower bound of the bracketing interval.
    xb : float
        Upper bound of the bracketing interval.
    tolx : float, optional
        Absolute tolerance for `x` value. The algorithm terminates when the search
        interval becomes smaller than approximately `tolx`.
    maxiter : int
        Maximum number of iterations.
    callback : Callable[[int, float, float], tuple[bool, bool]] | None
        Optional callback with signature `callback(niter, x, fx)->(stop, success)` called
        at each iteration. If `stop` is `True`, the iteration is terminated. If `success`
        is `True`, the optimization is considered successful.

    Returns
    -------
    OptimumResult
        Dataclass with the results of the optimization.

    See Also
    --------
    * [`fmin_secant`](fmin_secant.md):
      Derivative-based minimization method using the secant approach.

    Examples
    --------
    Find the minimum of the function `f(x) = x^4 - x + 1`.
    >>> from polykin.math import fmin_brent
    >>> f = lambda x: x**4 - x + 1
    >>> fmin_brent(f, -3.0, 3.0)
     method: Brent
    success: True
    message: |Δx| ≤ tolx
     nfeval: 14
      niter: 14
          x: 0.6299605262795589
          f: 0.5275296062894226
         df: None
    """
    method = "Brent"
    success = False
    message = ""
    nfeval = 0

    # Golden ratio constant: (3 - sqrt(5)) / 2
    c = 0.38196601125010515179

    # Initialization
    a = min(xa, xb)
    b = max(xa, xb)
    v = w = x = a + c * (b - a)
    fv = fw = fx = f(x)
    nfeval += 1

    d = e = 0.0
    niter = 0

    for niter in range(1, maxiter + 1):
        xm = 0.5 * (a + b)
        tol1 = eps * abs(x) + tolx / 3.0
        tol2 = 2.0 * tol1

        if callback is not None:
            stop, _success = callback(niter, x, fx)
            if stop:
                message = "Terminated by user callback."
                success = _success
                break

        if abs(x - xm) <= (tol2 - 0.5 * (b - a)):
            message = "|Δx| ≤ tolx"
            success = True
            break

        p = q = r = 0.0
        if abs(e) > tol1:
            # Fit parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = d

            # Is parabolic step acceptable?
            if abs(p) < abs(0.5 * q * r) and p > q * (a - x) and p < q * (b - x):
                d = p / q
                u = x + d
                # Convergence check for u
                if (u - a) < tol2 or (b - u) < tol2:
                    d = math.copysign(tol1, xm - x)
            else:
                # Golden section step
                e = b - x if x < xm else a - x
                d = c * e
        else:
            # Golden section step
            e = b - x if x < xm else a - x
            d = c * e

        # Numerical safety: ensure step is at least tol1
        u = x + d if abs(d) >= tol1 else x + math.copysign(tol1, d)
        fu = f(u)
        nfeval += 1

        # Update points
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return OptimumResult(method, success, message, nfeval, niter, x, fx)
