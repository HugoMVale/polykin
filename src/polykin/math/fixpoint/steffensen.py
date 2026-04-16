# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

from collections.abc import Callable

from polykin.math.roots import RootResult
from polykin.utils.math import eps

__all__ = ["fixpoint_steffensen"]


def fixpoint_steffensen(
    g: Callable[[float], float],
    x0: float,
    *,
    tolx: float = 1e-6,
    maxiter: int = 50,
    callback: Callable[[int, float, float], bool] | None = None,
) -> RootResult:
    r"""Find the solution of a scalar fixed-point problem using Steffensen's
    method.

    Steffensen's method accelerates the direct substitution iteration for a
    scalar fixed-point problem, `g(x)=x`, by applying Aitken's delta-squared
    process to the sequence of fixed-point iterates. The update can be written
    as:

    $$ x_{k+1} = x_k - \frac{(g(x_k) - x_k)^2}{g(g(x_k)) - 2 g(x_k) + x_k} $$

    When the denominator becomes very small, the accelerated step becomes
    numerically unreliable and the method terminates.

    Parameters
    ----------
    g : Callable[[float], float]
        Fixed-point mapping defining the problem `g(x) = x`.
    x0 : float
        Initial guess.
    tolx : float
        Absolute tolerance for the fixed-point residual. The algorithm will
        terminate when `|g(x) - x| <= tolx`.
    maxiter : int
        Maximum number of iterations.
    callback : Callable[[int, float, float], bool] | None
        Optional callback with signature `callback(niter, x, fx)` called at the end of
        each iteration. If the callback returns `True`, the iteration is terminated.

    Returns
    -------
    RootResult
        Dataclass with root solution results.

    Examples
    --------
    Find the fixed point of the cosine function.
    >>> from numpy import cos
    >>> from polykin.math import fixpoint_steffensen
    >>> sol = fixpoint_steffensen(cos, 0.5)
    >>> print(f"x = {sol.x:.6f}")
    x = 0.739085
    >>> print(f"g(x) - x = {cos(sol.x) - sol.x:.2e}")
    g(x) - x = 1.92e-11
    """
    method = "Steffensen fixed-point"
    success = False
    message = ""
    nfeval = 0

    x = x0
    fx = float("nan")
    niter = 0

    for niter in range(1, maxiter + 1):
        gx = g(x)
        nfeval += 1
        fx = gx - x

        if callback is not None and callback(niter, x, fx):
            message = "Terminated by user callback."
            success = True
            break

        if abs(fx) <= tolx:
            message = "|g(x) - x| ≤ tolx"
            success = True
            break

        if niter < maxiter:
            ggx = g(gx)
            nfeval += 1

            Δ2 = ggx - 2 * gx + x
            if abs(Δ2) <= eps * max(abs(ggx), abs(gx), abs(x), 1.0):
                message = f"Nearly zero Steffensen denominator at x={x} (Δ²={Δ2:.2e})."
                break

            x = x - fx**2 / Δ2

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return RootResult(method, success, message, nfeval, niter, x, fx)
