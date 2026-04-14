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
    k = 0

    for k in range(maxiter):
        gx = g(x)
        nfeval += 1
        fx = gx - x

        if abs(fx) <= tolx:
            message = "|g(x) - x| ≤ tolx"
            success = True
            break

        if k + 1 < maxiter:
            ggx = g(gx)
            nfeval += 1

            d2 = ggx - 2 * gx + x
            if abs(d2) <= eps * max(abs(ggx), abs(gx), abs(x), 1.0):
                message = f"Nearly zero Steffensen denominator at x={x} (Δ²={d2:.2e})."
                break

            x = x - fx**2 / d2

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return RootResult(method, success, message, nfeval, k + 1, x, fx)
