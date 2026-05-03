# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

from collections.abc import Callable
from typing import Literal

from polykin.math.derivatives import (
    derivative_centered,
    derivative_complex,
)
from polykin.math.machine import eps, sqrt_eps
from polykin.math.optimization.results import OptimumResult

__all__ = ["fmin_secant"]


def fmin_secant(
    f: Callable[[float], float] | Callable[[complex], complex],
    x0: float,
    x1: float,
    *,
    tolx: float = 1e-10,
    tolg: float = 1e-5,
    maxiter: int = 50,
    ndigit: int | None = None,
    diff_scheme: Literal["centered", "complex"] = "centered",
    callback: Callable[[int, float, float, float], tuple[bool, bool]] | None = None,
) -> OptimumResult:
    r"""Find the minimum of a scalar function using the secant method.

    The secant method starts from two initial guesses and applies the secant update
    to the first derivative $f'(x)$ to generate the next iterate:

    $$ x_{k+1} = x_k - f'(x_k) \frac{x_k - x_{k-1}}{f'(x_k) - f'(x_{k-1})} $$

    The derivative $f'(x)$ is approximated numerically using either centered finite
    differences or complex-step differentiation. The complex-step scheme is usually
    more accurate, but requires that `f` accepts complex-valued inputs.

    This is an efficient local method for smooth functions, but it is less robust
    than bracketed methods such as Brent's algorithm and may fail if the initial
    guesses are poor.

    Parameters
    ----------
    f : Callable[[float], float] | Callable[[complex], complex]
        Objective function to be minimized.
    x0 : float
        First initial guess.
    x1 : float
        Second initial guess.
    tolx : float
        Absolute tolerance for `x`. The algorithm will terminate when the change in
        `x` between two iterations is less or equal than `tolx`. If the value is too
        large, the algorithm may terminate prematurely. A value on the order of
        $\epsilon^{2/3}$ is typically recommended.
    tolg : float
        Absolute tolerance for the function gradient. This is the primary convergence
        criterion. The algorithm will terminate when `|f'(x)| <= tolg`. A value on the
        order of $\epsilon^{1/3}$ is typically recommended.
    maxiter : int
        Maximum number of iterations.
    ndigit : int | None
        Number of reliable digits returned by `f`. Used to set the step size for centered
        finite-difference derivative approximations. By default, 64-bit float precision is
        assumed (i.e., ~15 digits).
    diff_scheme : Literal['centered', 'complex']
        Numerical differentiation scheme used to approximate `f'(x)`. The 'centered'
        scheme uses a centered finite difference, while the 'complex' scheme uses
        complex step differentiation. The 'complex' scheme is more accurate, but requires
        that `f` can accept complex inputs.
    callback : Callable[[int, float, float, float], tuple[bool, bool]] | None
        Optional callback with signature `callback(niter, x, fx, dfx)->(stop, success)`
        called at each iteration. If `stop` is `True`, the iteration is terminated. If
        `success` is `True`, the optimization is considered successful.

    Returns
    -------
    OptimumResult
        Dataclass with the results of the optimization.

    See Also
    --------
    * [`fmin_brent`](fmin_brent.md):
      More robust derivative-free minimization method for bounded intervals.

    Examples
    --------
    Find the minimum of the function `f(x) = (x - 2)^2 + 1`.
    >>> from polykin.math import fmin_secant
    >>> f = lambda x: (x - 2)**2 + 1
    >>> sol = fmin_secant(f, 3.0, 3.1)
    >>> print(f"x = {sol.x:.6f}, f(x) = {sol.f:.6f}")
    x = 2.000000, f(x) = 1.000000
    """
    # Initialize results
    method = "Secant"
    success = False
    message = ""
    nfeval = 0

    # Set base step for centered finite difference
    h0 = 10 ** (-max(1, min(ndigit, 15)) // 3) if ndigit is not None else 0.0

    # Helper function to evaluate the derivative
    def eval_derivative(x: float) -> tuple[float, float]:
        nonlocal nfeval
        if diff_scheme == "centered":
            dfx, fx = derivative_centered(f, x, h=h0 * max(1.0, abs(x)))
            nfeval += 2
        elif diff_scheme == "complex":
            dfx, fx = derivative_complex(f, x)
            nfeval += 1
        else:
            raise ValueError(f"Invalid differentiation scheme: {diff_scheme!r}")

        return (dfx, fx)

    # Evaluate derivatives at initial guesses
    df0, f0 = eval_derivative(x0)
    if abs(df0) <= tolg:
        message = "|f'(x0)| ≤ tolg."
        success = True
        return OptimumResult(method, success, message, nfeval, 0, x0, f0, df0)

    df1, f1 = eval_derivative(x1)
    if abs(df1) <= tolg:
        message = "|f'(x1)| ≤ tolg."
        success = True
        return OptimumResult(method, success, message, nfeval, 0, x1, f1, df1)

    # Main optimization loop
    x2 = f2 = df2 = float("nan")
    niter = 0

    for niter in range(1, maxiter + 1):
        Δdf = df1 - df0
        if abs(Δdf) <= eps * max(abs(df0), abs(df1), 1.0):
            message = f"Nearly zero slope between x[k-1]={x0} and x[k]={x1} (Δf'={Δdf})."
            break

        x2 = x1 - df1 * (x1 - x0) / Δdf

        df2, f2 = eval_derivative(x2)

        if callback is not None:
            stop, _success = callback(niter, x2, f2, df2)
            if stop:
                message = "Terminated by user callback."
                success = _success
                break

        if (f2 - f1) > sqrt_eps * max(abs(f1), abs(f2), 1.0):
            message = (
                f"Function value increased from {f1} at x[k-1]={x1} to {f2} at x[k]={x2}."
            )
            break

        if abs(df2) <= tolg:
            message = "|f'(x)| ≤ tolg"
            success = True
            break

        if abs(x2 - x1) <= tolx:
            message = "|Δx| ≤ tolx"
            success = True
            break

        x0, f0, df0 = x1, f1, df1
        x1, f1, df1 = x2, f2, df2

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return OptimumResult(method, success, message, nfeval, niter, x2, f2, df2)
