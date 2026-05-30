# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

from collections.abc import Callable

import numpy as np

from polykin.math.derivatives import scalex
from polykin.math.machine import eps
from polykin.math.roots import VectorRootResult
from polykin.utils.typing import FloatVector

__all__ = ["fixpoint_dem"]


def fixpoint_dem(
    g: Callable[[FloatVector], FloatVector],
    x0: FloatVector,
    *,
    q: float = 0.2,
    tolev: float = 0.01,
    tolx: float = 1e-6,
    sclx: FloatVector | None = None,
    maxiter: int = 50,
    callback: Callable[[int, FloatVector, FloatVector], tuple[bool, bool]] | None = None,
) -> VectorRootResult:
    r"""Find the solution of a N-dimensional fixed-point problem using the dominant
    eigenvalue method (DEM).

    The dominant eigenvalue method is an extrapolation technique that accelerates
    fixed-point iterations by estimating the dominant eigenvalue of the iteration
    implicitly. When this estimate stabilizes (according to `tolev`), a promotion
    step is applied to improve convergence. A damping factor `q` can be used to
    enhance robustness for unstable or strongly coupled problems.

    DEM is particularly effective for slowly converging problems (e.g., recycle
    systems), reducing iteration count at the cost of modest additional overhead.

    **References**

    *   Orbach, O.; Crowe, C. M. Convergence Promotion in the Simulation of Chemical
        Processes with Recycle—The Dominant Eigenvalue Method. Can. J. Chem. Eng. 1971,
        49, 509-513.

    Parameters
    ----------
    g : Callable[[FloatVector], FloatVector]
        Fixed-point mapping defining the problem `g(x) = x`.
    x0 : FloatVector
        Initial guess.
    q : float
        Damping parameter in [0, 1). Typically 0.0-0.5; higher values improve stability.
    tolev : float
        Relative tolerance for the dominant eigenvalue convergence. The promotion step is
        applied when the relative change in the dominant eigenvalue estimate is less than
        `tolev`. Decreasing this value delays the application of the promotion step.
    tolx : float
        Absolute tolerance for `x` value. The algorithm will terminate when
        `||sclx*(g(x) - x)||∞ <= tolx`.
    sclx : FloatVector | None
        Positive scaling factors for the components of `x`. Ideally, these should be
        chosen so that `sclx*x` is of order 1 near the solution for all components. By
        default, scaling is determined automatically from `x0`.
    maxiter : int
        Maximum number of iterations.
    callback : Callable[[int, FloatVector, FloatVector], tuple[bool, bool]] | None
        Optional callback with signature `callback(niter, x, fx)->(stop, success)` called
        at each iteration. If `stop` is `True`, the iteration is terminated. If `success`
        is `True`, the optimization is considered successful.

    Returns
    -------
    VectorRootResult
        Dataclass with root solution results.

    See Also
    --------
    * [`fixpoint_anderson`](fixpoint_anderson.md):
      Alternative method better suited for problems with coupling between components.
    * [`fixpoint_damped`](fixpoint_damped.md):
      Alternative method for problems with weak coupling between components.
    * [`fixpoint_wegstein`](fixpoint_wegstein.md):
      Alternative method for problems with weak coupling between components.

    Examples
    --------
    Find the solution of a 2D fixed-point function.
    >>> from polykin.math import fixpoint_dem
    >>> import numpy as np
    >>> def g(x):
    ...     x1, x2 = x
    ...     g1 = 0.5*np.cos(x1) + 0.1*x2 + 0.5
    ...     g2 = np.sin(x2) - 0.2*x1 + 1.2
    ...     return np.array([g1, g2])
    >>> sol = fixpoint_dem(g, x0=np.array([0.0, 0.0]))
    >>> print(f"x = {sol.x}")
    x = [0.97458605 1.93830731]
    >>> print(f"g(x) = {g(sol.x)}")
    g(x) = [0.97458605 1.93830731]
    """
    method = "DEM fixed-point"
    success = False
    message = ""
    nfeval = 0

    if not (0.0 <= q < 1.0):
        raise ValueError("`q` must satisfy 0 <= q < 1.")

    sclx = np.abs(sclx) if sclx is not None else scalex(x0)

    x = x0.copy()
    n = x.size

    fx = np.full(n, np.nan)
    Δx = np.full(n, np.nan)
    Δxm = np.full(n, np.nan)

    ev1 = ev1m = float("nan")
    nss = 0
    niter = 0

    for niter in range(1, maxiter + 1):
        gx = g(x)
        nfeval += 1
        fx = gx - x

        if callback is not None:
            stop, _success = callback(niter, x, fx)
            if stop:
                message = "Terminated by user callback."
                success = _success
                break

        if np.linalg.norm(sclx * fx, np.inf) <= tolx:
            message = "||sclx*(g(x) - x)||∞ ≤ tolx"
            success = True
            break

        if niter == maxiter:
            message = f"Maximum number of iterations ({maxiter}) reached."
            break

        if nss > 1:
            Δxm_norm = np.linalg.norm(Δxm)
            Δx_norm = np.linalg.norm(Δx)
            ev1_abs = Δx_norm / Δxm_norm if Δxm_norm > eps else 0.0
            ev1_sign = np.sign(np.dot(Δx, Δxm))
            ev1m = ev1
            ev1 = ev1_abs * ev1_sign

        if nss > 2 and abs(ev1 - ev1m) < tolev * max((abs(ev1m), abs(ev1))):
            denom = 1.0 - ev1
            if abs(denom) < 10 * eps:
                message = f"Nearly zero DEM denominator at x={x}, (1 - λ1)={denom:.2e})."
                break
            x += (1.0 - q) * fx / denom
            nss = 0
            ev1 = ev1m = float("nan")
        else:
            nss += 1
            xm = x
            Δxm = Δx
            x = gx
            Δx = x - xm

    return VectorRootResult(method, success, message, nfeval, None, niter, x, fx, None)
