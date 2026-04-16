# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

from collections.abc import Callable

import numpy as np

from polykin.math import scalex
from polykin.math.roots import VectorRootResult
from polykin.utils.typing import FloatVector

__all__ = ["fixpoint_damped"]


def fixpoint_damped(
    g: Callable[[FloatVector], FloatVector],
    x0: FloatVector,
    *,
    q: float = 0.2,
    tolx: float = 1e-6,
    sclx: FloatVector | None = None,
    maxiter: int = 50,
    callback: Callable[[int, FloatVector, FloatVector], bool] | None = None,
) -> VectorRootResult:
    r"""Find the solution of a N-dimensional fixed-point problem using direct substitution
    with damping.

    Direct substitution with damping is a fixed-point iteration where the next iterate is
    obtained from a convex combination of the current iterate and its direct substitution
    update according to:

    $$ \mathbf{x}_{k+1} =
       \mathbf{x}_k  + (1 - q) \left( \mathbf{g}(\mathbf{x}_k) - \mathbf{x}_k \right) $$

    where $0 \leq q < 1$ is the damping parameter. When $q=0$, the method is equivalent
    to standard direct substitution. For $q>0$, the update is damped, which can improve
    robustness for mildly unstable problems.

    Parameters
    ----------
    g : Callable[[FloatVector], FloatVector]
        Fixed-point mapping defining the problem `g(x) = x`.
    x0 : FloatVector
        Initial guess.
    q : float
        Damping parameter in [0, 1). Typically 0.0–0.5; higher values improve stability.
    tolx : float
        Absolute tolerance for `x` value. The algorithm will terminate when
        `||sclx*(g(x) - x)||∞ <= tolx`.
    sclx : FloatVector | None
        Positive scaling factors for the components of `x`. Ideally, these should be
        chosen so that `sclx*x` is of order 1 near the solution for all components. By
        default, scaling is determined automatically from `x0`.
    maxiter : int
        Maximum number of iterations.
    callback : Callable[[int, FloatVector, FloatVector], bool] | None
        Optional callback with signature `callback(niter, x, fx)` called at the end of
        each iteration to carry out custom actions, e.g., logging. Moreover, if the
        function returns `True`, the iteration will terminate early.

    Returns
    -------
    VectorRootResult
        Dataclass with root solution results.

    See Also
    --------
    * [`fixpoint_anderson`](fixpoint_anderson.md):
      Acceleration method suited for problems with coupling between components.
    * [`fixpoint_wegstein`](fixpoint_wegstein.md):
      Extrapolation method for weakly coupled fixed-point problems.

    Examples
    --------
    Find the solution of a 2D fixed-point function.
    >>> from polykin.math import fixpoint_damped
    >>> import numpy as np
    >>> def g(x):
    ...     x1, x2 = x
    ...     g1 = 0.5*np.cos(x1) + 0.1*x2 + 0.5
    ...     g2 = np.sin(x2) - 0.2*x1 + 1.2
    ...     return np.array([g1, g2])
    >>> sol = fixpoint_damped(g, x0=np.array([0.0, 0.0]), q=0.2)
    >>> print(f"x = {sol.x}")
    x = [0.97458614 1.93830761]
    >>> print(f"g(x) = {g(sol.x)}")
    g(x) = [0.97458604 1.93830719]
    """
    method = "Damped fixed-point"
    success = False
    message = ""
    nfeval = 0

    if not (0.0 <= q < 1.0):
        raise ValueError("`q` must satisfy 0 <= q < 1.")

    sclx = sclx if sclx is not None else scalex(x0)

    x = x0.copy()
    fx = np.full_like(x, np.nan)
    niter = 0

    for niter in range(1, maxiter + 1):
        gx = g(x)
        nfeval += 1
        fx = gx - x

        if callback is not None and callback(niter, x, fx):
            message = "Terminated by user callback."
            success = True
            break

        if np.linalg.norm(sclx * fx, np.inf) <= tolx:
            message = "||sclx*(g(x) - x)||∞ ≤ tolx"
            success = True
            break

        if niter < maxiter:
            x += (1 - q) * fx

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return VectorRootResult(method, success, message, nfeval, None, niter, x, fx, None)
