# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from collections.abc import Callable

import numpy as np

from polykin.math import scalex
from polykin.math.roots import VectorRootResult
from polykin.utils.math import eps
from polykin.utils.types import FloatVector

__all__ = ["fixpoint_wegstein"]


def fixpoint_wegstein(
    g: Callable[[FloatVector], FloatVector],
    x0: FloatVector,
    *,
    tolx: float = 1e-6,
    sclx: FloatVector | None = None,
    wait: int = 1,
    qmin: float = -5.0,
    qmax: float = 0.0,
    maxiter: int = 50,
) -> VectorRootResult:
    r"""Find the solution of a N-dimensional fixed-point problem using the
    bounded Wegstein acceleration method.

    The bounded Wegstein acceleration method is an extrapolation technique to
    accelerate the convergence of fixed-point iterations. For N-dimensional
    problems, each component of the fixed-point vector is treated separately
    according to:

    $$ x_{k+1} = q_k x_k + (1 - q_k) g(x_k) $$

    where $q_{min} \leq q_k \leq q_{max}$ is the acceleration parameter
    determined by:

    \begin{aligned}
    q_k &= \frac{s_k}{s_k - 1} \\
    s_k &= \frac{g(x_k) - g(x_{k-1})}{x_k - x_{k-1}}
    \end{aligned}

    When $q=0$, the Wegstein method is equivalent to the standard fixed-point
    iteration. When $q<0$, the convergence is accelerated, and when $0<q<1$ the
    convergence is dampened.

    **References**

    * J.H. Wegstein, "Accelerating convergence of iterative processes",
      Communications of the ACM, 1(6): 9-13, 1958.

    Parameters
    ----------
    g : Callable[[FloatVector], FloatVector]
        Identity function for the fixed-point problem, i.e. `g(x) = x`.
    x0 : FloatVector
        Initial guess.
    tolx : float
        Absolute tolerance for `x` value. The algorithm will terminate when
        `||sclx*(g(x) - x)||∞ <= tolx`.
    sclx : FloatVector | None
        Positive scaling factors for the components of `x`. Ideally, these
        should be chosen so that `sclx*x` is of order 1 near the solution for
        all components. By default, scaling is determined automatically from `x0`.
    wait : int
        Number of direct substitution iterations before the first acceleration
        iteration.
    qmin : float
        Minimum value for the acceleration parameter.
    qmax : float, optional
        Maximum value for the acceleration parameter.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    VectorRootResult
        Dataclass with root solution results.

    See Also
    --------
    * [`fixpoint_anderson`](fixpoint_anderson.md): alternative method better
      suited for problems with coupling between components.

    Examples
    --------
    Find the solution of a 2D fixed-point function.
    >>> from polykin.math import fixpoint_wegstein
    >>> import numpy as np
    >>> def g(x):
    ...     x1, x2 = x
    ...     g1 = 0.5*np.cos(x1) + 0.1*x2 + 0.5
    ...     g2 = np.sin(x2) - 0.2*x1 + 1.2
    ...     return np.array([g1, g2])
    >>> sol = fixpoint_wegstein(g, x0=np.array([0.0, 0.0]), qmax=0.5)
    >>> print(f"x = {sol.x}")
    x = [0.97458605 1.93830731]
    >>> print(f"g(x) = {g(sol.x)}")
    g(x) = [0.97458605 1.93830731]
    """
    method = "Wegstein fixed-point"
    success = False
    message = ""
    nfeval = 0

    sclx = sclx if sclx is not None else scalex(x0)

    x = x0.copy()
    n = x.size
    gx = np.full(n, np.nan)
    xm = np.full(n, np.nan)

    wait = max(wait, 1)

    for k in range(maxiter):

        gxm = gx
        gx = g(x)
        nfeval += 1
        fx = gx - x

        if np.linalg.norm(sclx * fx, np.inf) <= tolx:
            message = "||sclx*(g(x) - x)||∞ ≤ tolx"
            success = True
            break

        if k + 1 < maxiter:
            if k < wait:
                xm = x
                x = gx
            else:
                Δx = x - xm
                Δg = gx - gxm
                s = np.zeros(n)
                mask_s = np.abs(Δx) > eps
                np.divide(Δg, Δx, out=s, where=mask_s)
                q = s / (s - 1)
                q = np.clip(q, qmin, qmax)
                xm = x
                x = q * x + (1 - q) * gx

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return VectorRootResult(method, success, message, nfeval, None, k + 1, x, fx, None)
