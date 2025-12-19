# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from collections.abc import Callable

import numpy as np
import scipy

from polykin.math import scalex
from polykin.math.roots import VectorRootResult
from polykin.utils.types import FloatVector

__all__ = [
    "fixpoint_anderson",
]


def fixpoint_anderson(
    g: Callable[[FloatVector], FloatVector],
    x0: FloatVector,
    *,
    m: int = 3,
    tolx: float = 1e-6,
    sclx: FloatVector | None = None,
    maxiter: int = 50,
) -> VectorRootResult:
    r"""Find the solution of a N-dimensional fixed-point problem using the
    Anderson acceleration method.

    The Anderson acceleration method is an extrapolation technique to
    accelerate the convergence of multidimentional fixed-point iterations.
    It uses information from $m$ previous iterations to construct a better
    approximation of the fixed point according to the formula:

    $$ \mathbf{x}_{k+1} = \mathbf{g}(\mathbf{x}_k) - \sum_{i=0}^{m_k-1}
       \gamma_i^{(k)} \left[ \mathbf{g}(\mathbf{x}_{k-m_k+i+1}) -
       \mathbf{g}(\mathbf{x}_{k-m_k+i}) \right] $$

    where $m_k=\min(m,k)$, and the coefficients $\gamma_i^{(k)}$ are determined
    at each step by solving a least-squares problem.

    **References**

    * D.G. Anderson, "Iterative Procedures for Nonlinear Integral Equations",
      Journal of the ACM, 12(4), 1965, pp. 547-560.
    * H.F. Walker, "Anderson Acceleration: Algorithms and Implementations",
      Worcester Polytechnic Institute, Report MS-6-15-50, 2011.

    Parameters
    ----------
    g : Callable[[FloatVector], FloatVector]
        Identity function for the fixed-point problem, i.e. `g(x) = x`.
    x0 : FloatVector
        Initial guess.
    m : int
        Number of previous steps (`m >= 1`) to use in the acceleration.
    tolx : float
        Absolute tolerance for `x` value. The algorithm will terminate when
        `||sclx*(g(x) - x)||∞ <= tolx`.
    sclx : FloatVector | None
        Positive scaling factors for the components of `x`. Ideally, these
        should be chosen so that `sclx*x` is of order 1 near the solution for
        all components. By default, scaling is determined automatically from `x0`.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    VectorRootResult
        Dataclass with root solution results.

    See Also
    --------
    * [`fixpoint_wegstein`](fixpoint_wegstein.md): alternative (simpler) method
      for problems with weak coupling between components.

    Examples
    --------
    Find the solution of a 2D fixed-point function.
    >>> from polykin.math import fixpoint_anderson
    >>> import numpy as np
    >>> def g(x):
    ...     x1, x2 = x
    ...     g1 = 0.5*np.cos(x1) + 0.1*x2 + 0.5
    ...     g2 = np.sin(x2) - 0.2*x1 + 1.2
    ...     return np.array([g1, g2])
    >>> sol = fixpoint_anderson(g, x0=np.array([0.0, 0.0]))
    >>> print(f"x = {sol.x}")
    x = [0.97458605 1.93830731]
    >>> print(f"g(x)={g(sol.x)}")
    g(x) = [0.97458605 1.93830731]
    """
    method = "Anderson fix-point"
    success = False
    message = ""
    nfeval = 0

    sclx = sclx if sclx is not None else scalex(x0)

    # Different ordering of arrays to optimize memory access
    n = x0.size
    m = max(m, 1)
    ΔG = np.zeros((m, n))
    ΔF = np.zeros((n, m))

    g0 = g(x0)
    nfeval += 1
    f0 = g0 - x0

    if np.linalg.norm(sclx * f0, np.inf) <= 1e-2 * tolx:
        message = "||sclx*(g(x0) - x0)||∞ ≤ 1e-2*tolx"
        return VectorRootResult(method, True, message, nfeval, None, 0, x0, f0, None)

    x = g0
    gx = g0
    fx = f0

    for k in range(1, maxiter):

        mk = min(m, k)

        ΔG[:-1, :] = ΔG[1:, :]
        ΔF[:, :-1] = ΔF[:, 1:]

        ΔG[-1, :] = -gx
        ΔF[:, -1] = -fx

        gx = g(x)
        nfeval += 1
        fx = gx - x

        ΔG[-1, :] += gx
        ΔF[:, -1] += fx

        if np.linalg.norm(sclx * fx, np.inf) <= tolx:
            message = "||sclx*(g(x) - x)||∞ ≤ tolx"
            success = True
            break

        try:

            if k == 1:
                Q, R = scipy.linalg.qr(ΔF[:, -mk:], mode="economic")
            if k > m:
                Q, R = scipy.linalg.qr_delete(Q, R, 0, which="col")
            if k > 1:
                Q, R = scipy.linalg.qr_insert(Q, R, ΔF[:, -1], mk - 1, which="col")

        except scipy.linalg.LinAlgError:
            message = "Error in QR factorization/update."
            break

        try:
            gamma = np.linalg.lstsq(R, Q.T @ fx)[0]
        except np.linalg.LinAlgError:
            message = "Error in least-squares solution."
            break

        if k + 1 < maxiter:
            x = gx - np.dot(gamma, ΔG[-mk:, :])

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return VectorRootResult(method, success, message, nfeval, None, k + 1, x, fx, None)
