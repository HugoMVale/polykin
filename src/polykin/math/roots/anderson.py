# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from dataclasses import dataclass
from typing import Callable

import numpy as np

from polykin.utils.types import FloatVector

__all__ = [
    'fixpoint_anderson',
    'VectorRootResult'
]


@dataclass
class VectorRootResult():
    """Dataclass with vector root solution results.

    Attributes
    ----------
    success: bool
        If `True`, the root was found.
    message: str
        Description of the exit status.
    nfeval: int
        Number of function evaluations.
    niter: int
        Number of iterations.
    x: FloatVector
        Root value.
    f: FloatVector
        Function (residual) value at root.
    """
    success: bool
    message: str
    nfeval: int
    niter: int
    x: FloatVector
    f: FloatVector


def fixpoint_anderson(
        g: Callable,
        x0: FloatVector,
        m: int = 3,
        xtol: float = 1e-6,
        maxiter: int = 50
) -> VectorRootResult:
    r"""Find the solution of a n-dimensional fixed-point function using Anderson
    acceleration.

    Parameters
    ----------
    g : Callable
        Identity function for the fixed-point problem, i.e. `g(x) = x`.
    x0 : FloatVector
        Initial guess.
    m : int, optional
        Number of previous steps (`m >= 1`) to use in the acceleration.
    xtol : float, optional
        Absolute tolerance for `x` value. The algorithm will terminate when
        `||g(x_k) - x_k||∞ <= xtol`.
    maxiter : int, optional
        Maximum number of iterations.

    Returns
    -------
    VectorRootResult
        Dataclass with root solution results.

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
    >>> print(f"x={sol.x}")
    x=[0.97458605 1.93830731]
    >>> print(f"g(x)={g(sol.x)}")
    g(x)=[0.97458605 1.93830731]
    """

    nfeval = 0
    message = ""

    # Different ordering of arrays to optimize memory access
    ΔG = np.zeros((m, x0.size))
    ΔF = np.zeros((x0.size, m))

    g0 = g(x0)
    nfeval += 1
    f0 = g0 - x0

    if np.linalg.norm(f0, np.inf) <= xtol:
        message = "||g(x0) - x0||∞ <= xtol"
        return VectorRootResult(True, message, nfeval, 0, x0, f0)

    x = g0
    gx = g0
    fx = f0
    success = False
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

        if np.linalg.norm(fx, np.inf) <= xtol:
            message = "||g(x_k) - x_k||∞ <= xtol"
            success = True
            break

        # There are methods to reuse the QR decomposition from the previous
        # iteration, but this is more complex to implement.
        try:
            Q, R = np.linalg.qr(ΔF[:, -mk:])
            gamma = np.linalg.lstsq(R, Q.T @ fx, rcond=None)[0]
        except np.linalg.LinAlgError:
            message = "Error in QR decomposition or least-squares solution."
            break

        if k < maxiter:
            x = gx - np.dot(gamma, ΔG[-mk:, :])

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return VectorRootResult(success, message, nfeval, k+1, x, fx)
