# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from typing import Callable

import numpy as np

from polykin.math.roots import VectorRootResult
from polykin.utils.math import eps
from polykin.utils.types import FloatVector

__all__ = [
    'fixpoint_anderson',
    'fixpoint_wegstein'
]


def fixpoint_anderson(
        g: Callable[[FloatVector], FloatVector],
        x0: FloatVector,
        m: int = 3,
        xtol: float = 1e-6,
        maxiter: int = 50
) -> VectorRootResult:
    r"""Find the solution of a N-dimensional fixed-point problem using the
    Anderson acceleration method.

    The Anderson acceleration method is an extrapolation technique to
    accelerate the convergence of multidimentional fixed-point iterations. 
    It uses information from $m$ previous iterations to construct a better 
    approximation of the fixed point according to the formula:

    $$ x_{k+1} = g(x_k) - \sum_{i=0}^{m_k-1} 
       \gamma_i^{(k)} \left[ g(x_{k-m_k+i+1}) - g(x_{k-m_k+i}) \right] $$

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
    xtol : float
        Absolute tolerance for `x` value. The algorithm will terminate when
        `||g(x_k) - x_k||∞ <= xtol`.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    VectorRootResult
        Dataclass with root solution results.

    See also
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
    >>> print(f"x={sol.x}")
    x=[0.97458605 1.93830731]
    >>> print(f"g(x)={g(sol.x)}")
    g(x)=[0.97458605 1.93830731]
    """

    nfeval = 0
    message = ""
    success = False

    # Different ordering of arrays to optimize memory access
    ΔG = np.zeros((m, x0.size))
    ΔF = np.zeros((x0.size, m))

    g0 = g(x0)
    nfeval += 1
    f0 = g0 - x0

    if np.linalg.norm(f0, np.inf) <= xtol:
        message = "||g(x0) - x0||∞ <= xtol"
        success = True
        return VectorRootResult(success, message, nfeval, 0, x0, f0)

    x = g0
    gx = g0
    fx = f0

    m = max(m, 1)

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

        # There are methods to reuse/update the QR decomposition from the
        # previous iteration, but this is more complex to implement.
        try:
            Q, R = np.linalg.qr(ΔF[:, -mk:])
            gamma = np.linalg.lstsq(R, Q.T @ fx, rcond=None)[0]
        except np.linalg.LinAlgError:
            message = "Error in QR decomposition or least-squares solution."
            break

        if k + 1 < maxiter:
            x = gx - np.dot(gamma, ΔG[-mk:, :])

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return VectorRootResult(success, message, nfeval, k+1, x, fx)


def fixpoint_wegstein(
    g: Callable[[FloatVector], FloatVector],
    x0: FloatVector,
    xtol: float = 1e-6,
    kwait: int = 1,
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
    xtol : float
        Absolute tolerance for `x` value. The algorithm will terminate when
        `||g(x_k) - x_k||∞ <= xtol`.
    kwait : int
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

    See also
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
    >>> print(f"x={sol.x}")
    x=[0.97458605 1.93830731]
    >>> print(f"g(x)={g(sol.x)}")
    g(x)=[0.97458605 1.93830731]
    """

    nfeval = 0
    message = ""
    success = False

    x = x0.copy()
    gx = np.full(x.size, np.nan)
    xp = np.full(x.size, np.nan)

    kwait = max(kwait, 1)

    for k in range(maxiter):

        gxp = gx
        gx = g(x)
        nfeval += 1
        fx = gx - x

        if np.linalg.norm(fx, np.inf) <= xtol:
            message = "||g(x_k) - x_k||∞ <= xtol"
            success = True
            break

        if k + 1 < maxiter:
            if k < kwait:
                xp = x
                x = gx
            else:
                Δx = x - xp
                Δg = gx - gxp
                s = np.zeros_like(Δg)
                mask_s = np.abs(Δx) > eps
                np.divide(Δg, Δx, out=s, where=mask_s)
                q = s / (s - 1)
                q = np.clip(q, qmin, qmax)
                xp = x
                x = q*x + (1 - q)*gx

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return VectorRootResult(success, message, nfeval, k+1, x, fx)
