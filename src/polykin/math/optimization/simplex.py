# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

from collections.abc import Callable

import numpy as np

from polykin.math.derivatives.ndiff import scalex
from polykin.math.optimization.results import VectorOptimumResult
from polykin.utils.typing import FloatMatrix, FloatVector, FloatVectorLike

__all__ = ["fmin_nelder_mead"]


def fmin_nelder_mead(
    f: Callable[[FloatVector], float],
    x0: FloatVectorLike,
    *,
    tolx: float = 1e-8,
    tolf: float = 1e-8,
    sclx: FloatVectorLike | None = None,
    maxiter: int | None = None,
    maxfeval: int | None = None,
    adaptive: bool = True,
    callback: Callable[[int, FloatMatrix, FloatVector], tuple[bool, bool]] | None = None,
) -> VectorOptimumResult:
    r"""Find the minimum of a multivariate function using the Nelder-Mead simplex
    algorithm.

    The Nelder-Mead simplex algorithm is a derivative-free optimization method for
    unconstrained minimization of multivariate functions. It maintains a simplex of `N+1`
    vertices in `N`-dimensional space and iteratively updates this simplex based on the
    function values at the vertices.

    **References**

    *   Nelder, J. A.; Mead, R. A Simplex Method for Function Minimization.
        Comput. J. 1965, 7, 308-313.
    *   Gao, F.; Han, L. Implementing the Nelder-Mead Simplex Algorithm with Adaptive
        Parameters. Comput. Optim. Appl. 2012, 51, 259-277.

    Parameters
    ----------
    f : Callable[[FloatVector (N)], float]
        Objective function to minimize.
    x0 : FloatVectorLike (N)
        Initial guess for the optimum. Moreover, if no user-defined scale `sclx` is
        provided, the scaling factors will be determined from this value.
    tolx : float
        Absolute tolerance for `x`. The algorithm terminates when the maximum scaled
        distance between the simplex vertices is less than `tolx`.
    tolf : float
        Absolute tolerance for `f`. The algorithm terminates when the maximum difference
        between the function values at the simplex vertices is less than `tolf`.
    sclx : FloatVectorLike (N) | None
        Positive scaling factors for the components of `x`. Ideally, these should be
        chosen so that `sclx*x` is of order 1 near the solution for all components. By
        default, scaling is determined from `x0`.
    maxiter : int | None
        Maximum number of iterations. By default, a value of `200*N` is used.
    maxfeval : int | None
        Maximum number of function evaluations. By default, a value of `200*N` is used.
    adaptive : bool
        Whether to use the adaptive parameter scheme proposed by Gao (2012). If `False`,
        the standard Nelder-Mead parameters are used.
    callback : Callable[[int, FloatMatrix (N+1, N), FloatVector (N+1)], tuple[bool, bool]] | None
        Optional callback with signature `callback(niter, x, fx)->(stop, success)` called
        at each iteration. If `stop` is `True`, the iteration is terminated. If `success`
        is `True`, the optimization is considered successful.

    Returns
    -------
    VectorOptimumResult
        Dataclass with the results of the optimization.

    Examples
    --------
    Find the minimum of the function `f(x) = (x[0] - 1e2)^2 + (x[1] - 1e10)^2`.
    >>> import numpy as np
    >>> from polykin.math import fmin_nelder_mead
    >>> fmin_nelder_mead(lambda x: (x[0] - 1e2)**2 + (x[1] - 1e10)**2, [1, 1e8])
     method: Nelder-Mead
    success: True
    message: Function value spread is less than `tolf`.
     nfeval: 226
      niter: 122
          x: [9.99999864e+01 1.00000000e+10]
          f: 1.9451564775785983e-09
    """  # noqa: E501
    method = "Nelder-Mead"
    message = ""
    success = False
    nfeval = 0

    x0 = np.asarray(x0, dtype=float)
    n = x0.size

    sclx = np.abs(np.asarray(sclx, dtype=float)) if sclx is not None else scalex(x0)

    maxiter = maxiter if maxiter is not None else 200 * n
    maxfeval = maxfeval if maxfeval is not None else 200 * n

    # Set the algorithm parameters
    if adaptive:
        # Dimension-dependent parameters from Gao (2012)
        a = 1.0
        b = 1.0 + 2.0 / n
        c = 0.75 - 1.0 / (2 * n)
        d = 1.0 - 1.0 / n
    else:
        a = 1.0
        b = 2.0
        c = 0.5
        d = 0.5

    # Initialize the simplex vertices
    x = np.empty((n + 1, n))
    x[0] = x0
    for k in range(n):
        x[k + 1, :] = x0
        step = 0.05 * max(abs(x0[k]), 1.0 / sclx[k])
        x[k + 1, k] += step

    # Evaluate the function at the simplex vertices
    fx = np.empty(n + 1)
    for k in range(n + 1):
        fx[k] = f(x[k, :])
    nfeval += n + 1

    # Main optimization loop
    niter = 0
    while True:
        niter += 1

        # Sort the complex points by their function values
        idx = np.argsort(fx)
        imin, imax2, imax = idx[0], idx[-2], idx[-1]
        xmin = x[imin]
        fmin = fx[imin]
        fmax2 = fx[imax2]
        fmax = fx[imax]

        if callback is not None:
            stop, _success = callback(niter, x, fx)
            if stop:
                success = _success
                message = "Terminated by user callback."
                break

        if np.max(np.abs(sclx * (x - xmin))) < tolx:
            success = True
            message = "Maximum distance between complex points is less than `tolx`."
            break

        if (fmax - fmin) < tolf:
            success = True
            message = "Function value spread is less than `tolf`."
            break

        if niter == maxiter:
            message = f"Maximum number of iterations ({maxiter}) reached."
            break

        if nfeval >= maxfeval:
            message = f"Maximum number of function evaluations ({maxfeval}) reached."
            break

        # Centroid excluding the worst point
        xc = (np.sum(x, axis=0) - x[imax]) / n

        # Reflection
        xr = (1.0 + a) * xc - a * x[imax]
        fr = f(xr)
        nfeval += 1

        shrink = False
        if fmin <= fr < fmax2:
            x[imax, :] = xr
            fx[imax] = fr
        elif fr < fmin:
            xe = (1.0 + a * b) * xc - a * b * x[imax]
            fe = f(xe)
            nfeval += 1
            if fe < fr:
                x[imax, :] = xe
                fx[imax] = fe
            else:
                x[imax, :] = xr
                fx[imax] = fr
        elif fmax2 <= fr < fmax:
            xoc = (1.0 + a * c) * xc - a * c * x[imax]
            foc = f(xoc)
            nfeval += 1
            if foc <= fr:
                x[imax, :] = xoc
                fx[imax] = foc
            else:
                shrink = True
        else:
            xic = (1.0 - a * c) * xc + a * c * x[imax]
            fic = f(xic)
            nfeval += 1
            if fic < fmax:
                x[imax, :] = xic
                fx[imax] = fic
            else:
                shrink = True

        if shrink:
            for k in range(n + 1):
                if k == imin:
                    continue
                xs = d * x[k, :] + (1.0 - d) * xmin
                x[k, :] = xs
                fx[k] = f(xs)
                nfeval += 1

    return VectorOptimumResult(
        method=method,
        success=success,
        message=message,
        nfeval=nfeval,
        ngeval=None,
        nheval=None,
        niter=niter,
        x=xmin,
        f=fmin,
    )
