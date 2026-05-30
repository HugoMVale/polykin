# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

import math
from collections.abc import Callable
from typing import Literal
from warnings import warn

import numpy as np
import scipy
from numpy.linalg import norm

from polykin.math.derivatives.ndiff import (
    gradient_forward,
    hessian_forward,
    jacobian_forward,
    scalex,
)
from polykin.math.machine import eps, sqrt_eps
from polykin.math.optimization.globalmethods import dogleg, line_search
from polykin.math.optimization.results import VectorOptimumResult
from polykin.utils.typing import FloatMatrix, FloatVector, FloatVectorLike

__all__ = ["fmin_qnewton"]


def fmin_qnewton(
    f: Callable[[FloatVector], float],
    x0: FloatVectorLike,
    *,
    tolx: float = 1e-10,
    tolg: float = 1e-5,
    sclx: FloatVectorLike | None = None,
    sclf: float = 1.0,
    maxiter: int = 100,
    maxlenfac: float = 1e3,
    trustlen: float | None = None,
    epsf: float | None = None,
    global_method: Literal["line-search", "dogleg"] | None = "dogleg",
    bfgs_update: bool | None = None,
    bfgs_method: Literal["factored", "unfactored", "inverse"] | None = None,
    grad: Callable[[FloatVector], FloatVector] | None = None,
    hess: Callable[[FloatVector], FloatMatrix] | None = None,
    H0: FloatMatrix | None = None,
    callback: Callable[[int, FloatVector, float, FloatVector], tuple[bool, bool]]
    | None = None,
    verbose: bool = False,
) -> VectorOptimumResult:
    r"""Find the minimum of a multivariate function using a quasi-Newton method with
    optional global strategies.

    Parameters
    ----------
    f : Callable[[FloatVector (N)], float]
        Objective function to minimize.
    x0 : FloatVectorLike (N)
        Initial guess for the optimum. Moreover, if no user-defined scale `sclx` is
        provided, the scaling factors will be determined from this value.
    tolx : float
        Tolerance for the scaled step size. The algorithm terminates when the scaled
        distance between two successive iterates `||Δx/max(x, 1/sclx)||∞` is below this
        threshold. If the value is too large, the algorithm may terminate prematurely.
        A value on the order of $\epsilon^{2/3}$ is typically recommended.
    tolg : float
        Tolerance for the scaled gradient. This is the main convergence criterion. The
        algorithm terminates when the scaled gradient `||∇f(x)*sclf/sclx||∞` is below this
        threshold. If the value is too large, the algorithm may terminate prematurely.
        A value on the order of $\epsilon^{1/3}$ is typically recommended.
    sclx : FloatVectorLike (N) | None
        Positive scaling factors for the components of `x`. Ideally, these should be
        chosen so that `sclx*x` is of order 1 near the solution for all components. By
        default, scaling is determined from `x0`.
    sclf : float
        Positive scaling factor for the function values. Ideally, this should be chosen so
        that `sclf*f(x)` is of order 1 across the domain of interest. The value is used to
        scale the gradient; if too low a value is provided, the algorithm may terminate
        prematurely.
    maxiter : int
        Maximum number of outer quasi-Newton iterations.
    maxlenfac : float
        Factor determining the maximum allowable scaled step length `||Δx*sclx||₂` for
        global methods. Used to prevent steps that would cause the algorithm to overflow,
        leave the domain of interest, or diverge. It should be chosen small enough to
        prevent such issues, but large enough to allow any anticipated reasonable step
        length.
    trustlen : float | None
        Initial trust region radius for the `dogleg` global method. By default, the length
        of the initial scaled gradient is used.
    epsf : float | None
        Machine precision of the function values. If `None`, machine precision of 64-bit
        floating-point type is assumed. If the number of reliable base-10 digits in the
        results returned by the function is $n$, then `epsf` is approximately $10^{-n}$.
    global_method : Literal['line-search','dogleg'] | None
        Global strategy to improve convergence from remote starting points. With
        `line-search`, the search direction is computed using the quasi-Newton step and
        the length of the step is determined by backtracking until the Armijo condition is
        fulfilled. With `dogleg`, a trust-region dogleg method is used to compute both the
        step direction and length. If `None`, no global strategy is used and the full
        quasi-Newton step is taken at each iteration.
    bfgs_update : bool | None
        Whether to update the Hessian approximation using the BFGS positive definite
        secant formula. If `False`, the Hessian is evaluated anew at each iteration
        using `hess` or forward finite differences. If `True`, the Hessian is updated
        using the BFGS formula, avoiding the cost of a full evaluation. The BFGS
        update typically increases the number of iterations required for convergence,
        but decreases the total number of function evaluations when the Hessian is
        approximated via finite differences. By default, the BFGS update is used if no
        Hessian function is provided, and is not used otherwise.
    bfgs_method : Literal['factored', 'unfactored', 'inverse'] | None
        Method to carry out the BFGS Hessian update. In theory, all methods produce the
        same result, but the computational cost and the numerical stability may differ.
        By default, the `factored` form is used when the dogleg global method is selected,
        because the dogleg method requires the Cholesky factor of the Hessian. Otherwise,
        the `inverse` form is used because it has the lowest algorithmic overhead.
    grad : Callable[[FloatVector (N)], FloatVector (N)] | None
        Function to compute the gradient of `f`. By default, the gradient is approximated
        using forward finite differences. In this case, setting `epsf` appropriately is
        essential.
    hess : Callable[[FloatVector (N)], FloatMatrix (N, N)] | None
        Function to compute the Hessian of `f`. By default, the Hessian is approximated
        using forward finite differences, using `grad` if available or `f` otherwise.
        In this case, setting `epsf` appropriately is essential.
    H0 : FloatMatrix (N, N) | None
        Initial Hessian approximation at `x0`, expected to be symmetric positive definite.
        By default, if `bfgs_update` is `False`, `H0` is computed using `hess` or forward
        finite differences. If `bfgs_update` is `True` and `hess` is not provided, `H0` is
        initialized to the identity matrix. If `bfgs_update` is `True` and `hess` is
        provided, `H0` is initialized from `hess`.
    callback : Callable[[int, FloatVector (N), float, FloatVector (N)], tuple[bool, bool]] | None
        Optional callback with signature `callback(niter, x, fx, ∇fx)->(stop, success)`
        called at each iteration. If `stop` is `True`, the iteration is terminated. If
        `success` is `True`, the optimization is considered successful.
    verbose : bool
        Print iteration information.

    Returns
    -------
    VectorOptimumResult
        Dataclass with the results of the optimization.

    Examples
    --------
    Find the minimum of the Rosenbrock function `f(x,y) = (1 - x)^2 + 100*(y - x^2)^2`.
    >>> import numpy as np
    >>> from polykin.math import fmin_qnewton
    >>> f = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    >>> fmin_qnewton(f, [-1.0, -1.0], global_method="dogleg")
     method: Quasi-Newton (Global: Dogleg, BFGS: Factored)
    success: True
    message: ||∇f(x)*sclf/sclx||∞ ≤ tolg
     nfeval: 98
     ngeval: 0
     nheval: 0
      niter: 28
          x: [0.9999964  0.99999277]
          f: 1.3039626656861953e-11
          g: [ 7.80959496e-06 -3.03099592e-06]
          L: [[-28.34545479   0.        ]
              [ 14.14251074  -0.70808768]]
    """  # noqa: E501
    # Check/set method options
    if bfgs_update is None:
        bfgs_update = hess is None

    if bfgs_update:
        if bfgs_method is None:
            if global_method == "dogleg":
                bfgs_method = "factored"
            else:
                bfgs_method = "inverse"
        elif bfgs_method != "factored" and global_method == "dogleg":
            bfgs_method = "factored"
            warn(
                "BFGS update with dogleg global method requires factored form. Setting `bfgs_method` to 'factored'."  # noqa: E501
            )

    # Construct method name for result
    method = "Quasi-Newton"
    method_options = []
    _global_method = global_method if global_method else "none"
    method_options.append(f"Global: {_global_method.title()}")
    _bfgs_method = bfgs_method if bfgs_method else "none"
    if bfgs_update:
        method_options.append(f"BFGS: {_bfgs_method.title()}")
    if method_options:
        method += " (" + ", ".join(method_options) + ")"

    # Initialize result variables
    success = False
    message = ""
    nfeval = 0
    ngeval = 0
    nheval = 0
    niter = 0

    x0 = np.asarray(x0, dtype=float)
    n = x0.size

    # Set scaling factors
    sclx = np.abs(np.asarray(sclx, dtype=float)) if sclx is not None else scalex(x0)
    sclf = max(abs(sclf), 1.0)

    # Set maximum step length for global methods
    maxlen = max(0.0, maxlenfac) * max(norm(sclx * x0).item(), norm(sclx).item())

    # Set initial trust region radius for dogleg method
    if trustlen is None:
        trustlen = -1.0  # Sentinel value
    else:
        trustlen = min(trustlen, maxlen)

    # Helper functions to evaluate gradient and Hessian
    def eval_grad(x: FloatVector, fx: float) -> FloatVector:
        nonlocal ngeval, nfeval
        if grad is not None:
            g = grad(x)
            ngeval += 1
        else:
            g = gradient_forward(f, x, fx=fx, sclx=sclx, epsf=epsf)
            nfeval += n
        return g

    def eval_hess(x: FloatVector, fx: float, gx: FloatVector) -> FloatMatrix:
        nonlocal nheval, ngeval, nfeval
        if hess is not None:
            H = hess(x)
            nheval += 1
        elif grad is not None:
            H = jacobian_forward(grad, x, fx=gx, sclx=sclx, epsf=epsf)
            ngeval += n
        else:
            H = hessian_forward(f, x, fx=fx, sclx=sclx, epsf=epsf)
            nfeval += n * (n + 3) // 2
        return H

    # Evaluate function at x0
    xc = x0.copy()
    fc = f(xc)
    nfeval += 1

    # Evaluate gradient at x0
    grad_analytic = grad is not None
    gc = eval_grad(xc, fc)

    # Check initial solution with tight tolerance
    if gradient_condition(xc, fc, gc, sclx, sclf, 1e-3 * tolg):
        message = "||∇f(x0)*sclf/sclx||∞ ≤ 1e-3*tolg"
        return VectorOptimumResult(
            method, True, message, nfeval, ngeval, nheval, niter, x0, fc, gc
        )

    # Evaluate Hessian at x0
    H = L = Hinv = np.array([])
    if H0 is not None:
        H = H0.copy()
    elif bfgs_update and hess is None:
        # D = max(abs(fc), 1 / sclf) * sclx**2
        D = np.ones_like(x0)
        if bfgs_method == "unfactored":
            H = np.zeros((n, n))
            np.fill_diagonal(H, D)
        elif bfgs_method == "factored":
            L = np.zeros((n, n))
            np.fill_diagonal(L, np.sqrt(D))
        elif bfgs_method == "inverse":
            Hinv = np.zeros((n, n))
            np.fill_diagonal(Hinv, 1 / D)
        else:
            raise ValueError(f"Unknown `bfgs_method`: {bfgs_method}.")
    else:
        H = eval_hess(xc, fc, gc)

    # Factorize or invert Hessian if BFGS update requires it
    if bfgs_update and H.size > 0:
        if bfgs_method == "unfactored" and hess is not None:
            try:
                H = make_hessian_spd(H, enforce_symmetry=True)[0]
            except np.linalg.LinAlgError as e:
                message = f"Cholesky factorization of initial Hessian failed: {e}."
                return VectorOptimumResult(
                    method, False, message, nfeval, ngeval, nheval, niter, xc, fc, gc, H
                )
        elif bfgs_method == "factored":
            try:
                L = make_hessian_spd(H, enforce_symmetry=True)[1]
            except np.linalg.LinAlgError as e:
                message = f"Cholesky factorization of initial Hessian failed: {e}."
                return VectorOptimumResult(
                    method, False, message, nfeval, ngeval, nheval, niter, xc, fc, gc, H
                )
        elif bfgs_method == "inverse":
            try:
                Hinv = np.linalg.inv(H)
            except np.linalg.LinAlgError as e:
                message = f"Inverse of initial Hessian failed: {e}."
                return VectorOptimumResult(
                    method, False, message, nfeval, ngeval, nheval, niter, xc, fc, gc, H
                )
        else:
            pass

    # Main optimization loop
    gm_nmaxsteps = 0
    for niter in range(1, maxiter + 1):
        if verbose:
            print(f"Iteration {niter:3d}:", flush=True)
            print("    Current point:", xc)
            print("    Function value:", fc)
            print("    Gradient value:", gc)

        # Ensure Hessian is SPD
        if not bfgs_update:
            try:
                H, L, _ = make_hessian_spd(H, enforce_symmetry=True)
            except np.linalg.LinAlgError as e:
                message = f"Perturbed Cholesky factorization of Hessian failed: {e}."
                break

        # Compute Newton step
        if bfgs_update and bfgs_method == "inverse":
            # p = - H⁻¹.gc
            p = Hinv @ gc
            p *= -1
        elif bfgs_update and bfgs_method == "unfactored":
            # H.p = - gc
            try:
                L, _ = scipy.linalg.cho_factor(
                    H,
                    lower=True,
                    overwrite_a=False,
                    check_finite=False,
                )
                p = scipy.linalg.cho_solve(
                    (L, True),
                    gc,
                    overwrite_b=False,
                    check_finite=False,
                )
                p *= -1
                if global_method == "dogleg":
                    L = np.tril(L)
            except scipy.linalg.LinAlgError as e:
                message = f"Cholesky solve for Newton step failed: {e}."
                break
        elif not bfgs_update or bfgs_method == "factored":
            # (L.Lᵀ).p = - gc
            try:
                # L.y = gc
                y = scipy.linalg.solve_triangular(
                    L,
                    gc,
                    lower=True,
                    overwrite_b=False,
                    check_finite=False,
                )
                # Lᵀ.p = y
                p = scipy.linalg.solve_triangular(
                    L,
                    y,
                    lower=True,
                    trans="T",
                    overwrite_b=True,
                    check_finite=False,
                )
                p *= -1
            except scipy.linalg.LinAlgError as e:
                message = f"Triangular solve for Newton step failed: {e}."
                break
        else:
            raise ValueError(f"Unknown `bfgs_method`: {bfgs_method}.")

        if verbose:
            print("    Search direction:", p)

        # Compute actual x step
        if global_method is None:
            xp = xc + p
            fp = f(xp)
            gm_ismaxstep = True
            gm_success = True
            gm_nfeval = 1
        elif global_method == "line-search":
            gm_success, gm_ismaxstep, gm_nfeval, xp, fp, _ = line_search(
                f, p, xc, fc, gc, tolx, sclx, maxlen, verbose
            )
        elif global_method == "dogleg":
            gm_success, gm_ismaxstep, gm_nfeval, trustlen, xp, fp, _ = dogleg(
                f, p, xc, fc, gc, L.T, tolx, sclx, maxlen, trustlen, verbose
            )
        else:
            raise ValueError(f"Unknown `global_method`: {global_method}.")

        nfeval += gm_nfeval
        gm_nmaxsteps = gm_nmaxsteps + 1 if gm_ismaxstep else 0

        # Display iteration progress
        if verbose:
            print(f"  x = {xp}\n  f(x) = {fp:.2e}", flush=True)

        # Evaluate gradient at new point
        gp = eval_grad(xp, fp)

        # Call user callback
        if callback is not None:
            stop, _success = callback(niter, xp, fp, gp)
            if stop:
                success = _success
                message = "Terminated by user callback."
                break

        # Check termination and convergence conditions
        if not gm_success:
            message = """Last global step failed to decrease f(x) sufficiently.
            Either `x` is an approximate local minimizer and no more accuracy is possible,
            or the finite difference gradient approximation is too inaccurate, or `tolx`
            is too large."""
            stop = True
        elif gradient_condition(xp, fp, gp, sclx, sclf, tolg):
            message = "||∇f(x)*sclf/sclx||∞ ≤ tolg"
            success = True
            stop = True
        elif norm((xp - xc) / np.maximum(np.abs(xp), 1 / sclx), np.inf) <= tolx:
            message = """||Δx/max(x, 1/sclx)||∞ ≤ tolx
            `x` may be an approximate local minimizer, but it is also possible that the
            algorithm is making slow progress and is not near a minimizer, or that `tolx`
            is too large."""
            stop = True
        elif global_method and gm_nmaxsteps >= 5:
            message = """Maximum number (5) of consecutive steps of length `maxlen`
            reached. Perhaps f(x) is unbounded below, or f(x) has a finite asymptote in
            some direction, or `maxlen` is too small."""
            stop = True
        else:
            stop = False

        if stop:
            xc, fc, gc = xp, fp, gp
            break

        # Update Hessian approximation
        if bfgs_update:
            if bfgs_method == "factored":
                L = _update_bfgs_factored(xc, xp, gc, gp, L, grad_analytic)
            elif bfgs_method == "unfactored":
                H = _update_bfgs_unfactored(xc, xp, gc, gp, H, grad_analytic)
            elif bfgs_method == "inverse":
                Hinv = _update_bfgs_inverse(xc, xp, gc, gp, Hinv, grad_analytic)
            else:
                raise ValueError(f"Unknown `bfgs_method`: {bfgs_method}.")
        else:
            H = eval_hess(xp, fp, gp)

        # Next iteration will start at xp
        xc, fc, gc = xp, fp, gp

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    return VectorOptimumResult(
        method=method,
        success=success,
        message=message,
        nfeval=nfeval,
        ngeval=ngeval,
        nheval=nheval,
        niter=niter,
        x=xc,
        f=fc,
        g=gc,
        H=H if not bfgs_update or bfgs_method == "factored" else None,
        Hinv=Hinv if bfgs_method == "inverse" else None,
        L=L if bfgs_method == "factored" else None,
    )


def gradient_condition(
    xc: FloatVector,
    fc: float,
    gc: FloatVector,
    sclx: FloatVector,
    sclf: float,
    tolg: float,
) -> bool:
    """Check if function gradient satisfies the tolerance condition."""
    sclg = np.maximum(np.abs(xc), 1 / sclx) / max(abs(fc), 1 / sclf)
    return bool(np.max(np.abs(gc) * sclg) <= tolg)


def _update_bfgs_unfactored(
    xc: FloatVector,
    xp: FloatVector,
    gc: FloatVector,
    gp: FloatVector,
    Hc: FloatMatrix,
    grad_analytic: bool,
) -> FloatMatrix:
    r"""Update the Hessian approximation using the BFGS positive definite secant formula.

    $$ H \gets H - \frac{H s s^T H}{s^T H s} + \frac{y y^T}{y^T s} $$

    where $s = x_p - x_c$ and $y = g(x_p) - g(x_c)$.

    **References**

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
        Optimization and Nonlinear Equations", SIAM, 1996, p. 355.

    Parameters
    ----------
    xc : FloatVector
        Current value of the variable vector.
    xp : FloatVector
        Next value of the variable vector.
    gc : FloatVector
        Current gradient vector, `g(xc)`.
    gp : FloatVector
        Next gradient vector, `g(xp)`.
    Hc : FloatMatrix
        Current Hessian approximation. This will be updated in-place.
    grad_analytic : bool
        Whether the gradient is analytic or not.

    Returns
    -------
    FloatMatrix
        Updated Hessian approximation.
    """
    s = xp - xc
    y = gp - gc

    # Check curvature condition: yᵀ.s > 0
    ys = np.dot(y, s)
    if ys < sqrt_eps * norm(s) * norm(y):
        return Hc

    # Check if the update is small enough to be ignored
    tol = eps if grad_analytic else sqrt_eps
    Hs = Hc @ s
    if np.all(np.abs(y - Hs) < tol * (np.abs(gc) + np.abs(gp))):
        return Hc

    sHs = np.dot(s, Hs)
    Hc += np.outer(y, y) / ys
    Hc -= np.outer(Hs, Hs) / sHs

    Hc += Hc.T
    Hc /= 2

    return Hc


def _update_bfgs_factored(
    xc: FloatVector,
    xp: FloatVector,
    gc: FloatVector,
    gp: FloatVector,
    Lc: FloatMatrix,
    grad_analytic: bool,
) -> FloatMatrix:
    r"""Update the Cholesky factor of the Hessian approximation using the BFGS positive
    definite secant formula.

    $$ J = L + \frac{(y - \alpha L L^T s)(\alpha L^T s)^T}{y^T s} $$

    $$ \alpha = \left(\frac{y^T s}{s^T L L^T s}\right)^{1/2} $$

    where $s = x_p - x_c$ and $y = g(x_p) - g(x_c)$.

    **References**

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
        Optimization and Nonlinear Equations", SIAM, 1996, p. 356.

    Parameters
    ----------
    xc : FloatVector
        Current value of the variable vector.
    xp : FloatVector
        Next value of the variable vector.
    gc : FloatVector
        Current gradient vector, `g(xc)`.
    gp : FloatVector
        Next gradient vector, `g(xp)`.
    Lc : FloatMatrix
        Current lower-triangular Cholesky factor of Hessian approximation. This will be
        overwritten.
    grad_analytic : bool
        Whether the gradient is analytic or not.

    Returns
    -------
    FloatMatrix
        Updated lower-triangular Cholesky factor of Hessian approximation.
    """
    s = xp - xc
    y = gp - gc

    ys = np.dot(y, s)
    if ys < sqrt_eps * norm(s) * norm(y):
        return Lc

    LTs = Lc.T @ s
    sHs = np.dot(LTs, LTs)

    alpha = math.sqrt(ys / sHs)  # α = √(yᵀ.s / sᵀ.L.Lᵀ.s)

    Hs = Lc @ LTs  # L.Lᵀ.s = H.s

    tol = eps if grad_analytic else sqrt_eps
    if np.all(np.abs(y - Hs) < tol * (np.abs(gc) + np.abs(gp))):
        return Lc

    # u = y - α.L.Lᵀ.s
    Hs *= -alpha
    Hs += y

    # t = Lᵀ.s / √(yᵀ.s * sᵀ.L.Lᵀ.s)
    LTs /= math.sqrt(ys * sHs)

    # QR rank-1 update
    n = Lc.shape[0]
    _, Rp = scipy.linalg.qr_update(np.eye(n), Lc.T, LTs, Hs, overwrite_qruv=True)
    Lp = Rp.T

    return Lp


def _update_bfgs_inverse(
    xc: FloatVector,
    xp: FloatVector,
    gc: FloatVector,
    gp: FloatVector,
    Bc: FloatMatrix,
    grad_analytic: bool,
) -> FloatMatrix:
    r"""Update the inverse Hessian approximation using the expanded Sherman-Morrison
    formula.

    $$ B \gets B + \frac{(s^T y + y^T B y)(s s^T)}{(s^T y)^2}
                 - \frac{B y s^T + s y^T B}{s^T y} $$

    where $s = x_p - x_c$ and $y = g(x_p) - g(x_c)$.

    Parameters
    ----------
    xc : FloatVector
        Current value of the variable vector.
    xp : FloatVector
        Next value of the variable vector.
    gc : FloatVector
        Current gradient vector, `g(xc)`.
    gp : FloatVector
        Next gradient vector, `g(xp)`.
    Bc : FloatMatrix
        Current inverse Hessian approximation (B = H⁻¹). This will be updated in-place.
    grad_analytic : bool
        Whether the gradient is analytic or not.

    Returns
    -------
    FloatMatrix
        Updated inverse Hessian approximation.
    """
    s = xp - xc
    y = gp - gc

    # Check curvature condition: yᵀ.s > 0
    ys = np.dot(y, s)
    if ys < sqrt_eps * norm(s) * norm(y):
        return Bc

    # Check if the update is small enough to be ignored
    By = Bc @ y
    tol = eps if grad_analytic else sqrt_eps
    scale = np.maximum.reduce([np.ones_like(s), np.abs(s), np.abs(By)])
    if np.all(np.abs(s - By) < tol * scale):
        return Bc

    yBy = np.dot(y, By)
    Bc += ((ys + yBy) * np.outer(s, s)) / (ys * ys)
    Bc -= (np.outer(By, s) + np.outer(s, By)) / ys

    Bc += Bc.T
    Bc /= 2

    return Bc


def make_hessian_spd(
    H: FloatMatrix,
    enforce_symmetry: bool = True,
    maxiter: int = 10,
) -> tuple[FloatMatrix, FloatMatrix, float]:
    """Make a Hessian matrix positive definite by adding a scaled identity matrix.

    Parameters
    ----------
    H : FloatMatrix
        Hessian matrix to be made positive definite.
    enforce_symmetry : bool
        Whether to enforce symmetry of the Hessian by averaging it with its transpose.
    maxiter : int
        Maximum number of perturbation iterations to attempt. In each iteration, the
        perturbation is increased by a factor of 10 until the Hessian becomes positive
        definite.

    Returns
    -------
    tuple[FloatMatrix, FloatMatrix, float]
        Tuple containing the perturbed Hessian matrix, its Cholesky factor, and the
        perturbation value.
    """
    H = H.copy()
    if enforce_symmetry:
        H += H.T
        H *= 0.5

    # Ensure all diagonal elements are positive and sufficiently large compared to the
    # off-diagonal elements, to improve the chances of the Hessian being SPD
    maxdiag = H.diagonal().max()
    mindiag = H.diagonal().min()
    maxposdiag = max(maxdiag, 0.0)

    if mindiag <= sqrt_eps * maxposdiag:
        mu = 2 * (maxposdiag - mindiag) * sqrt_eps - mindiag
        maxdiag += mu
    else:
        mu = 0.0

    Ht = np.abs(H)
    diag = np.diag_indices_from(H)
    Ht[diag] = 0.0
    maxoff = Ht.max()

    if maxoff * (1.0 + 2 * sqrt_eps) > maxdiag:
        mu += (maxoff - maxdiag) + 2 * sqrt_eps * maxoff
        maxdiag = maxoff * (1.0 + 2 * sqrt_eps)

    if maxdiag == 0.0:
        mu = 1.0
        maxdiag = 1.0

    if mu > 0.0:
        H[diag] += mu

    # Perturb Hessian by adding a scaled identity matrix until it becomes SPD
    Hp = np.empty_like(H)
    perturb = 0.0
    perturb0 = eps * max(float(norm(H, np.inf)), 1.0)

    for _ in range(maxiter + 1):
        Hp[:] = H
        Hp[diag] += perturb

        try:
            Lp = np.linalg.cholesky(Hp)
            return Hp, Lp, perturb + mu
        except np.linalg.LinAlgError:
            perturb = max(perturb0, 10 * perturb)

    raise np.linalg.LinAlgError(
        f"Perturbed Cholesky failed after total perturbation={mu + perturb:.3e}"
    )
