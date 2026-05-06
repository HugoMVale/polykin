# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import scipy
from numpy.linalg import norm

from polykin.math.derivatives import jacobian_forward, scalex
from polykin.math.machine import eps, sqrt_eps
from polykin.math.optimization.globalmethods import dogleg, line_search
from polykin.math.roots.results import VectorRootResult
from polykin.utils.typing import FloatMatrix, FloatVector, FloatVectorLike

all = ["rootvec_qnewton"]


def rootvec_qnewton(
    f: Callable[[FloatVector], FloatVector],
    x0: FloatVectorLike,
    *,
    tolx: float = 1e-10,
    tolf: float = 1e-5,
    sclx: FloatVectorLike | None = None,
    sclf: FloatVectorLike | None = None,
    maxiter: int = 100,
    maxlenfac: float = 1e3,
    trustlen: float | None = None,
    epsf: float | None = None,
    global_method: Literal["line-search", "dogleg"] | None = "line-search",
    broyden_update: bool = False,
    jac: Callable[[FloatVector], FloatMatrix] | None = None,
    jac0: FloatMatrix | None = None,
    jac_check: bool = True,
    verbose: bool = False,
) -> VectorRootResult:
    r"""Find the root of a system of nonlinear equations using a quasi-Newton
    method with optional global strategies.

    This function implements a quasi-Newton solver for systems of nonlinear
    equations according to Dennis and Schnabel (1996). The user can choose the
    approach to calculate and update the Jacobian approximation, as well as the
    global strategy to improve convergence from remote starting points.

    The default settings are meant to favor the likelihood of convergence at the
    expense of computational efficiency. For situations where maximum efficiency
    is desired and the initial guess is known to be close to the root, consider
    disabling the global method and using Broyden's update for the Jacobian.

    !!! note

        Solving systems of nonlinear equations is a surprisingly complex task —
        often more difficult than solving systems of differential equations or
        even multivariate optimization problems.
        Convergence is guaranteed only when the initial guess is sufficiently
        close to the root, which is rarely true in practice. The choice of a
        good initial guess, appropriate scaling factors, and a suitable global
        strategy is an essential part of solving the problem.

    **References**

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained Optimization
        and Nonlinear Equations", SIAM, 1996.

    Parameters
    ----------
    f : Callable[[FloatVector], FloatVector]
        Function whose root is to be found.
    x0 : FloatVectorLike (N)
        Initial guess for the root. Moreover, if no user-defined scale `sclx` is provided,
        the scaling factors will be determined from this value.
    tolx : float
        Tolerance for the scaled step size. The algorithm terminates when the scaled
        distance between two successive iterates `||Δx/max(x, 1/sclx)||∞` is below this
        threshold. If the value is too large, the algorithm may terminate prematurely. A
        value on the order of $\epsilon_f^{2/3}$ is typically recommended.
    tolf : float
        Tolerance for the scaled residual norm. This is the main convergence criterion.
        The algorithm terminates when the infinity norm of the scaled function values
        `||sclf*f(x)||∞` is below this threshold. A value on the order of
        $\epsilon_f^{1/3}$ is typically recommended.
    sclx : FloatVectorLike (N) | None
        Positive scaling factors for the components of `x`. Ideally, these should be
        chosen so that `sclx*x` is of order 1 near the solution for all components. By
        default, scaling is determined from `x0`.
    sclf : FloatVectorLike (N) | None
        Positive scaling factors for the components of `f`. Ideally, these should be
        chosen so that `sclf*f` is of order 1 near the root for all components. By
        default, scaling is determined from the initial Jacobian.
    maxiter : int
        Maximum number of outer quasi-Newton iterations.
    maxlenfac : float
        Factor determining the maximum allowable scaled step length `||sclx*Δx||₂` for
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
        fulfiled. With `dogleg`, a trust-region dogleg method is used to compute both the
        step direction and length. If `None`, no global strategy is used and the full
        quasi-Newton step is taken at each iteration.
    broyden_update : bool
        If `True`, the Jacobian is updated at each iteration using Broyden's rank-1 update
        formula. If `False`, the Jacobian is computed at each iteration using either the
        provided `jac` function or finite differences. Broyden's update significantly
        reduces the number of function/Jacobian evaluations required, but may lead to
        inaccurate Jacobian approximations and poor convergence if the initial guess is
        far from the root or if the function is highly nonlinear.
    jac : Callable[[FloatVector], FloatMatrix] | None
        Function to compute the Jacobian  of `f`. By default, the Jacobian is approximated
        using forward finite differences. In this case, setting `epsf` appropriately is
        essential.
    jac0 : FloatMatrix | None
        Initial Jacobian approximation at `x0`. If provided, it is used instead of
        computing the Jacobian at the first iteration. This can be useful in case of a
        restart, or when a simple initial approximation is sufficient (e.g., the identity
        matrix) and one wants to reduce the number of function calls.
    jac_check : bool
        If `True`, the Jacobian provided by `jac` is checked against a forward
        finite-difference approximation at `x0`. This can help identify errors in the
        user-provided Jacobian.
    verbose : bool
        Print iteration information.

    Returns
    -------
    VectorRootResult
        Dataclass with root solution results.

    Examples
    --------
    Find the steady-state concentration of species A, B, and C at the outlet of
    a CSTR, assuming a consecutive scheme of type A+B→C, C+B→D.
    >>> from polykin.math import rootvec_qnewton
    >>> import numpy as np
    >>> def f(x, A0=1.0, B0=2.0, C0=0.0, k1=1e-3, k2=5e-4, tau=1e3):
    ...     "Steady-state mole balances: inflow - outflow ± reaction"
    ...     A, B, C = x
    ...     f = np.zeros_like(x)
    ...     f[0] = (A0 - A)/tau - k1*A*B
    ...     f[1] = (B0 - B)/tau - k1*A*B
    ...     f[2] = (C0 - C)/tau + k1*A*B - k2*C*B
    ...     return f
    >>> sol = rootvec_qnewton(f, np.array([0.5, 1.0, 0.5]))
    >>> sol.x
    array([0.41421356, 1.41421356, 0.34314575])
    """
    # Construct method name for result
    method = "Quasi-Newton"
    method_options = []
    _global_method = global_method if global_method else "none"
    method_options.append(f"Global: {_global_method.title()}")
    if broyden_update:
        method_options.append("Broyden update")
    if method_options:
        method += " (" + ", ".join(method_options) + ")"

    # Initialize results variables
    success = False
    message = ""
    nfeval = 0
    njeval = 0

    # Evaluate function at x0
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    xc = x0.copy()
    fc = f(xc)
    nfeval += 1

    # Set x scaling factors
    sclx = np.abs(np.asarray(sclx, dtype=float)) if sclx is not None else scalex(x0)

    # Evaluate Jacobian at x0
    if jac0 is not None:
        J = jac0.copy()
    else:
        if jac is not None:
            J = jac(xc)
            njeval += 1
        else:
            J = jacobian_forward(f, xc, fx=fc, sclx=sclx, epsf=epsf)
            nfeval += n

    # Check user-provided Jacobian against finite-difference approximation
    if jac_check and jac is not None:
        J_fd = jacobian_forward(f, xc, fx=fc, sclx=sclx, epsf=epsf)
        nfeval += n
        _epsf = max(epsf, eps) if epsf is not None else eps
        tol = 1e2 * math.sqrt(_epsf)
        if not np.allclose(J, J_fd, rtol=tol, atol=tol):
            message = "User-provided Jacobian `jac` does not match finite-difference approximation."  # noqa: E501
            return VectorRootResult(method, False, message, nfeval, njeval, 0, x0, fc, J)

    # Set f scaling factors
    if sclf is None:
        sclf = np.max(np.abs(J), axis=1)
        sclf[sclf == 0.0] = 1.0
        sclf = 1.0 / sclf
    else:
        sclf = np.abs(np.asarray(sclf, dtype=float))

    # Check initial solution with tight tolerance
    if norm(sclf * fc, np.inf) <= 1e-2 * tolf:
        message = "||sclf*f(x0)||∞ ≤ 1e-2*tolf"
        return VectorRootResult(method, True, message, nfeval, njeval, 0, x0, fc, J)

    # Set maximum step length for global methods
    maxlen = max(0.0, maxlenfac) * float(max(norm(sclx * x0), norm(sclx)))

    # Set initial trust region radius for dogleg method
    if trustlen is None:
        trustlen = -1.0  # Sentinel value
    else:
        trustlen = min(trustlen, maxlen)

    # Objective function for global methods
    def fN(x: FloatVector) -> tuple[float, FloatVector]:
        """1/2*||sclf*f(x)||²."""
        fx = f(x)
        fNx = 0.5 * np.sum((sclf * fx) ** 2)
        return (fNx, fx)

    gm_nmaxsteps = 0
    restart = True
    niter = 0
    Q = R = np.array([])
    gc = np.array([])
    fNc = float("nan")

    for niter in range(1, maxiter + 1):
        if verbose:
            print(f"Iteration {niter:3d}:", flush=True)

        # QR decomposition of scaled Jacobian
        if not broyden_update or restart:
            try:
                Q, R = np.linalg.qr(sclf[:, None] * J)
            except np.linalg.LinAlgError as e:
                message = f"QR decomposition of Jacobian failed: {e}."
                break

        # Condition number of R
        Rcond = np.linalg.cond(R / sclx, 1)

        # Solve (Q*R)*p = - sclf*fc
        if Rcond < 1 / sqrt_eps:
            p = scipy.linalg.solve_triangular(
                R,
                Q.T @ (sclf * fc),
                overwrite_b=True,
                check_finite=False,
            )
            p *= -1
            if global_method:
                gc = R.T @ (Q.T @ (sclf * fc))
                fNc = 0.5 * np.sum((sclf * fc) ** 2)
        else:
            if verbose:
                print("R is ill-conditioned (cond={Rcond:.2e}).", flush=True)
            H = R.T @ R
            Hnorm = norm(H / (sclx[:, None] * sclx[None, :]), 1)
            H[np.diag_indices_from(H)] += math.sqrt(n * eps) * Hnorm * sclx**2
            gc = R.T @ (Q.T @ (sclf * fc))
            R, _ = scipy.linalg.cho_factor(H, overwrite_a=True, check_finite=False)
            p = scipy.linalg.cho_solve((R, False), gc, check_finite=False)
            p *= -1

        # Compute actual x step
        if global_method is None:
            xp = xc + p
            fp = f(xp)
            gm_ismaxstep = True
            gm_success = True
            gm_nfeval = 1
        elif global_method == "line-search":
            gm_success, gm_ismaxstep, gm_nfeval, xp, _, fp = line_search(
                fN, p, xc, fNc, gc, tolx, sclx, maxlen, verbose
            )
        elif global_method == "dogleg":
            (gm_success, gm_ismaxstep, gm_nfeval, trustlen, xp, _, fp) = dogleg(
                fN, p, xc, fNc, gc, R, tolx, sclx, maxlen, trustlen, verbose
            )
        else:
            raise ValueError(f"Unknown `global_method`: {global_method}.")

        nfeval += gm_nfeval
        gm_nmaxsteps = gm_nmaxsteps + 1 if gm_ismaxstep else 0

        # Display iteration progress
        if verbose:
            print(
                f"  x = {xp}\n  ||sclx*f(x)||∞ = {norm(sclf * fp, np.inf):.2e}",
                flush=True,
            )

        # If global method step failed, restart once
        if not gm_success and not restart:
            if jac is not None:
                J = jac(xc)
                njeval += 1
            else:
                J = jacobian_forward(f, xc, fx=fc, sclx=sclx, epsf=epsf)
                nfeval += n
            restart = True
            continue

        # Check termination and convergence conditions
        if not gm_success:
            message = """Last global step failed to decrease ||sclx*f(x)||₂ sufficiently.
            Either `x` is close to a root and no more accuracy is possible, or the secant
            approximation to the Jacobian is inaccurate, or `tolx` is too large."""
            stop = True
        elif norm(sclf * fp, np.inf) <= tolf:
            message = "||sclf*f(x)||∞ ≤ tolf"
            success = True
            stop = True
        elif norm((xp - xc) / np.maximum(np.abs(xp), 1 / sclx), np.inf) <= tolx:
            message = """||Δx/max(x, 1/sclx)||∞ ≤ tolx
            `x` may be an approximate root, but it is also possible that the algorithm is
            making slow progress and is not near a root, or that `tolx` is too large."""
            stop = True
        elif global_method and gm_nmaxsteps >= 5:
            message = """Maximum number (5) of consecutive steps of length `maxlen`
            reached. Perhaps stuck in a flat region or `maxlen` is too small."""
            stop = True
        else:
            stop = False

        if stop:
            xc, fc = xp, fp
            break

        # Update Jacobian
        if broyden_update:
            Q, R = _update_broyden(xc, xp, fc, fp, Q, R, sclx, sclf)
        else:
            if jac is not None:
                J = jac(xp)
                njeval += 1
            else:
                J = jacobian_forward(f, xp, fx=fp, sclx=sclx, epsf=epsf)
                nfeval += n

        # Next iteration
        xc, fc = xp, fp
        restart = False

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    if broyden_update:
        J = (Q @ R) / sclf[:, None]

    return VectorRootResult(method, success, message, nfeval, njeval, niter, xc, fc, J)


def _update_broyden(
    xc: FloatVector,
    xp: FloatVector,
    fc: FloatVector,
    fp: FloatVector,
    Qc: FloatMatrix,
    Rc: FloatMatrix,
    sclx: FloatVector,
    sclf: FloatVector,
) -> tuple[FloatMatrix, FloatMatrix]:
    r"""Perform a Broyden update of the QR decomposition of a Jacobian approximation.

    This function updates the QR factors of an approximate Jacobian according to the
    Broyden rank-1 update formula, using scaling factors for both the variables and
    function values to improve numerical conditioning.

    **References**

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
        Optimization and Nonlinear Equations", SIAM, 1996.

    Parameters
    ----------
    xc : FloatVector
        Current value of the variable vector
    xp : FloatVector
        Next value of the variable vector.
    fc : FloatVector
        Current function value, `f(xc)`.
    fp : FloatVector
        Next function value, `f(xp)`.
    Qc : FloatMatrix
        Orthogonal factor of the current Jacobian QR decomposition.
    Rc : FloatMatrix
        Upper-triangular factor of the current Jacobian QR decomposition.
    sclx : FloatVector
        Scaling factors for the components of `x`.
    sclf : FloatVector
        Scaling factors for the components of `f`.

    Returns
    -------
    tuple[FloatMatrix, FloatMatrix]
        The updated orthogonal and upper-triangular factors, `(Qp, Rp)`.
    """
    s = xp - xc
    y = fp - fc

    w = sclf * y - Qc @ (Rc @ s)
    w[np.abs(w) < eps * sclf * (np.abs(fp) + np.abs(fc))] = 0.0

    t = s * sclx**2
    s = t / np.dot(s, t)
    Qp, Rp = scipy.linalg.qr_update(Qc, Rc, w, s, overwrite_qruv=True)

    return (Qp, Rp)
