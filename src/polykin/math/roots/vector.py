# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from collections.abc import Callable
from enum import IntEnum
from typing import Literal

import numpy as np
import scipy
from numpy import dot, isclose, sqrt
from numpy.linalg import norm

from polykin.math.derivatives import jacobian_forward, scalex
from polykin.math.roots.results import VectorRootResult
from polykin.utils.math import eps
from polykin.utils.types import FloatMatrix, FloatVector

all = ["rootvec_qnewton"]


def rootvec_qnewton(
    f: Callable[[FloatVector], FloatVector],
    x0: FloatVector,
    *,
    tolx: float = 4e-11,
    tolf: float = 6e-6,
    sclx: FloatVector | None = None,
    sclf: FloatVector | None = None,
    maxiter: int = 100,
    maxlenfac: float = 1e3,
    trustlen: float | None = None,
    ndigit: int | None = None,
    global_method: Literal["line-search", "dogleg"] | None = "line-search",
    broyden_update: bool = False,
    jac: Callable[[FloatVector], FloatMatrix] | None = None,
    jac0: FloatMatrix | None = None,
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

    * J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
      Optimization and Nonlinear Equations", SIAM, 1996.

    Parameters
    ----------
    f : Callable[[FloatVector], FloatVector]
        Function whose root is to be found.
    x0 : FloatVector
        Initial guess for the root. Moreover, if no user-defined scale `sclx`
        is provided, the scaling factors will be determined from this value.
    tolx : float
        Tolerance for the scaled step size. The algorithm terminates when the
        scaled distance between two successive iterates `||Δx/max(x, 1/sclx)||∞`
        is below this threshold. If the value is too large, the algorithm may
        terminate prematurely. A value on the order of $\epsilon^{2/3}$ is
        typically recommended.
    tolf : float
        Tolerance for the scaled residual norm. This is the main convergence
        criterion. The algorithm terminates when the infinity norm of the scaled
        function values `||sclf*f(x)||∞` is below this threshold. A value on
        the order of $\epsilon^{1/3}$ is typically recommended.
    sclx : FloatVector | None
        Positive scaling factors for the components of `x`. Ideally, these
        should be chosen so that `sclx*x` is of order 1 near the solution for
        all components. By default, scaling is determined from `x0`.
    sclf : FloatVector | None
        Positive scaling factors for the components of `f`. Ideally, these
        should be chosen so that `sclf*f` is of order 1 near the root for all
        components. By default, scaling is determined from the initial Jacobian.
    maxiter : int
        Maximum number of outer quasi-Newton iterations.
    maxlenfac : float
        Factor determining the maximum allowable scaled step length `||sclx*Δx||₂`
        for global methods. Used to prevent steps that would cause the algorithm
        to overflow, leave the domain of interest, or diverge. It should be
        chosen small enough to prevent such issues, but large enough to allow
        any anticipated reasonable step length.
    trustlen : float | None
        Initial trust region radius for the `dogleg` global method. By default,
        the length of the initial scaled gradient is used.
    ndigit : int | None
        Number of reliable digits returned by `f`. Used to set the step size
        for finite-difference Jacobian approximations. By default, 64-bit float
        precision is assumed (i.e., ~15 digits).
    global_method : Literal['line-search','dogleg'] | None
        Global strategy to improve convergence from remote starting points. With
        `line-search`, the search direction is computed using the quasi-Newton
        step and the length of the step is determined by backtracking until
        the Armijo condition is fullfiled. With `dogleg`, a trust-region dogleg
        method is used to compute both the step direction and length. If `None`,
        no global strategy is used and the full quasi-Newton step is taken at
        each iteration.
    broyden_update : bool
        If `True`, the Jacobian is updated at each iteration using Broyden's
        rank-1 update formula. If `False`, the Jacobian is computed at each
        iteration using either the provided `jac` function or finite differences.
        Broyden's update significantly reduces the number of function/Jacobian
        evaluations required, but may lead to inaccurate Jacobian approximations
        and poor convergence if the initial guess is far from the root or if the
        function is highly nonlinear.
    jac : Callable[[FloatVector], FloatMatrix] | None
        Function to compute the Jacobian  of `f`. By default, the Jacobian is
        approximated using forward finite differences. In this case, setting
        `ndigit` appropriately is essential.
    jac0 : FloatMatrix | None
        Initial Jacobian approximation at `x0`. If provided, it is used instead
        of computing the Jacobian at the first iteration. This can be useful in
        case of a restart, or when a simple initial approximation is sufficient
        (e.g., the identity matrix) and one wants to reduce the number of function
        calls.
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
    method = "Quasi-Newton"
    if global_method:
        method += f" ({global_method})"

    success = False
    message = ""
    nfeval = 0
    njeval = 0

    # Evaluate function at x0
    n = x0.size
    xc = x0.copy()
    fc = f(xc)
    nfeval += 1

    # Set x scaling factors
    if sclx is None:
        sclx = scalex(x0)

    # Evaluate Jacobian at x0
    if jac0 is not None:
        J = jac0.copy()
    else:
        if jac is not None:
            J = jac(xc)
            njeval += 1
        else:
            J = jacobian_forward(f, xc, fx=fc, sclx=sclx, ndigit=ndigit)
            nfeval += n

    # Set f scaling factors
    if sclf is None:
        sclf = np.max(np.abs(J), axis=1)
        sclf[sclf == 0.0] = 1.0
        sclf = 1 / sclf

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

    # Norm function for global methods
    def fN(fx: FloatVector) -> float:
        """1/2*||sclf*f(x)||²."""
        return 0.5 * np.sum((sclf * fx) ** 2)

    gm_nmaxsteps = 0
    restart = True
    Q = np.array([])
    R = np.array([])

    for niter in range(1, maxiter + 1):

        if verbose:
            print(f"Iteration {niter:3d}:", flush=True)

        # QR decomposition of scaled Jacobian
        if not broyden_update or restart:
            try:
                Q, R = scipy.linalg.qr(sclf[:, None] * J)
            except Exception as e:
                message = f"QR decomposition of Jacobian failed: {e}."
                break

        # Condition number of R
        Rcond = np.linalg.cond(R / sclx, 1)

        # Solve (Q*R)*p = - sclf*fc
        if Rcond < 1 / sqrt(eps):
            p = -scipy.linalg.solve_triangular(R, Q.T @ (sclf * fc))
            if global_method:
                gc = R.T @ (Q.T @ (sclf * fc))
        else:
            if verbose:
                print("R is ill-conditioned (cond={Rcond:.2e}).", flush=True)
            H = R.T @ R
            Hnorm = norm(H / (sclx[:, None] * sclx[None, :]), 1)
            H[np.diag_indices_from(H)] += sqrt(n * eps) * Hnorm * sclx**2
            gc = R.T @ (Q.T @ (sclf * fc))
            R, _ = scipy.linalg.cho_factor(H, overwrite_a=True)
            p = -scipy.linalg.cho_solve((R, False), gc)

        # Compute actual x step
        if global_method is None:
            xp = xc + p
            fp = f(xp)
            gm_ismaxstep = True
            gm_success = True
            gm_nfeval = 1
        elif global_method == "line-search":
            gm_success, gm_ismaxstep, gm_nfeval, xp, fp, _ = line_search(
                f, fN, xc, fc, gc, p, tolx, sclx, maxlen, verbose
            )
        elif global_method == "dogleg":
            gm_success, gm_ismaxstep, gm_nfeval, xp, fp, _, trustlen = dogleg(
                f, fN, xc, fc, gc, p, R, tolx, sclx, maxlen, trustlen, verbose
            )
        else:
            raise ValueError(f"Unknown `global_method`: {global_method}.")

        nfeval += gm_nfeval
        gm_nmaxsteps = gm_nmaxsteps + 1 if gm_ismaxstep else 0

        # Display iteration progress
        if verbose:
            print(
                f"  x = {xp}\n" f"  ||sclx*f(x)||∞ = {norm(sclf*fp, np.inf):.2e}",
                flush=True,
            )

        # If global method step failed, restart once
        if not gm_success and not restart:
            if jac is not None:
                J = jac(xc)
                njeval += 1
            else:
                J = jacobian_forward(f, xc, fx=fc, sclx=sclx, ndigit=ndigit)
                nfeval += n
            restart = True
            continue

        # Check termination and convergence conditions
        if not gm_success:
            message = """Last global step failed to decrease ½||sclx*f(x)||₂ sufficiently.
            Either `x` is close to a root and no more accuracy is possible,
            or the secant approximation to the Jacobian is inaccurate,
            or `tolx` is too large."""
            stop = True
        elif norm(sclf * fp, np.inf) <= tolf:
            message = "||sclf*f(x)||∞ ≤ tolf"
            success = True
            stop = True
        elif norm((xp - xc) / np.maximum(np.abs(xp), 1 / sclx), np.inf) <= tolx:
            message = """||Δx/max(x, 1/sclx)||∞ ≤ tolx
            `x` may be an approximate root, but it is also possible that the
            the algorithm is making slow progress and is not near a root,
            or that `tolx` is too large."""
            stop = True
        elif global_method and gm_nmaxsteps >= 5:
            message = """Maximum number (5) of consecutive steps of length `maxlen` reached.
            Perhaps stuck in a flat region or `maxlen` is too small."""
            stop = True
        else:
            stop = False

        if stop:
            xc, fc = xp, fp
            break

        # Update Jacobian
        if broyden_update:
            Q, R = _update_broyden_qr(xc, xp, fc, fp, Q, R, sclx, sclf)
        else:
            if jac is not None:
                J = jac(xp)
                njeval += 1
            else:
                J = jacobian_forward(f, xp, fx=fp, sclx=sclx, ndigit=ndigit)
                nfeval += n

        # Next iteration
        xc, fc = xp, fp
        restart = False

    else:
        message = f"Maximum number of iterations ({maxiter}) reached."

    if broyden_update:
        J = (Q @ R) / sclf[:, None]

    return VectorRootResult(method, success, message, nfeval, njeval, niter, xc, fc, J)


def _update_broyden_qr(
    xc: FloatVector,
    xp: FloatVector,
    fc: FloatVector,
    fp: FloatVector,
    Qc: FloatMatrix,
    Rc: FloatMatrix,
    sclx: FloatVector,
    sclf: FloatVector,
) -> tuple[FloatMatrix, FloatMatrix]:
    r"""Perform a Broyden update of the QR decomposition of a Jacobian
    approximation.

    This function updates the QR factors of an approximate Jacobian according
    to the Broyden rank-1 update formula, using scaling factors for both the
    variables and function values to improve numerical conditioning.

    **References**

    * J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
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
    s = t / dot(s, t)
    Qp, Rp = scipy.linalg.qr_update(Qc, Rc, w, s, overwrite_qruv=True)

    return (Qp, Rp)


def line_search(
    f: Callable[[FloatVector], FloatVector],
    fN: Callable[[FloatVector], float],
    xc: FloatVector,
    fc: FloatVector,
    gc: FloatVector,
    p: FloatVector,
    tolx: float,
    sclx: FloatVector,
    maxlen: float,
    verbose: bool = False,
) -> tuple[bool, bool, int, FloatVector, FloatVector, float]:
    r"""Perform a line search.

    This function performs a line search along the quasi-Newton direction to
    find a step size that satisfies the Armijo condition.

    **References**

    * J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
      Optimization and Nonlinear Equations", SIAM, 1996.

    Parameters
    ----------
    f : Callable[[FloatVector], FloatVector]
        Function whose root is to be found.
    fN : Callable[[FloatVector], float]
        Norm function.
    xc : FloatVector
        Current value of the variable vector.
    fc : FloatVector
        Current function value, `f(xc)`.
    gc : FloatVector
        Gradient of the norm function, `∇fN(xc)`.
    p : FloatVector
        Quasi-Newton step.
    tolx : float
        Tolerance for the step size.
    sclx : FloatVector
        Scaling factors for `x`.
    maxlen : float
        Maximum step length.
    verbose : bool
        Print iteration information.

    Returns
    -------
    tuple[bool, bool, int, FloatVector, FloatVector, float]
        `(success, ismaxstep, nfeval, xp, fp, fNp)`
    """
    nfeval = 0
    success = False
    ismaxstep = False

    newtlen = norm(sclx * p)
    if newtlen > maxlen:
        p = p * (maxlen / newtlen)
        newtlen = maxlen

    fNc = fN(fc)
    slope = dot(gc, p)

    α = 1e-4
    λ = 1.0
    λmin = tolx / np.max(np.abs(p) / np.maximum(np.abs(xc), 1 / sclx))

    A = np.empty((2, 2))
    B = np.empty(2)
    λ_prev = np.nan
    fNp_prev = np.nan

    first = True
    while True:

        xp = xc + λ * p
        fp = f(xp)
        fNp = fN(fp)
        nfeval += 1

        if verbose:
            print(f"  λ = {λ:.2e}, ½||sclx*f(x)||² = {fNp:.2e}", flush=True)

        if fNp <= fNc + α * λ * slope:
            success = True
            if first and (newtlen > 0.99 * maxlen):
                ismaxstep = True
            break
        elif λ < λmin:
            success = False
            xp = xc
            break
        else:
            if first:
                λtemp = -slope / (2 * (fNp - fNc - slope))
                first = False
            else:
                A[0, 0] = 1 / λ**2
                A[0, 1] = -1 / λ_prev**2
                A[1, 0] = -λ_prev / λ**2
                A[1, 1] = λ / λ_prev**2
                B[0] = fNp - fNc - λ * slope
                B[1] = fNp_prev - fNc - λ_prev * slope
                a, b = 1 / (λ - λ_prev) * A @ B
                if isclose(a, 0.0):
                    λtemp = -slope / (2 * b)
                else:
                    λtemp = (-b + sqrt(b**2 - 3 * a * slope)) / (3 * a)
                λtemp = min(λtemp, 0.5 * λ)
            λ_prev = λ
            fNp_prev = fNp
            λ = max(0.1 * λ, λtemp)

    return (success, ismaxstep, nfeval, xp, fp, fNp)


class TrustState(IntEnum):
    """Codes for the status of the trust region step and update."""

    accepted = 0
    convergence = 1
    rejected = 2
    exploratory_success = 3
    start = 4


def dogleg(
    f: Callable[[FloatVector], FloatVector],
    fN: Callable[[FloatVector], float],
    xc: FloatVector,
    fc: FloatVector,
    gc: FloatVector,
    p: FloatVector,
    R: FloatMatrix,
    tolx: float,
    sclx: FloatVector,
    maxlen: float,
    trustlen: float,
    verbose: bool = False,
) -> tuple[bool, bool, int, FloatVector, FloatVector, float, float]:
    r"""Perform a dogleg step.

    **References**

    * J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
      Optimization and Nonlinear Equations", SIAM, 1996.

    Parameters
    ----------
    f : Callable[[FloatVector], FloatVector]
        Function whose root is to be found.
    fN : Callable[[FloatVector], float]
        Norm function.
    xc : FloatVector
        Current value of the variable vector.
    fc : FloatVector
        Current function value, `f(xc)`.
    gc : FloatVector
        Gradient of the norm function, `∇fN(xc)`.
    p : FloatVector
        Quasi-Newton step.
    R : FloatMatrix
        Upper-triangular factor of the current Jacobian QR decomposition.
    tolx : float
        Tolerance for the step size.
    sclx : FloatVector
        Scaling factors for `x`.
    maxlen : float
        Maximum step length.
    trustlen : float
        Current trust region radius.
    verbose : bool
        Print iteration information.

    Returns
    -------
    tuple[bool, bool, int, FloatVector, FloatVector, float, float]
        `(success, ismaxstep, nfeval, xp, fp, fNp, trustlen)`
    """
    nfeval = 0
    state = TrustState.start
    ismaxstep = False

    cauchylen = np.nan
    η = np.nan
    v = np.full(p.size, np.nan)
    sSD = np.full(p.size, np.nan)
    xp_prev = xc
    fp_prev = fc
    fNp_prev = 0.0

    newtlen = float(norm(sclx * p))
    fNc = fN(fc)

    first = True
    while state not in (TrustState.accepted, TrustState.convergence):

        # Perform dogleg step to determine s
        if newtlen <= trustlen:
            isnewtstep = True
            s = p
            trustlen = newtlen
        else:
            isnewtstep = False
            if first:
                first = False
                α = norm(gc / sclx) ** 2
                β = norm(R @ (gc / sclx**2)) ** 2
                sSD = -(α / β) * (gc / sclx)
                cauchylen = α * sqrt(α) / β
                η = 0.2 + (0.8 * α**2 / (β * abs(dot(gc, p))))
                v = η * (p * sclx) - sSD
                if trustlen <= 0.0:
                    trustlen = min(cauchylen, maxlen)

            if η * newtlen <= trustlen:
                s = (trustlen / newtlen) * p
            elif cauchylen >= trustlen:
                s = (trustlen / cauchylen) * (sSD / sclx)
            else:
                a = dot(v, v)
                b = dot(v, sSD)
                λ = (-b + sqrt(b**2 - a * (cauchylen**2 - trustlen**2))) / a
                s = (sSD + λ * v) / sclx

        # Update trust region
        state, ismaxstep, trustlen, xp, fp, fNp, xp_prev, fp_prev, fNp_prev = (
            _update_trust_region(
                f,
                fN,
                fNc,
                xc,
                gc,
                R,
                s,
                sclx,
                tolx,
                maxlen,
                trustlen,
                isnewtstep,
                xp_prev,
                fp_prev,
                fNp_prev,
                state,
            )
        )
        nfeval += 1

        # Display iteration progress
        if verbose:
            print(f"  δ = {trustlen:.2e}, ½||sclx*f(x)||² = {fNp:.2e}", flush=True)

    return (state == 0, ismaxstep, nfeval, xp, fp, fNp, trustlen)


def _update_trust_region(
    f: Callable[[FloatVector], FloatVector],
    fN: Callable[[FloatVector], float],
    fNc: float,
    xc: FloatVector,
    gc: FloatVector,
    R: FloatMatrix,
    s: FloatVector,
    sclx: FloatVector,
    tolx: float,
    maxlen: float,
    trustlen: float,
    isnewtstep: bool,
    xp_prev: FloatVector,
    fp_prev: FloatVector,
    fNp_prev: float,
    state: TrustState,
) -> tuple[
    TrustState,
    bool,
    float,
    FloatVector,
    FloatVector,
    float,
    FloatVector,
    FloatVector,
    float,
]:
    r"""Perform trust-region update.

    **References**

    * J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
      Optimization and Nonlinear Equations", SIAM, 1996.

    Parameters
    ----------
    f : Callable[[FloatVector], FloatVector]
        Function whose root is to be found.
    fN : Callable[[FloatVector], float]
        Norm function.
    fNc : float
        Norm function value at current point, `fN(fc)`.
    xc : FloatVector
        Current value of the variable vector.
    gc : FloatVector
        Gradient of the norm function, `∇fN(xc)`.
    R : FloatMatrix
        Upper-triangular factor of the current Jacobian QR decomposition.
    s : FloatVector
        Step vector.
    sclx : FloatVector
        Scaling factors for `x`.
    tolx : float
        Tolerance for the step size.
    maxlen : float
        Maximum step length.
    trustlen : float
        Current trust region radius.
    isnewtstep : bool
        Flag indicating if the step is a Newton step.
    xp_prev : FloatVector
        Previous value of `xp`.
    fp_prev : FloatVector
        Previous value of `fp`.
    fNp_prev : float
        Previous value of `fN(fp)`.
    state : TrustRegionState
        Code indicating the current state of the algorithm.

    Returns
    -------
    tuple[TrustRegionState, bool, float, FloatVector, FloatVector, float, FloatVector, FloatVector, float]
        `(state, ismaxstep, trustlen, xp, fp, fNp, xp_prev, fp_prev, fNp_prev)`
    """
    α = 1e-4
    ismaxstep = False

    steplen = norm(sclx * s)
    slope = dot(gc, s)

    xp = xc + s
    fp = f(xp)

    fNp = fN(fp)
    ΔfN = fNp - fNc

    if (state == TrustState.exploratory_success) and (
        (fNp >= fNp_prev) or (ΔfN > α * slope)
    ):
        state = TrustState.accepted
        xp, fp, fNp = xp_prev, fp_prev, fNp_prev
        trustlen *= 0.5
    elif ΔfN >= α * slope:
        rlen = np.max(np.abs(s) / np.maximum(np.abs(xp), 1 / sclx))
        if rlen < tolx:
            state = TrustState.convergence
            xp = xc
        else:
            state = TrustState.rejected
            trustlen = np.clip(
                -slope * steplen / (2 * (ΔfN - slope)), 0.1 * trustlen, 0.5 * trustlen
            )
    else:
        ΔfN_pred = slope + 0.5 * norm(R @ s) ** 2
        if (
            (state != TrustState.rejected)
            and (not isnewtstep)
            and (trustlen <= 0.99 * maxlen)
            and ((abs(ΔfN_pred - ΔfN) <= 0.1 * abs(ΔfN)) or (ΔfN <= slope))
        ):
            state = TrustState.exploratory_success
            xp_prev, fp_prev, fNp_prev = xp, fp, fNp
            trustlen = min(2 * trustlen, maxlen)
        else:
            state = TrustState.accepted
            if steplen >= 0.99 * maxlen:
                ismaxstep = True
            if ΔfN >= 0.1 * ΔfN_pred:
                trustlen *= 0.5
            elif ΔfN <= 0.75 * ΔfN_pred:
                trustlen = min(2 * trustlen, maxlen)
            else:
                trustlen = trustlen  # no change

    return (state, ismaxstep, trustlen, xp, fp, fNp, xp_prev, fp_prev, fNp_prev)
