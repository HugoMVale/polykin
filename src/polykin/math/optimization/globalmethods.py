# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

import math
from collections.abc import Callable
from enum import IntEnum

import numpy as np
from numpy.linalg import norm

from polykin.math.machine import sqrt_eps
from polykin.utils.typing import FloatMatrix, FloatVector

__all__ = ["line_search", "dogleg"]


def line_search(
    f: Callable[[FloatVector], float | tuple[float, FloatVector]],
    p: FloatVector,
    xc: FloatVector,
    fc: float,
    gc: FloatVector,
    tolx: float,
    sclx: FloatVector,
    maxlen: float,
    verbose: bool = False,
) -> tuple[bool, bool, int, FloatVector, float, FloatVector]:
    r"""Perform a line search.

    This function performs a line search along the quasi-Newton direction to find a step
    size that satisfies the Armijo condition.

    **References**

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
        Optimization and Nonlinear Equations", SIAM, 1996.

    Parameters
    ----------
    f : Callable[[FloatVector], float | tuple[float, FloatVector]]
        Objective function. For compability with optimization and root-finding algorithms,
        `f` can return either a scalar objective function value or a tuple including the
        scaled norm of the vector root function and the vector root function itself.
    p : FloatVector
        Quasi-Newton step.
    xc : FloatVector
        Current value of the variable vector.
    fc : float
        Current objective function value, `f(xc)`.
    gc : FloatVector
        Current gradient of the objective function, `∇f(xc)`.
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
    tuple[bool, bool, int, FloatVector, float, FloatVector]
        `(success, ismaxstep, nfeval, xp, fp, Fp)`
    """
    nfeval = 0
    success = False
    ismaxstep = False

    newtlen = norm(sclx * p)
    if newtlen > maxlen:
        p = p * (maxlen / newtlen)
        newtlen = maxlen

    slope = np.dot(gc, p)

    α = 1e-4
    λ = 1.0
    λ_min = tolx / np.max(np.abs(p) / np.maximum(np.abs(xc), 1 / sclx))

    A = np.empty((2, 2))
    B = np.empty(2)
    λ_prev = float("nan")
    fp_prev = float("nan")
    Fp = np.array([])

    first = True
    while True:
        xp = xc + λ * p

        res = f(xp)
        nfeval += 1
        fp, Fp = res if isinstance(res, tuple) else (res, Fp)

        if verbose:
            print(f"  λ = {λ:.2e}, f={fp:.2e}", flush=True)

        if fp <= fc + α * λ * slope:
            success = True
            if first and (newtlen > 0.99 * maxlen):
                ismaxstep = True
            break
        elif λ < λ_min:
            success = False
            xp = xc
            break
        else:
            if first:
                λ_temp = -slope / (2 * (fp - fc - slope))
                first = False
            else:
                A[0, 0] = 1 / λ**2
                A[0, 1] = -1 / λ_prev**2
                A[1, 0] = -λ_prev / λ**2
                A[1, 1] = λ / λ_prev**2
                B[0] = fp - fc - λ * slope
                B[1] = fp_prev - fc - λ_prev * slope
                a, b = 1 / (λ - λ_prev) * A @ B
                if abs(a) < sqrt_eps:
                    λ_temp = -slope / (2 * b)
                else:
                    λ_temp = (-b + math.sqrt(b**2 - 3 * a * slope)) / (3 * a)
                λ_temp = min(λ_temp, 0.5 * λ)
            λ_prev = λ
            fp_prev = fp
            λ = max(0.1 * λ, λ_temp)

    return (success, ismaxstep, nfeval, xp, fp, Fp)


class TrustState(IntEnum):
    """Codes for the status of the trust region step and update."""

    accepted = 0
    convergence = 1
    rejected = 2
    exploratory_success = 3
    start = 4


def dogleg(
    f: Callable[[FloatVector], float | tuple[float, FloatVector]],
    p: FloatVector,
    xc: FloatVector,
    fc: float,
    gc: FloatVector,
    R: FloatMatrix,
    tolx: float,
    sclx: FloatVector,
    maxlen: float,
    trustlen: float,
    verbose: bool = False,
) -> tuple[bool, bool, int, float, FloatVector, float, FloatVector]:
    r"""Perform a dogleg step.

    **References**

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
        Optimization and Nonlinear Equations", SIAM, 1996.

    Parameters
    ----------
    f : Callable[[FloatVector], float | tuple[float, FloatVector]]
        Objective function. For compability with optimization and root-finding algorithms,
        `f` can return either a scalar objective function value or a tuple including the
        scaled norm of the vector root function and the vector root function itself.
    p : FloatVector
        Quasi-Newton step.
    xc : FloatVector
        Current value of the variable vector.
    fc : float
        Current objective function value, `f(xc)`.
    gc : FloatVector
        Current gradient of the objective function, `∇f(xc)`.
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
    tuple[bool, bool, int, float, FloatVector, float, FloatVector]
        `(success, ismaxstep, nfeval, trustlen, xp, fp, Fp)`
    """
    nfeval = 0
    state = TrustState.start
    ismaxstep = False

    cauchylen = float("nan")
    η = float("nan")
    v = np.full(p.size, np.nan)
    sSD = np.full(p.size, np.nan)

    xp = xp_prev = xc
    fp = fp_prev = float("nan")
    Fp = Fp_prev = np.array([])

    newtlen = norm(sclx * p)

    first = True
    while state not in (TrustState.accepted, TrustState.convergence):
        # Perform dogleg step to determine s
        if newtlen <= trustlen:
            isnewtstep = True
            s = p
            trustlen = newtlen  # type: ignore
        else:
            isnewtstep = False
            if first:
                first = False
                α = norm(gc / sclx) ** 2
                β = norm(R @ (gc / sclx**2)) ** 2
                sSD = -(α / β) * (gc / sclx)
                cauchylen = α * math.sqrt(α) / β
                η = 0.2 + (0.8 * α**2 / (β * abs(np.dot(gc, p))))
                v = η * (p * sclx) - sSD
                if trustlen <= 0.0:
                    trustlen = min(cauchylen, maxlen)  # type: ignore

            if η * newtlen <= trustlen:
                s = (trustlen / newtlen) * p
            elif cauchylen >= trustlen:
                s = (trustlen / cauchylen) * (sSD / sclx)
            else:
                a = np.dot(v, v)
                b = np.dot(v, sSD)
                λ = (-b + math.sqrt(b**2 - a * (cauchylen**2 - trustlen**2))) / a
                s = (sSD + λ * v) / sclx

        # Update trust region
        state, ismaxstep, trustlen, xp, fp, Fp, xp_prev, fp_prev, Fp_prev = (
            _update_trust_region(
                f,
                xc,
                fc,
                gc,
                R,
                s,
                tolx,
                sclx,
                maxlen,
                trustlen,
                isnewtstep,
                state,
                xp_prev,
                fp_prev,
                Fp_prev,
            )
        )
        nfeval += 1

        # Display iteration progress
        if verbose:
            print(f"  δ = {trustlen:.2e}, ½||sclx*f(x)||² = {fp:.2e}", flush=True)

    return (state == TrustState.accepted, ismaxstep, nfeval, trustlen, xp, fp, Fp)


def _update_trust_region(
    f: Callable[[FloatVector], float | tuple[float, FloatVector]],
    xc: FloatVector,
    fc: float,
    gc: FloatVector,
    R: FloatMatrix,
    s: FloatVector,
    tolx: float,
    sclx: FloatVector,
    maxlen: float,
    trustlen: float,
    isnewtstep: bool,
    state: TrustState,
    xp_prev: FloatVector,
    fp_prev: float,
    Fp_prev: FloatVector,
) -> tuple[
    TrustState,
    bool,
    float,
    FloatVector,
    float,
    FloatVector,
    FloatVector,
    float,
    FloatVector,
]:
    r"""Perform trust-region update.

    **References**

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
        Optimization and Nonlinear Equations", SIAM, 1996.

    Parameters
    ----------
    f : Callable[[FloatVector], float | tuple[float, FloatVector]]
        Objective function. For compability with optimization and root-finding algorithms,
        `f` can return either a scalar objective function value or a tuple including the
        scaled norm of the vector root function and the vector root function itself.
    xc : FloatVector
        Current value of the variable vector.
    fc : float
        Current objective function value, `f(xc)`.
    gc : FloatVector
        Current gradient of the objective function, `∇f(xc)`.
    R : FloatMatrix
        Upper-triangular factor of the current Jacobian QR decomposition.
    s : FloatVector
        Step vector.
    tolx : float
        Tolerance for the step size.
    sclx : FloatVector
        Scaling factors for `x`.
    maxlen : float
        Maximum step length.
    trustlen : float
        Current trust region radius.
    isnewtstep : bool
        Flag indicating if the step is a Newton step.
    state : TrustState
        Code indicating the current state of the algorithm.
    xp_prev : FloatVector
        Previous value of `xp`.
    fp_prev : float
        Previous value of `fp`.
    Fp_prev : FloatVector
        Previous value of `Fp`.

    Returns
    -------
    tuple[TrustState, bool, float, FloatVector, float, FloatVector, FloatVector, float, FloatVector]
        `(state, ismaxstep, trustlen, xp, fp, Fp, xp_prev, fp_prev, Fp_prev)`
    """  # noqa: E501
    α = 1e-4
    ismaxstep = False

    steplen = norm(sclx * s)
    slope = np.dot(gc, s)

    xp = xc + s

    res = f(xp)
    fp, Fp = res if isinstance(res, tuple) else (res, np.array([]))

    Δf = fp - fc

    if (state == TrustState.exploratory_success) and (
        (fp >= fp_prev) or (Δf > α * slope)
    ):
        state = TrustState.accepted
        xp, fp, Fp = xp_prev, fp_prev, Fp_prev
        trustlen *= 0.5
    elif Δf >= α * slope:
        rlen = np.max(np.abs(s) / np.maximum(np.abs(xp), 1 / sclx))
        if rlen < tolx:
            state = TrustState.convergence
            xp = xc
        else:
            state = TrustState.rejected
            trustlen = np.clip(
                -slope * steplen / (2 * (Δf - slope)), 0.1 * trustlen, 0.5 * trustlen
            )
    else:
        Δf_pred = slope + 0.5 * norm(R @ s) ** 2
        if (
            (state != TrustState.rejected)
            and (not isnewtstep)
            and (trustlen <= 0.99 * maxlen)
            and ((abs(Δf_pred - Δf) <= 0.1 * abs(Δf)) or (Δf <= slope))
        ):
            state = TrustState.exploratory_success
            xp_prev, fp_prev, Fp_prev = xp, fp, Fp
            trustlen = min(2 * trustlen, maxlen)
        else:
            state = TrustState.accepted
            if steplen >= 0.99 * maxlen:
                ismaxstep = True
            if Δf >= 0.1 * Δf_pred:
                trustlen *= 0.5
            elif Δf <= 0.75 * Δf_pred:
                trustlen = min(2 * trustlen, maxlen)
            else:
                trustlen = trustlen  # no change

    return (state, ismaxstep, trustlen, xp, fp, Fp, xp_prev, fp_prev, Fp_prev)
