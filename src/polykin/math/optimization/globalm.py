# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

import math
from collections.abc import Callable
from enum import IntEnum

import numpy as np
from numpy.linalg import norm

from polykin.utils.typing import FloatMatrix, FloatVector

__all__ = ["line_search", "dogleg"]


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

    This function performs a line search along the quasi-Newton direction to find a step
    size that satisfies the Armijo condition.

    **References**

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
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
    slope = np.dot(gc, p)

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
                if np.isclose(a, 0.0):
                    λtemp = -slope / (2 * b)
                else:
                    λtemp = (-b + math.sqrt(b**2 - 3 * a * slope)) / (3 * a)
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

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
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
                cauchylen = α * math.sqrt(α) / β
                η = 0.2 + (0.8 * α**2 / (β * abs(np.dot(gc, p))))
                v = η * (p * sclx) - sSD
                if trustlen <= 0.0:
                    trustlen = min(cauchylen, maxlen)

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

    *   J.E. Dennis Jr., R.B. Schnabel, "Numerical Methods for Unconstrained
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
    """  # noqa: E501
    α = 1e-4
    ismaxstep = False

    steplen = norm(sclx * s)
    slope = np.dot(gc, s)

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
