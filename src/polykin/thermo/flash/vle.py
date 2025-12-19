from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy import dot, exp, log

from polykin.math import root_brent, root_newton
from polykin.utils.math import eps
from polykin.utils.tools import colored_bool
from polykin.utils.types import FloatVector

__all__ = [
    "FlashResult",
    "flash2_PV",
    "flash2_PT",
    "flash2_TV",
    "residual_Rachford_Rice",
    "solve_Rachford_Rice",
]


@dataclass(frozen=True)
class FlashResult:
    """Flash result dataclass.

    Attributes
    ----------
    method: str
        Method used to perform the flash calculation.
    success : bool
        Flag indicating if the solver converged.
    message: str
        Description of the exit status.
    T : float
        Temperature [K].
    P : float
        Pressure [Pa].
    F : float
        Feed mole flowrate [mol/s].
    L : float
        Liquid mole flowrate [mol/s].
    V : float
        Vapor mole flowrate [mol/s].
    beta : float
        Vapor phase fraction [mol/mol].
    z : FloatVector
        Feed mole fractions [mol/mol].
    x : FloatVector
        Liquid mole fractions [mol/mol].
    y : FloatVector
        Vapor mole fractions [mol/mol].
    K : FloatVector
        K-values.
    """

    method: str
    success: bool
    message: str
    T: float
    P: float
    F: float
    L: float
    V: float
    beta: float
    z: FloatVector
    x: FloatVector
    y: FloatVector
    K: FloatVector

    def __repr__(self) -> str:
        """Return a string representation of the flash result."""
        return (
            f" method: {self.method}\n"
            f"success: {colored_bool(self.success)}\n"
            f"message: {self.message}\n"
            f"      T: {self.T}\n"
            f"      P: {self.P}\n"
            f"      F: {self.F}\n"
            f"      L: {self.L}\n"
            f"      V: {self.V}\n"
            f"   beta: {self.beta}\n"
            f"      z: {self.z}\n"
            f"      x: {self.x}\n"
            f"      y: {self.y}\n"
            f"      K: {self.K}"
        )


def flash2_PT(
    F: float,
    z: FloatVector,
    P: float,
    T: float,
    Kcalc: Callable[[float, float, FloatVector, FloatVector], FloatVector],
    *,
    x0: FloatVector | None = None,
    y0: FloatVector | None = None,
    beta0: float | None = None,
    maxiter: int = 50,
    atol_inner: float = 1e-9,
    rtol_outer: float = 1e-6,
    alpha_outer: float = 1.0,
) -> FlashResult:
    r"""Solve a 2-phase flash problem at given pressure and temperature.

    **References**

    *  M.L. Michelsen and J. Mollerup, "Thermodynamic models: Fundamentals and
       computational aspects", Tie-Line publications, 2nd edition, 2007.

    Parameters
    ----------
    F : float
        Feed mole flowrate [mol/s].
    z : FloatVector
        Feed mole fractions [mol/mol].
    T : float
        Temperature [K].
    P : float
        Pressure [Pa].
    Kcalc : Callable[[float, float, FloatVector, FloatVector], FloatVector]
        Function to calculate K-values, with signature `Kcalc(T, P, x, y)`.
    x0 : FloatVector | None
        Initial guess for liquid mole fractions [mol/mol].
    y0 : FloatVector | None
        Initial guess for vapor mole fractions [mol/mol].
    beta0 : float | None
        Initial guess for vapor phase fraction [mol/mol].
    maxiter : int
        Maximum number of iterations.
    atol_inner : float
        Absolute tolerance for the inner beta-loop.
    rtol_outer : float
        Relative tolerance for the outer K-values loop.
    alpha_outer : float
        Relaxation factor for the outer K-values loop.

    Returns
    -------
    FlashResult
        Flash result.
    """
    method = "2-Phase PT Flash"
    message = ""
    success = False

    # Initial guesses
    z = z / z.sum()
    x = x0 / x0.sum() if x0 is not None else z
    y = y0 / y0.sum() if y0 is not None else z
    K = Kcalc(T, P, x, y)
    beta = np.clip(beta0, 0.0, 1.0) if beta0 is not None else np.nan

    for _ in range(maxiter):

        # Find beta
        sol = solve_Rachford_Rice(K, z, beta, maxiter=maxiter, atol_res=atol_inner)
        beta = sol.beta
        if not sol.success:
            message = f"Inner Rachford-Rice loop did not converge after {maxiter} iterations. Solution: {sol}."
            break

        # Compute x, y
        x = z / (1 + beta * (K - 1))
        y = K * x
        x /= x.sum()
        y /= y.sum()

        # Update K
        K_old = K.copy()
        K_new = Kcalc(T, P, x, y)

        # Check convergence
        k0 = min(k for k in K_new if k > 0)
        if np.allclose(K_new, K_old, atol=k0 * rtol_outer):
            success = True
            message = "Outer loop converged within specified tolerance."
            break
        else:
            K = exp(alpha_outer * log(K_new) + (1 - alpha_outer) * log(K_old))

    else:
        message = f"Outer loop did not converge after {maxiter} iterations."

    # Overall balance
    V = F * beta
    L = F - V

    return FlashResult(method, success, message, T, P, F, L, V, beta, z, x, y, K)


def flash2_PV(
    F: float,
    z: FloatVector,
    P: float,
    beta: float,
    Kcalc: Callable[[float, float, FloatVector, FloatVector], FloatVector],
    *,
    x0: FloatVector | None = None,
    y0: FloatVector | None = None,
    T0: float = 300.0,
    maxiter: int = 50,
    atol_inner: float = 1e-9,
    rtol_outer: float = 1e-6,
) -> FlashResult:
    r"""Solve a 2-phase flash problem at given pressure and vapor fraction.

    **References**

    *  J.F. Boston, H.I. Britt, "A radically different formulation and solution
       of the single-stage flash problem", Computers & Chemical Engineering,
       Volume 2, Issues 2-3, 1978, p. 109-122,

    Parameters
    ----------
    F : float
        Feed mole flowrate [mol/s].
    z : FloatVector
        Feed mole fractions [mol/mol].
    P : float
        Pressure [Pa].
    beta : float
        Vapor phase fraction [mol/mol].
    Kcalc : Callable[[float, float, FloatVector, FloatVector], FloatVector]
        Function to calculate K-values, with signature `Kcalc(T, P, x, y)`.
    x0 : FloatVector | None
        Initial guess for liquid mole fractions [mol/mol].
    y0 : FloatVector | None
        Initial guess for vapor mole fractions [mol/mol].
    T0 : float
        Initial guess for temperature [K].
    maxiter : int
        Maximum number of iterations.
    atol_inner : float
        Absolute tolerance for the inner R-loop.
    rtol_outer : float
        Relative tolerance for the outer volatility parameters loop.

    Returns
    -------
    FlashResult
        Flash result.
    """
    method = "2-Phase PV Flash"
    message = ""
    success = False

    # Initial guesses
    z = z / z.sum()
    x = x0 / x0.sum() if x0 is not None else z
    y = y0 / y0.sum() if y0 is not None else z
    T = T0
    K = Kcalc(T, P, x, y)
    x = z / (1 + beta * (K - 1))
    y = K * x
    x /= x.sum()
    y /= y.sum()

    # Initialize volatility parameters
    Tref = 300.0
    u, Kb, A, B = _parameters_PV(T, P, x, y, beta, Kcalc, Tref, all=True)
    Kb0 = Kb

    # Outer loop
    for _ in range(maxiter):

        v_old = np.concatenate((u, [A]))

        # Inner R-loop
        if abs(beta - 0) <= eps:
            R = 0.0
        elif abs(beta - 1) <= eps:
            R = 1.0
        else:
            sol = root_brent(
                lambda R: _Rloop(R, u, Kb0, beta, z),
                0.0,
                1.0,
                maxiter=maxiter,
                tolx=atol_inner,
                tolf=atol_inner,
            )
            R = sol.x
            if not sol.success:
                message = f"Inner R-loop did not converge after {maxiter} iterations. Solution: {sol}."
                break

        # Compute x, y, T
        p = z / (1 - R + Kb0 * R * exp(u))
        eup = exp(u) * p
        sum_p = p.sum()
        sum_eup = eup.sum()
        x = p / sum_p
        y = eup / sum_eup
        Kb = sum_p / sum_eup
        T = 1 / (1 / Tref + (log(Kb) - A) / B)

        # Update u, A, K
        u, A, K = _parameters_PV(T, P, x, y, beta, Kcalc, Tref, all=False, B=B)
        v_new = np.concatenate((u, [A]))

        # Check convergence
        v0 = min(vi for vi in v_new if vi > 0.0)
        if np.allclose(v_new, v_old, atol=v0 * rtol_outer):
            success = True
            message = "Outer loop converged within specified tolerance."
            break

    else:
        message = f"Outer loop did not converge after {maxiter} iterations."

    # Overall balances
    V = F * beta
    L = F - V

    return FlashResult(method, success, message, T, P, F, L, V, beta, z, x, y, K)


def flash2_TV(
    F: float,
    z: FloatVector,
    T: float,
    beta: float,
    Kcalc: Callable[[float, float, FloatVector, FloatVector], FloatVector],
    *,
    x0: FloatVector | None = None,
    y0: FloatVector | None = None,
    P0: float = 1e5,
    maxiter: int = 50,
    atol_inner: float = 1e-9,
    rtol_outer: float = 1e-6,
) -> FlashResult:
    r"""Solve a 2-phase flash problem at given temperature and vapor fraction.

    **References**

    *  J.F. Boston, H.I. Britt, "A radically different formulation and solution
       of the single-stage flash problem", Computers & Chemical Engineering,
       Volume 2, Issues 2-3, 1978, p. 109-122,

    Parameters
    ----------
    F : float
        Feed mole flowrate [mol/s].
    z : FloatVector
        Feed mole fractions [mol/mol].
    T : float
        Temperature [K].
    beta : float
        Vapor phase fraction [mol/mol].
    Kcalc : Callable[[float, float, FloatVector, FloatVector], FloatVector]
        Function to calculate K-values, with signature `Kcalc(T, P, x, y)`.
    x0 : FloatVector | None
        Initial guess for liquid mole fractions [mol/mol].
    y0 : FloatVector | None
        Initial guess for vapor mole fractions [mol/mol].
    P0 : float
        Initial guess for pressure [Pa].
    maxiter : int
        Maximum number of iterations.
    atol_inner : float
        Absolute tolerance for the inner R-loop.
    rtol_outer : float
        Relative tolerance for the outer volatility parameters loop.

    Returns
    -------
    FlashResult
        Flash result.
    """
    method = "2-Phase TV Flash"
    message = ""
    success = False

    # Initial guesses
    z = z / z.sum()
    x = x0 / x0.sum() if x0 is not None else z
    y = y0 / y0.sum() if y0 is not None else z
    P = P0
    K = Kcalc(T, P, x, y)
    x = z / (1 + beta * (K - 1))
    y = K * x
    x /= x.sum()
    y /= y.sum()

    # Initialize volatility parameters
    Pref = 1e5
    u, Kb, A, B = _parameters_TV(T, P, x, y, beta, Kcalc, Pref, all=True)
    Kb0 = Kb

    # Outer loop
    for _ in range(maxiter):

        v_old = np.concatenate((u, [A]))

        # Inner R-loop
        if abs(beta - 0) <= eps:
            R = 0.0
        elif abs(beta - 1) <= eps:
            R = 1.0
        else:
            sol = root_brent(
                lambda R: _Rloop(R, u, Kb0, beta, z),
                0.0,
                1.0,
                maxiter=maxiter,
                tolx=atol_inner,
                tolf=atol_inner,
            )
            R = sol.x
            if not sol.success:
                message = f"Inner R-loop did not converge after {maxiter} iterations. Solution: {sol}."
                break

        # Compute x, y
        p = z / (1 - R + Kb0 * R * exp(u))
        eup = exp(u) * p
        sum_p = p.sum()
        sum_eup = eup.sum()
        x = p / sum_p
        y = eup / sum_eup

        # Compute P
        Kb = sum_p / sum_eup
        sol = root_newton(
            lambda P: log(Kb * P) - A - B * (P / Pref),
            x0=exp(A) / Kb,
            tolx=0.0,
            tolf=1e-10,
        )
        if sol.success:
            P = sol.x
        else:
            message = f"Pressure did not converge after {maxiter} iterations. Solution: {sol}."
            break

        # Update u, A, K
        u, A, K = _parameters_TV(T, P, x, y, beta, Kcalc, Pref, all=False, B=B)
        v_new = np.concatenate((u, [A]))

        # Check convergence
        v0 = min(vi for vi in v_new if vi > 0.0)
        if np.allclose(v_new, v_old, atol=v0 * rtol_outer):
            success = True
            message = "Outer loop converged within specified tolerance."
            break

        else:
            message = f"Outer loop did not converge after {maxiter} iterations."

    # Overall balances
    V = F * beta
    L = F - V

    return FlashResult(method, success, message, T, P, F, L, V, beta, z, x, y, K)


@dataclass(frozen=True)
class RachfordRiceResult:
    """Rachford-Rice result dataclass."""

    success: bool
    niter: int
    beta: float
    F: float


def solve_Rachford_Rice(
    K: FloatVector,
    z: FloatVector,
    beta0: float,
    *,
    maxiter: int = 50,
    atol_res: float = 1e-9,
) -> RachfordRiceResult:
    r"""Solve the Rachford-Rice flash residual equation.

    The numerical solution of the Rachford-Rice equation is carried out using
    a combination of the Newton and bisection methods, ensuring efficient and
    robust convergence.

    **References**

    *  M.L. Michelsen and J. Mollerup, "Thermodynamic models: Fundamentals and
       computational aspects", Tie-Line publications, 2nd edition, 2007.

    Parameters
    ----------
    K : FloatVector(N)
        K-values.
    z : FloatVector(N)
        Feed mole fractions [mol/mol].
    beta0 : float
        Initial guess for vapor phase fraction [mol/mol]. If `NaN`, an initial
        guess is automatically computed.
    maxiter : int
        Maximum number of iterations.
    atol_res : float
        Absolute tolerance for residual.

    Returns
    -------
    RachfordRiceResult
        Result.

    See Also
    --------
    * [`residual_Rachford_Rice`](residual_Rachford_Rice.md): related method to
      determine the residual and its derivative.
    """
    # Trivial subcooled and superheated cases
    F0 = residual_Rachford_Rice(0.0, K, z)[0]
    if F0 < 0.0:
        return RachfordRiceResult(True, 0, 0.0, F0)
    F1 = residual_Rachford_Rice(1.0, K, z)[0]
    if F1 > 0.0:
        return RachfordRiceResult(True, 0, 1.0, F1)

    # Bounds on beta
    beta_min = np.where(K > 1.0, (K * z - 1) / (K - 1), 0.0)
    beta_min = beta_min[(beta_min > 0.0) & (beta_min < 1.0)]
    beta_min = max(beta_min) if len(beta_min) > 0 else 0.0
    beta_max = np.where(K < 1.0, (1 - z) / (1 - K), 1.0)
    beta_max = beta_max[(beta_max > 0.0) & (beta_max < 1.0)]
    beta_max = min(beta_max) if len(beta_max) > 0 else 1.0

    # Initial guess
    if not np.isnan(beta0):
        beta = np.clip(beta0, beta_min, beta_max)
    else:
        beta = (beta_min + beta_max) / 2

    # Iteration loop
    success = False
    for iter in range(maxiter):

        F, dF = residual_Rachford_Rice(beta, K, z, derivative=True)

        # Check convergence
        if abs(F) <= atol_res:
            success = True
            break

        # Update bounds
        if F > 0.0:
            beta_min = beta
        else:
            beta_max = beta

        # Update beta (Newton or bisection)
        beta_new = beta - F / dF
        if (beta_new > beta_min) and (beta_new < beta_max):
            beta = beta_new
        else:
            beta = (beta_min + beta_max) / 2

    return RachfordRiceResult(success, iter + 1, beta, F)


def residual_Rachford_Rice(
    beta: float,
    K: FloatVector,
    z: FloatVector,
    derivative: bool = False,
) -> tuple[float, ...]:
    r"""Rachford-Rice flash residual function and its derivative.

    The residual function is defined as:

    $$ F = \sum_i \frac{z_i(K_i - 1)}{1 + \beta(K_i - 1)} $$

    and the derivative with respect to $\beta$ is:

    $$ \frac{\partial F}{\partial \beta} =
       -\sum_i \frac{z_i(K_i - 1)^2}{(1 + \beta(K_i - 1))^2} $$

    where $K$ is the vector of K-values, $z$ is the vector of feed mole
    fractions, and $\beta$ is the vapor phase fraction.

    **References**

    *  Rachford, H.H., and J.D. Rice. "Procedure for Use of Electronic Digital
       Computers in Calculating Flash Vaporization Hydrocarbon Equilibrium",
       J. Pet. Technol. 4 (1952): 19-3.

    Parameters
    ----------
    beta : float
        Vapor phase fraction [mol/mol].
    K : FloatVector(N)
        K-values.
    z : FloatVector(N)
        Feed mole fractions [mol/mol].
    derivative : bool
        Flag specifying if the derivative should be returned.

    Returns
    -------
    tuple[float, ...]
        Tuple with residual and derivative, `(F, dF)`.

    See Also
    --------
    * [`solve_Rachford_Rice`](solve_Rachford_Rice.md): related method to solve
      the equation.
    """
    F = np.sum(z * (K - 1) / (1 + beta * (K - 1)))

    if not derivative:
        return (F,)
    else:
        dF = -np.sum(z * (K - 1) ** 2 / (1 + beta * (K - 1)) ** 2)
        return (F, dF)


def _Rloop(
    R: float,
    u: FloatVector,
    Kb0: float,
    beta: float,
    z: FloatVector,
) -> float:
    """Inner R-loop objective function.

    **References**

    *  J.F. Boston, H.I. Britt, "A radically different formulation and solution
       of the single-stage flash problem", Computers & Chemical Engineering,
       Volume 2, Issues 2-3, 1978, p. 109-122,
    """
    p = z / (1 - R + Kb0 * R * exp(u))
    return 1 - beta - (1 - R) * p.sum()


def _parameters_PV(
    T: float,
    P: float,
    x: FloatVector,
    y: FloatVector,
    beta: float,
    Kcalc: Callable,
    Tref: float,
    dT: float = 1.0,
    all: bool = False,
    B: float = 0.0,
) -> tuple:
    r"""Calculate volatility parameters for PV flash.

    $$ \ln{K_b} = A + B (1/T - 1/T_{ref}) $$

    **References**

    *  J.F. Boston, H.I. Britt, "A radically different formulation and solution
       of the single-stage flash problem", Computers & Chemical Engineering,
       Volume 2, Issues 2-3, 1978, p. 109-122,
    """
    # Evaluations at T
    K = Kcalc(T, P, x, y)
    ln_K = log(K)

    t = y / (1 + beta * (K - 1))
    w = t / t.sum()

    ln_Kb = dot(w, ln_K)
    Kb = exp(ln_Kb)
    u = ln_K - ln_Kb

    # Evaluations at T + dT
    if all:
        Tp = T + dT
        Kp = Kcalc(Tp, P, x, y)
        ln_Kp = log(Kp)

        ln_Kbp = dot(w, ln_Kp)
        B = (ln_Kbp - ln_Kb) / (1 / Tp - 1 / T)

    A = ln_Kb - B * (1 / T - 1 / Tref)

    if all:
        return u, Kb, A, B
    else:
        return u, A, K


def _parameters_TV(
    T: float,
    P: float,
    x: FloatVector,
    y: FloatVector,
    beta: float,
    Kcalc: Callable,
    Pref: float,
    dP: float = 1e3,
    all: bool = False,
    B: float = 0.0,
) -> tuple:
    r"""Calculate volatility parameters for TV flash.

    $$ \ln{(K_b P)} = A + B \ln{(P/P_{ref})} $$

    **References**

    *  J.F. Boston, H.I. Britt, "A radically different formulation and solution
       of the single-stage flash problem", Computers & Chemical Engineering,
       Volume 2, Issues 2-3, 1978, p. 109-122,
    """
    # Evaluations at P
    K = Kcalc(T, P, x, y)
    ln_K = log(K)

    t = y / (1 + beta * (K - 1))
    w = t / t.sum()

    ln_Kb = dot(w, ln_K)
    Kb = exp(ln_Kb)
    u = ln_K - ln_Kb

    # Evaluations at P + dP
    if all:
        Pp = P + dP
        Kp = Kcalc(T, Pp, x, y)
        ln_Kp = log(Kp)

        ln_Kbp = dot(w, ln_Kp)
        B = (ln_Kbp - ln_Kb + log(Pp / P)) * Pref / dP

    A = ln_Kb + log(P) - B * (P / Pref)

    if all:
        return u, Kb, A, B
    else:
        return u, A, K
