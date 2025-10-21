import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy import dot, exp, log

from polykin.math import fzero_brent
from polykin.utils.exceptions import ConvergenceError, ConvergenceWarning
from polykin.utils.types import FloatVector
from polykin.utils.math import eps

__all__ = [
    'FlashResult',
    'flash2_PV',
    'flash2_PT',
    'residual_Rachford_Rice',
    'solve_Rachford_Rice'
]


def _bubble_residual(
    Kcalc: Callable[[float, float, FloatVector, FloatVector], FloatVector],
    T: float,
    P: float,
    x: FloatVector,
    y0: FloatVector,
    maxiter: int,
    atol_y: float,
    alpha_y: float
) -> tuple:

    # Inner y-loop
    y = y0
    for _ in range(maxiter):
        yold = y.copy()
        K = Kcalc(T, P, x, y)
        ynew = K*x
        ynew /= ynew.sum()
        if np.allclose(ynew, yold, atol=atol_y):
            break
        y = alpha_y*ynew + (1 - alpha_y)*yold
    else:
        raise ConvergenceError(
            f"Inner y-loop failed to converge after {maxiter} iterations.")

    # Bubble condition
    F = dot(x, K) - 1.0

    return F, y, K


def _dew_residual(
    Kcalc: Callable[[float, float, FloatVector, FloatVector], FloatVector],
    T: float,
    P: float,
    x0: FloatVector,
    y: FloatVector,
    maxiter: int,
    atol_x: float,
    alpha_x: float
) -> tuple[float, FloatVector, FloatVector]:

    # Inner y-loop
    x = x0
    for _ in range(maxiter):
        xold = x.copy()
        K = Kcalc(T, P, x, y)
        xnew = y/K
        xnew /= xnew.sum()
        if np.allclose(xnew, xold, atol=atol_x):
            break
        x = alpha_x*xnew + (1 - alpha_x)*xold
    else:
        raise ConvergenceError(
            f"Inner x-loop failed to converge after {maxiter} iterations.")

    # Bubble condition
    F = dot(y, 1/K) - 1.0

    return F, x, K


def bubble_P(
    T: float,
    x: FloatVector,
    Kcalc: Callable[[float, float, FloatVector, FloatVector], FloatVector],
    rtol_P: float = 1e-4,
    atol_y: float = 1e-6,
    atol_F: float = 1e-8,
    alpha_y: float = 1.0,
    maxiter: int = 50,
    P0: float = 1e5,
    Pmin: float = 0.0,
    Pmax: float = 100e5
) -> tuple[float, FloatVector, FloatVector]:

    def bubble_residual(P, y0):
        return _bubble_residual(Kcalc, T, P, x, y0, maxiter, atol_y, alpha_y)

    # Solve for P using constrained secant method
    P1, P2 = P0, P0*1.01
    y = x.copy()
    F1, y, _ = bubble_residual(P1, y)
    F2, _, _ = bubble_residual(P2, y)
    for _ in range(maxiter):
        P = P2 - F2*(P2 - P1)/(F2 - F1)
        P = np.clip(P, Pmin, Pmax)
        F, y, K = bubble_residual(P, y)
        if (abs(P - P2) < P*rtol_P) or abs(F) < atol_F:
            break
        P1, P2 = P2, P
        F1, F2 = F2, F
    else:
        raise ConvergenceError(
            f"`bubble_P` did not converge after {maxiter} iterations."
            f"Last iteration values: P={P}, y={y}")

    return P, y, K


def dew_P(
    T: float,
    y: FloatVector,
    Kcalc: Callable[[float, float, FloatVector, FloatVector], FloatVector],
    rtol_P: float = 1e-4,
    atol_x: float = 1e-6,
    atol_F: float = 1e-8,
    alpha_x: float = 1.0,
    maxiter: int = 50,
    P0: float = 1e5,
    Pmin: float = 0.0,
    Pmax: float = 100e5
) -> tuple[float, FloatVector, FloatVector]:

    def dew_residual(P, x0):
        return _dew_residual(Kcalc, T, P, x0, y, maxiter, atol_x, alpha_x)

    # Solve for P using constrained secant method
    P1, P2 = P0, P0*1.01
    x = y.copy()
    F1, x, _ = dew_residual(P1, x)
    F2, _, _ = dew_residual(P2, x)
    for _ in range(maxiter):
        P = P2 - F2*(P2 - P1)/(F2 - F1)
        P = np.clip(P, Pmin, Pmax)
        F, x, K = dew_residual(P, x)
        if (abs(P - P2) < P*rtol_P) or abs(F) < atol_F:
            break
        P1, P2 = P2, P
        F1, F2 = F2, F
    else:
        raise ConvergenceError(
            f"`bubble_P` did not converge after {maxiter} iterations."
            f"Last iteration values: P={P}, x={x}")

    return P, x, K


# %% Flash solvers


@dataclass(frozen=True)
class FlashResult():
    """Flash result dataclass.

    Attributes
    ----------
    success : bool
        Flag indicating if the solver converged.
    T : float
        Temperature (K).
    P : float
        Pressure (Pa).
    F : float
        Feed mole flowrate (mol/s).
    L : float
        Liquid mole flowrate (mol/s).
    V : float
        Vapor mole flowrate (mol/s).
    beta : float
        Vapor phase fraction (mol/mol). 
    z : FloatVector
        Feed mole fractions (mol/mol).
    x : FloatVector
        Liquid mole fractions (mol/mol).    
    y : FloatVector
        Vapor mole fractions (mol/mol).
    K : FloatVector
        K-values.
    """
    success: bool
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


def flash2_PT(
    F: float,
    z: FloatVector,
    P: float,
    T: float,
    Kcalc: Callable[[float, float, FloatVector, FloatVector], FloatVector],
    *,
    beta0: float | None = None,
    maxiter: int = 50,
    atol_inner: float = 1e-9,
    rtol_outer: float = 1e-6,
    alpha_outer: float = 1.0,
) -> FlashResult:
    r"""Solve a 2-phase flash problem at given temperature and pressure.

    **References**

    *  M.L. Michelsen and J. Mollerup, "Thermodynamic models: Fundamentals and
       computational aspects", Tie-Line publications, 2nd edition, 2007.

    Parameters
    ----------
    F : float
        Feed mole flowrate (mol/s).
    z : FloatVector
        Feed mole fractions (mol/mol).
    T : float
        Temperature (K).
    P : float
        Pressure (Pa).
    Kcalc : Callable[[float, float, FloatVector, FloatVector], FloatVector]
        Function to calculate K-values, with signature `Kcalc(T, P, x, y)`.
    beta0 : float | None
        Initial guess for vapor phase fraction.
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

    # Initial guesses
    z = z/z.sum()
    x = z
    y = z
    beta = np.nan
    K = Kcalc(T, P, z, z)

    success = False
    for _ in range(maxiter):

        # Find beta
        sol = solve_Rachford_Rice(K, z, beta0, maxiter, atol_inner)
        beta = sol.beta
        if not sol.success:
            warnings.warn(
                f"Inner Rachford-Rice loop did not converge after {maxiter} iterations.\n"
                f"Solution: {sol}.",
                ConvergenceWarning)

        # Compute x, y
        x = z/(1 + beta*(K - 1))
        y = K*x
        x /= x.sum()
        y /= y.sum()

        # Update K
        K_old = K.copy()
        K_new = Kcalc(T, P, x, y)

        # Check convergence
        k0 = min(k for k in K_new if k > 0)
        if np.allclose(K_new, K_old, atol=k0*rtol_outer):
            success = True
            break
        else:
            K = exp(alpha_outer*log(K_new) + (1 - alpha_outer)*log(K_old))
            beta0 = beta

    else:
        warnings.warn(
            f"Outer loop did not converge after {maxiter} iterations.",
            ConvergenceWarning)

    # Overall balance
    V = F*beta
    L = F - V

    return FlashResult(success, T, P, F, L, V, beta, z, x, y, K)


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
    beta0: float | None = None,
    maxiter: int = 50,
    atol_res: float = 1e-9
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
        Feed mole fractions (mol/mol).
    beta0 : float | None
        Initial guess for vapor phase fraction (mol/mol).
    maxiter : int
        Maximum number of iterations.
    atol_res : float
        Absolute tolerance for residual.

    Returns
    -------
    RachfordRiceResult
        Result.

    See also
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
    beta_min = np.where(K > 1.0, (K*z - 1)/(K - 1), 0.0)
    beta_min = beta_min[(beta_min > 0.0) & (beta_min < 1.0)]
    beta_min = max(beta_min) if len(beta_min) > 0 else 0.0

    beta_max = np.where(K < 1.0, (1 - z)/(1 - K), 1.0)
    beta_max = beta_max[(beta_max > 0.0) & (beta_max < 1.0)]
    beta_max = min(beta_max) if len(beta_max) > 0 else 1.0

    # Initial guess
    if beta0 is not None:
        beta = np.clip(beta0, beta_min, beta_max)
    else:
        beta = (beta_min + beta_max)/2

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
        beta_new = beta - F/dF
        if (beta_new > beta_min) and (beta_new < beta_max):
            beta = beta_new
        else:
            beta = (beta_min + beta_max)/2

    return RachfordRiceResult(success, iter+1, beta, F)


def residual_Rachford_Rice(
    beta: float,
    K: FloatVector,
    z: FloatVector,
    derivative: bool = False
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
        Vapor phase fraction (mol/mol).
    K : FloatVector(N)
        K-values.
    z : FloatVector(N)
        Feed mole fractions (mol/mol).
    derivative : bool
        Flag specifying if the derivative should be returned.

    Returns
    -------
    tuple[float, ...]
        Tuple with residual and derivative, `(F, dF)`.

    See also
    --------
    * [`solve_Rachford_Rice`](solve_Rachford_Rice.md): related method to solve
      the equation. 
    """

    F = np.sum(z*(K - 1)/(1 + beta*(K - 1)))

    if not derivative:
        return (F,)
    else:
        dF = -np.sum(z*(K - 1)**2/(1 + beta*(K - 1))**2)
        return (F, dF)


def flash2_PV(
    F: float,
    z: FloatVector,
    P: float,
    beta: float,
    Kcalc: Callable[[float, float, FloatVector, FloatVector], FloatVector],
    *,
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
        Feed mole flowrate (mol/s).
    z : FloatVector
        Feed mole fractions (mol/mol).
    P : float
        Pressure (Pa).
    beta : float
        Vapor phase fraction (mol/mol).
    Kcalc : Callable[[float, float, FloatVector, FloatVector], FloatVector]
        Function to calculate K-values, with signature `Kcalc(T, P, x, y)`.
    T0 : float
        Initial guess for temperature (K).
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
    # Initial guesses
    z = z/z.sum()
    T = T0
    K = Kcalc(T, P, z, z)
    x = z/(1 + beta*(K - 1))
    y = K*x
    x /= x.sum()
    y /= y.sum()

    # Inner loop objective function
    def fobj(R, u, Kb0, beta, z) -> float:
        p = z/(1 - R + Kb0*R*exp(u))
        return 1 - beta - (1 - R)*p.sum()

    # Outer loop
    Tref = 300.0
    u, Kb, A, B = _parameters_PV(T, P, x, y, beta, Kcalc, Tref, all=True)
    Kb0 = Kb
    success = False
    for _ in range(maxiter):

        v_old = np.concatenate((u, [A]))

        # Inner R-loop
        if abs(beta - 0) <= eps:
            R = 0.0
        elif abs(beta - 1) < eps:
            R = 1.0
        else:
            sol = fzero_brent(lambda R: fobj(R, u, Kb0, beta, z),
                              0.0, 1.0,
                              maxiter=maxiter,
                              tolx=atol_inner,
                              tolf=atol_inner)
            R = sol.x
            if not sol.success:
                warnings.warn(
                    f"Inner R-loop did not converge after {maxiter} iterations.",
                    ConvergenceWarning)

        # Compute x, y
        p = z/(1 - R + Kb0*R*exp(u))
        eup = exp(u)*p
        sum_p = p.sum()
        sum_eup = eup.sum()
        Kb = sum_p/sum_eup
        T = 1/(1/Tref + (log(Kb) - A)/B)
        x = p/sum_p
        y = eup/sum_eup

        # Update u, A
        u, A = _parameters_PV(T, P, x, y, beta, Kcalc, Tref, all=False, B=B)
        v_new = np.concatenate((u, [A]))

        # Check convergence
        v0 = min(k for k in v_new if k > 0)
        if np.allclose(v_new, v_old, atol=v0*rtol_outer):
            success = True
            break

    else:
        warnings.warn(
            f"Outer loop did not converge after {maxiter} iterations.",
            ConvergenceWarning)

    # Overall balances
    V = F*beta
    L = F - V

    return FlashResult(success, T, P, F, L, V, beta, z, x, y, K)


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
        B: float = 0.0
) -> tuple:
    """Calculate volatility parameters for PV flash."""

    # Evaluations at T
    K = Kcalc(T, P, x, y)
    ln_K = log(K)

    t = y/(1 + beta*(K - 1))
    w = t/t.sum()

    ln_Kb = dot(w, ln_K)
    Kb = exp(ln_Kb)
    u = ln_K - ln_Kb

    # Evaluations at T + dT
    if all:
        Tp = T + dT
        Kp = Kcalc(Tp, P, x, y)
        ln_Kp = log(Kp)

        ln_Kbp = dot(w, ln_Kp)
        B = (ln_Kbp - ln_Kb)/(1/Tp - 1/T)

    A = ln_Kb - B*(1/T - 1/Tref)

    if all:
        return u, Kb, A, B
    else:
        return u, A
