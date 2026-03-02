# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import exp, sqrt
from numpy import log as ln
from scipy.special import ive, loggamma

from polykin.utils.exceptions import ConvergenceError
from polykin.utils.typing import FloatVector, IntVectorLike

__all__ = [
    "nbar_Li_Brooks",
    "nbar_Stockmayer_OToole",
    "nbar_Ugelstad",
    "SmithEwart_qssa_pdf",
]


def nbar_Stockmayer_OToole(alpha: float, m: float) -> float:
    r"""Average number of radicals per particle according to the
    Stockmayer-O'Toole exact solution.

    $$ \bar{n} = \frac{a}{4} \frac{I_m(a)}{I_{m-1}(a)} $$

    where $a=\sqrt{8 \alpha}$, and $I$ is the modified Bessel function of the
    first kind.

    **References**

    *   O'Toole JT. Kinetics of emulsion polymerization. J Appl Polym Sci 1965; 9:1291-7.

    Parameters
    ----------
    alpha : float
        Dimensionless entry frequency.
    m : float
        Dimensionless desorption frequency.

    Returns
    -------
    float
        Average number of radicals per particle.

    See Also
    --------
    * [`nbar_Li_Brooks`](nbar_Li_Brooks.md):
      Approximate solution; typically an order of magnitude faster.
    * [`nbar_Ugelstad`](nbar_Ugelstad.md):
      Alternative exact solution based on continued fractions.

    Examples
    --------
    Evaluate the average number of radicals per particle for α=1e-2 and m=1e-4.
    >>> from polykin.kinetics import nbar_Stockmayer_OToole
    >>> nbar = nbar_Stockmayer_OToole(alpha=1e-2, m=1e-4)
    >>> print(f"nbar = {nbar:.2e}")
    nbar = 5.02e-01
    """
    if alpha == 0 and m > 0:
        return 0.0
    else:
        a = sqrt(8 * alpha)
        # Use exponentially scaled Bessel function for numerical robustness
        return (a / 4) * ive(m, a) / ive(m - 1, a)


def nbar_Li_Brooks(alpha: float, m: float) -> float:
    r"""Average number of radicals per particle according to the Li-Brooks
    approximate solution.

    $$ \bar{n} = \frac{2 \alpha}{m + \sqrt{m^2 +
        \frac{8 \alpha \left( 2 \alpha + m \right)}{2 \alpha + m + 1}}} $$

    This formula agrees well with the exact Stockmayer-O'Toole solution,
    with a maximum deviation of about 4%.

    **References**

    *   Li B-G, Brooks BW. Prediction of the average number of radicals per particle for
        emulsion polymerization. J Polym Sci, Part A: Polym Chem 1993;31:2397-402.

    Parameters
    ----------
    alpha : float
        Dimensionless entry frequency.
    m : float
        Dimensionless desorption frequency.

    Returns
    -------
    float
        Average number of radicals per particle.


    See Also
    --------
    * [`nbar_Stockmayer_OToole`](nbar_Stockmayer_OToole.md):
      Preferred exact solution; simpler and numerically robust.
    * [`nbar_Ugelstad`](nbar_Ugelstad.md):
      Alternative exact solution based on continued fractions.

    Examples
    --------
    Evaluate the average number of radicals per particle for α=1e-2 and m=1e-4.
    >>> from polykin.kinetics import nbar_Li_Brooks
    >>> nbar = nbar_Li_Brooks(alpha=1e-2, m=1e-4)
    >>> print(f"nbar = {nbar:.2e}")
    nbar = 5.02e-01
    """
    return (
        2 * alpha / (m + sqrt(m**2 + 8 * alpha * (2 * alpha + m) / (2 * alpha + m + 1)))
    )


def nbar_Ugelstad(
    alpha: float,
    m: float,
    *,
    tol: float = 1e-10,
    maxiter: int = 100,
) -> float:
    r"""Average number of radicals per particle according to the Ugelstad-Mørk exact
    solution.

    $$ \bar{n} = \frac{1}{2}
      \frac{2 \alpha}{m +
      \frac{2 \alpha}{m + 1 +
      \frac{2 \alpha}{m + 2 + \frac{2 \alpha}{m + 3 + ...}}}} $$

    The continued fraction is evaluated using Lentz's modified method.

    **References**

    *   Ugelstad J., Mörk P.C. and Aasen J.O., Kinetics of emulsion polymerization.
        J. Polym. Sci. A-1 Polym. Chem., 1967; 5:2281-2288.

    Note
    ----
    Included mainly for historical completeness. Today, the Stockmayer-O'Toole solution is
    preferred: it is equally exact, simpler to evaluate, and numerically stable for large
    α and m.

    Parameters
    ----------
    alpha : float
        Dimensionless entry frequency.
    m : float
        Dimensionless desorption frequency.
    tol : float
        Tolerance for convergence of the continued fraction evaluation.
    maxiter : int
        Maximum number of iterations for the continued fraction evaluation.

    Returns
    -------
    float
        Average number of radicals per particle.


    See Also
    --------
    * [`nbar_Stockmayer_OToole`](nbar_Stockmayer_OToole.md):
      Preferred exact solution; simpler and numerically robust.
    * [`nbar_Li_Brooks`](nbar_Li_Brooks.md):
      Approximate solution; typically an order of magnitude faster.

    Examples
    --------
    Evaluate the average number of radicals per particle for α=1e-2 and m=1e-4.
    >>> from polykin.kinetics import nbar_Ugelstad
    >>> nbar = nbar_Ugelstad(alpha=1e-2, m=1e-4)
    >>> print(f"nbar = {nbar:.2e}")
    nbar = 5.02e-01
    """
    tiny = 1e-30
    a = 2.0 * alpha
    delta = 1e99

    # Initial denominator term (b0 = m)
    f = max(m, tiny)
    C = f
    D = 0.0

    for k in range(1, maxiter + 1):
        b = m + k

        D = 1.0 / max(b + a * D, tiny)
        C = max(b + a / C, tiny)

        delta = C * D
        f *= delta

        if abs(delta - 1.0) < tol:
            return alpha / f

    raise ConvergenceError(
        f"Lentz method failed to converge after {maxiter} iterations "
        f"(alpha={alpha}, m={m}, tol={tol}). "
        f"Final |delta-1|={abs(delta - 1):.3e}, F={f:.6e}"
    )


def SmithEwart_qssa_pdf(
    n: int | IntVectorLike,
    alpha: float,
    m: float,
) -> float | FloatVector:
    r"""Probability distribution of the number of radicals per particle according to the
    Smith-Ewart quasi-steady-state approximation.

    The number fraction of particles containing $n$ radicals is given by:

    $$ p_n = \frac{2^{(m-1)/2}}{I_{m-1}((8\alpha)^{1/2})}
             \frac{\alpha^{n/2}}{n!} I_{m-1+n}(2 \alpha^{1/2}) $$

    where $I$ is the modified Bessel function of the first kind.

    Parameters
    ----------
    n : int | IntVectorLike
        Number of radicals per particle (≥0).
    alpha : float
        Dimensionless entry frequency.
    m : float
        Dimensionless desorption frequency.

    Returns
    -------
    float | FloatVector
        Number probability distribution.
    """
    scalar_input = np.isscalar(n)
    n = np.atleast_1d(n).astype(np.int_)

    arg_num = 2 * sqrt(alpha)
    arg_den = sqrt(8 * alpha)

    ln_prefactor = ((m - 1) / 2) * ln(2) - (ln(ive(m - 1, arg_den)) + arg_den)
    ln_factor = (
        (n / 2) * ln(alpha) - loggamma(n + 1) + (ln(ive(m - 1 + n, arg_num)) + arg_num)
    )
    pn = exp(ln_prefactor + ln_factor)

    if scalar_input:
        return pn[0]
    else:
        return pn
