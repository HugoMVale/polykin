# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from typing import overload

import numpy as np
from numpy import exp, sqrt
from numpy import log as ln
from scipy.special import ive, loggamma

from polykin.utils.exceptions import ConvergenceError
from polykin.utils.typing import FloatArray, IntArrayLike

__all__ = [
    "compartmentalization_factor",
    "nbar_Li_Brooks",
    "nbar_Stockmayer_OToole",
    "nbar_Ugelstad",
    "n2bar",
    "SmithEwart_qssa_pdf",
]


@overload
def nbar_Stockmayer_OToole(
    alpha: float,
    m: float,
) -> float: ...


@overload
def nbar_Stockmayer_OToole(
    alpha: FloatArray,
    m: FloatArray,
) -> FloatArray: ...


@overload
def nbar_Stockmayer_OToole(
    alpha: FloatArray,
    m: float,
) -> FloatArray: ...


@overload
def nbar_Stockmayer_OToole(
    alpha: float,
    m: FloatArray,
) -> FloatArray: ...


def nbar_Stockmayer_OToole(
    alpha: float | FloatArray,
    m: float | FloatArray,
) -> float | FloatArray:
    r"""Calculate the average number of radicals per particle according to the
    Stockmayer-O'Toole exact quasi-steady-state solution.

    $$ \bar{n} = \frac{a}{4} \frac{I_m(a)}{I_{m-1}(a)} $$

    where $a=\sqrt{8 \alpha}$, and $I$ is the modified Bessel function of the first kind.

    **References**

    *   Stockmayer, W. H. Note on the Kinetics of Emulsion Polymerization. J. Polym. Sci.
        1957, 24, 314-317.
    *   O'Toole, J. T. Kinetics of Emulsion Polymerization. J. Appl. Polym. Sci.
        1965, 9, 1291-1297.

    Parameters
    ----------
    alpha : float | FloatArray
        Dimensionless entry frequency.
    m : float | FloatArray
        Dimensionless desorption frequency.

    Returns
    -------
    float | FloatArray
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
    alpha, m = np.broadcast_arrays(np.asarray(alpha), np.asarray(m))
    result = np.zeros_like(alpha, dtype=np.float64)

    mask_zero = (alpha == 0) & (m > 0)
    mask_general = ~mask_zero

    if np.any(mask_general):
        a = sqrt(8 * alpha[mask_general])
        result[mask_general] = (
            a / 4 * ive(m[mask_general], a) / ive(m[mask_general] - 1.0, a)
        )

    if result.ndim == 0:
        return float(result)
    else:
        return result


@overload
def nbar_Li_Brooks(
    alpha: float,
    m: float,
) -> float: ...


@overload
def nbar_Li_Brooks(
    alpha: FloatArray,
    m: FloatArray,
) -> FloatArray: ...


@overload
def nbar_Li_Brooks(
    alpha: FloatArray,
    m: float,
) -> FloatArray: ...


@overload
def nbar_Li_Brooks(
    alpha: float,
    m: FloatArray,
) -> FloatArray: ...


def nbar_Li_Brooks(
    alpha: float | FloatArray,
    m: float | FloatArray,
) -> float | FloatArray:
    r"""Calculate the average number of radicals per particle according to the
    Li-Brooks approximate quasi-steady-state solution.

    $$ \bar{n} = \frac{2 \alpha}{m + \sqrt{m^2 +
        \frac{8 \alpha \left( 2 \alpha + m \right)}{2 \alpha + m + 1}}} $$

    This formula agrees well with the exact Stockmayer-O'Toole solution, with a maximum
    deviation of about 4%.

    **References**

    *   Li B-G, Brooks BW. Prediction of the average number of radicals per particle for
        emulsion polymerization. J Polym Sci, Part A: Polym Chem 1993;31:2397-402.

    Parameters
    ----------
    alpha : float | FloatArray
        Dimensionless entry frequency.
    m : float | FloatArray
        Dimensionless desorption frequency.

    Returns
    -------
    float | FloatArray
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
    r"""Calculate the average number of radicals per particle according to the
    Ugelstad-Mørk exact quasi-steady-state solution.

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


@overload
def n2bar(
    alpha: float,
    m: float,
) -> float: ...


@overload
def n2bar(
    alpha: FloatArray,
    m: FloatArray,
) -> FloatArray: ...


@overload
def n2bar(
    alpha: FloatArray,
    m: float,
) -> FloatArray: ...


@overload
def n2bar(
    alpha: float,
    m: FloatArray,
) -> FloatArray: ...


def n2bar(
    alpha: float | FloatArray,
    m: float | FloatArray,
) -> float | FloatArray:
    r"""Calculate the second moment of the normalized radical number distribution
    according to the Stockmayer-O'Toole quasi-steady-state solution.

    $$ \overline{n^2} = \bar{n} \left(1 + \frac{m}{2} \right) + \frac{\alpha}{2} $$

    where $\bar{n}(\alpha, m)$ is the average number of radicals per particle.

    **References**

    *   Stockmayer, W. H. Note on the Kinetics of Emulsion Polymerization. J. Polym. Sci.
        1957, 24, 314-317.
    *   O'Toole, J. T. Kinetics of Emulsion Polymerization. J. Appl. Polym. Sci.
        1965, 9, 1291-1297.

    Parameters
    ----------
    alpha : float | FloatArray
        Dimensionless entry frequency.
    m : float | FloatArray
        Dimensionless desorption frequency.

    Returns
    -------
    float | FloatArray
        Second moment of the normalized radical number distribution.

    Examples
    --------
    Evaluate the average square number of radicals per particle for α=1e-2 and m=1e-4.
    >>> from polykin.kinetics import n2bar
    >>> n2bar_ = n2bar(alpha=1e-2, m=1e-4)
    >>> print(f"n2bar = {n2bar_:.2e}")
    n2bar = 5.07e-01
    """
    nbar = nbar_Stockmayer_OToole(alpha, m)
    return nbar * (1 - m / 2) + alpha / 2


@overload
def compartmentalization_factor(
    alpha: float,
    m: float,
) -> float: ...


@overload
def compartmentalization_factor(
    alpha: FloatArray,
    m: FloatArray,
) -> FloatArray: ...


@overload
def compartmentalization_factor(
    alpha: FloatArray,
    m: float,
) -> FloatArray: ...


@overload
def compartmentalization_factor(
    alpha: float,
    m: FloatArray,
) -> FloatArray: ...


def compartmentalization_factor(
    alpha: float | FloatArray,
    m: float | FloatArray,
) -> float | FloatArray:
    r"""Calculate the compartmentalization factor according to the Stockmayer-O'Toole
    quasi-steady-state solution.

    The compartmentalization factor expresses the reduction of the effective termination
    rate in a compartmentalized system relative to that in a bulk system with the same
    overall radical concentration. It is defined as:

    $$ D_f \equiv \frac{\overline{n(n-1)}}{\bar{n}^2 }
           = \frac{1}{2 \bar{n}} \left(\frac{\alpha}{\bar{n}} - m \right) $$

    where $\bar{n}(\alpha, m)$ is the average number of radicals per particle.

    **References**

    *   Wulkow, M.; Richards, J. R. Evaluation of the Chain Length Distribution in
        Free-Radical Emulsion Polymerization—The Compartmentalization Problem.
        Ind. Eng. Chem. Res. 2014, 53, 7275-7295.

    Parameters
    ----------
    alpha : float | FloatArray
        Dimensionless entry frequency.
    m : float | FloatArray
        Dimensionless desorption frequency.

    Returns
    -------
    float | FloatArray
        Compartmentalization factor.

    Examples
    --------
    Evaluate the compartmentalization factor for α=1e-2 and m=1e-4.
    >>> from polykin.kinetics import compartmentalization_factor
    >>> Df = compartmentalization_factor(alpha=1e-2, m=1e-4)
    >>> print(f"Df = {Df:.2e}")
    Df = 1.97e-02
    """
    nbar = nbar_Stockmayer_OToole(alpha, m)
    return (alpha / nbar - m) / (2 * nbar)


@overload
def SmithEwart_qssa_pdf(
    n: int,
    alpha: float,
    m: float,
) -> float: ...


@overload
def SmithEwart_qssa_pdf(
    n: IntArrayLike,
    alpha: float,
    m: float,
) -> FloatArray: ...


def SmithEwart_qssa_pdf(
    n: int | IntArrayLike,
    alpha: float,
    m: float,
) -> float | FloatArray:
    r"""Probability distribution of the number of radicals per particle according to the
    Smith-Ewart quasi-steady-state approximation.

    The number fraction of particles containing $n$ radicals is given by:

    $$ p_n = \frac{2^{(m-1)/2}}{I_{m-1}((8\alpha)^{1/2})}
             \frac{\alpha^{n/2}}{n!} I_{m-1+n}(2 \alpha^{1/2}) $$

    where $I$ is the modified Bessel function of the first kind.

    Parameters
    ----------
    n : int | IntArrayLike
        Number of radicals per particle (≥0).
    alpha : float
        Dimensionless entry frequency.
    m : float
        Dimensionless desorption frequency.

    Returns
    -------
    float | FloatArray
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
