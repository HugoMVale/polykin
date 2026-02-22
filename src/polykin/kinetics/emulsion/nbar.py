# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import sqrt
from scipy.special import ive

from polykin.utils.exceptions import ConvergenceError

__all__ = [
    "nbar_Li_Brooks",
    "nbar_Stockmayer_OToole",
    "nbar_Ugelstad_Mork",
]


def nbar_Stockmayer_OToole(alpha: float, m: float) -> float:
    r"""Average number of radicals per particle according to the
    Stockmayer-O'Toole solution.

    $$ \bar{n} = \frac{a}{4} \frac{I_m(a)}{I_{m-1}(a)} $$

    where $a=\sqrt{8 \alpha}$, and $I$ is the modified Bessel function of the
    first kind.

    **References**

    *   O'Toole JT. Kinetics of emulsion polymerization. J Appl Polym Sci 1965;
        9:1291-7.

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

    Examples
    --------
    Evaluate the average number of radicals per particle for α=1e-2 and m=1e-4.
    >>> from polykin.kinetics import nbar_Stockmayer_OToole
    >>> nbar = nbar_Stockmayer_OToole(alpha=1e-2, m=1e-4)
    >>> print(f"nbar = {nbar:.2e}")
    nbar = 5.02e-01
    """
    if alpha == 0:
        return 0.0
    else:
        a = sqrt(8 * alpha)
        return (a / 4) * ive(m, a) / ive(m - 1, a)


def nbar_Li_Brooks(alpha: float, m: float) -> float:
    r"""Average number of radicals per particle according to the Li-Brooks
    approximation.

    $$ \bar{n} = \frac{2 \alpha}{m + \sqrt{m^2 +
        \frac{8 \alpha \left( 2 \alpha + m \right)}{2 \alpha + m + 1}}} $$

    This formula agrees well with the exact Stockmayer-O'Toole solution,
    with a maximum deviation of about 4%.

    **References**

    *   Li B-G, Brooks BW. Prediction of the average number of radicals per
        particle for emulsion polymerization. J Polym Sci, Part A: Polym Chem
        1993;31:2397-402.

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


def nbar_Ugelstad_Mork(
    alpha: float,
    m: float,
    tol: float = 1e-10,
    maxiter: int = 100,
) -> float:
    r"""Average number of radicals per particle according to the Ugelstad-Mørk
    approximation.

    $$ \bar{n} = \frac{1}{2}
      \frac{2 \alpha}{m +
      \frac{2 \alpha}{m + 1 +
      \frac{2 \alpha}{m + 2 + \frac{2 \alpha}{m + 3 + ...}}}} $$

    The continued fraction is evaluated using Lentz's modified method.

    **References**

    *   Ugelstad J., Mörk P.C. and Aasen J.O., Kinetics of emulsion
        polymerization. J. Polym. Sci. A-1 Polym. Chem., 1967; 5:2281-2288.

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

    Examples
    --------
    Evaluate the average number of radicals per particle for α=1e-2 and m=1e-4.
    >>> from polykin.kinetics import nbar_Ugelstad_Mork
    >>> nbar = nbar_Ugelstad_Mork(alpha=1e-2, m=1e-4)
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
