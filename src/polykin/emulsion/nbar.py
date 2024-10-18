# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import sqrt
from scipy.special import iv

__all__ = ['nbar_Stockmayer_OToole',
           'nbar_Li_Brooks'
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
    """
    a = sqrt(8*alpha)
    return (a/4)*iv(m, a)/iv(m-1, a)


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
    """
    return 2*alpha/(m + sqrt(m**2 + 8*alpha*(2*alpha + m)/(2*alpha + m + 1)))
