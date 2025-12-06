# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

import numpy as np

from polykin.utils.types import FloatArray, FloatOrArrayLike

__all__ = ['convert_polymer_standards']


def convert_polymer_standards(
    M1: FloatOrArrayLike,
    K1: float,
    K2: float,
    a1: float,
    a2: float
) -> FloatArray:
    r"""Convert a molar mass from a given polymer standard to another using the 
    respective Mark-Houwink parameters.

    The conversion from a polymer standard 1 to a polymer standard 2 is given
    by:

    $$ M_2 = \left(\frac{K_1}{K_2}\right)^{\frac{1}{1 + a_2}} 
            M_1^{\frac{1 + a_1}{1 + a_2}} $$

    where $M_i$ is the molar mass in standard $i$, and $K_i$ and $a_i$ are the
    Mark-Houwink parameters for standard $i$.

    !!! tip

        This tranformation is linear in terms of the logarithm of the molar
        mass, i.e., $d \ln M_2/d \ln M_1 = \frac{1 + a_1}{1 + a_2}$. This means
        that a GPC distribution can be converted from one standard to another
        by applying this transformation to the x-axis. If you need to convert a
        number or weight distribution, then the y-axis must also be converted
        by a suitable approach. 

    Parameters
    ----------
    M1 : FloatOrArrayLike
        Molar mass in standard 1.
    K1 : float
        Mark-Houwink coefficient for standard 1.
    K2 : float
        Mark-Houwink coefficient for standard 2.
    a1 : float
        Mark-Houwink exponent for standard 1.
    a2 : float
        Mark-Houwink exponent for standard 2.

    Returns
    -------
    FloatArray
        Molar mass in standard 2.

    Examples
    --------
    A sample of PMMA was mesured to have a molar mass of 100 kg/mol in PS
    equivalent weight. What is the sample molar mass in actual PMMA weight?

    >>> from polykin.distributions import convert_polymer_standards
    >>> a1 = 0.77      # PS in THF
    >>> K1 = 6.82e-3   # PS in THF
    >>> a2 = 0.69      # PMMA in THF
    >>> K2 = 1.28e-2   # PMMA in THF
    >>> M2 = convert_polymer_standards(100, K1, K2, a1, a2) 
    >>> print(f"{M2:.2f} kg/mol")
    85.68 kg/mol
    """
    M1 = np.asarray(M1, dtype=np.float64)
    return (K1/K2)**(1/(1 + a2)) * M1**((1 + a1)/(1 + a2))
