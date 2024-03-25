# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from numpy import cbrt, dot, exp, log

from polykin.utils.types import FloatVectorLike

__all__ = ['MULMX2_Perry']

# %% Mixing rules


def MULMX2_Perry(x: FloatVectorLike,
                 mu: FloatVectorLike,
                 hydrocarbons: bool = False,
                 ) -> float:
    r"""Calculate the viscosity of a liquid mixture from the viscosities of the
    pure components using the mixing rules recommended in Perry's.

    For hydrocarbon mixtures:

    $$ \mu_m = \left ( \sum_{i=1}^N x_i \mu_i^{1/3} \right )^3 $$

    and for nonhydrocarbon mixtures:

    $$ \ln{\mu_m} = \sum_{i=1}^N x_i \ln{\mu_i} $$

    **References**

    *   Perry, R. H., D. W. Green, and J. Maloney. Perrys Chemical Engineers
        Handbook, 7th ed. 1999, p. 2-367.

    Parameters
    ----------
    x : FloatVectorLike
        Mole fractions of all components. Unit = mol/mol.
    mu : FloatVectorLike
        Viscosities of all components, $\mu$. Unit = Any.
    hydrocarbons : bool
        Method selection. `True` for hydrocarbon mixtures, `False` for
        nonhydrocarbon mixtures.

    Returns
    -------
    float
        Mixture viscosity, $\mu_m$. Unit = [mu].

    Examples
    --------
    Estimate the viscosity of a 50 mol% styrene/toluene liquid mixture at 20°C.
    >>> from polykin.properties.viscosity import MULMX2_Perry
    >>>
    >>> x = [0.5, 0.5]
    >>> mu = [0.76, 0.59] # cP, from literature
    >>>
    >>> mu_mix = MULMX2_Perry(x, mu)
    >>>
    >>> print(f"{mu_mix:.2f} cP")
    0.67 cP

    """
    x = np.asarray(x)
    mu = np.asarray(mu)

    if hydrocarbons:
        result = dot(x, cbrt(mu))**3
    else:
        result = exp(dot(x, log(mu)))
    return result
