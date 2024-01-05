# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from numpy import cbrt, dot, exp, log

from polykin.utils.types import FloatVector

__all__ = ['MULMX2_Perry']

# %% Mixing rules


def MULMX2_Perry(x: FloatVector,
                 mu: FloatVector,
                 hydrocarbons: bool = False,
                 ) -> float:
    r"""Calculate the viscosity of a liquid mixture from the viscosities of the
    pure components using the mixing rules recommended in Perry's.

    For hydrocarbon mixtures:

    $$ \mu_m = \left ( \sum_i x_i \mu_i^{1/3} \right )^3 $$

    and for nonhydrocarbon mixtures:

    $$ \ln{\mu_m} = \sum_i x_i \ln{\mu_i} $$

    References
    ----------
    * Perry, R. H., D. W. Green, and J. Maloney. Perrys Chemical Engineers
    Handbook, 7th ed. 1999, p. 2-367.

    Parameters
    ----------
    x : FloatVector
        Mole fractions of all components. Unit = mol/mol.
    mu : FloatVector
        Viscosities of all components, $\mu$. Unit = Any.
    hydrocarbons : bool
        Method selection. `True` for hydrocarbon mixtures, `False` for
        nonhydrocarbon mixtures.

    Returns
    -------
    float
        Mixture viscosity, $\mu_m$. Unit = [mu].
    """
    if hydrocarbons:
        result = dot(x, cbrt(mu))**3
    else:
        result = exp(dot(x, log(mu)))
    return result
