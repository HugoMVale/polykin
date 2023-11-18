# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector

import numpy as np

__all__ = ['MULMX2_perry']


def MULMX2_perry(x: FloatVector,
                 visc: FloatVector,
                 hydrocarbons: bool = False,
                 ) -> float:
    r"""Calculate the viscosity of a liquid mixture from the viscosities of the
    pure components using the mixing rules recommended in Perry's.

    For hydrocarbon mixtures:

    $$ \eta_m = \left ( \sum_i x_i \eta_i^{1/3} \right )^3 $$

    and for nonhydrocarbon mixtures:

    $$ \eta_m = \sum_i x_i \ln{\eta_i} $$

    where the meaning of the parameters is as defined below.

    Reference:

    * Perry, R. H., D. W. Green, and J. Maloney. Perrys Chemical Engineers
    Handbook, 7th ed. 1999, p. 2-367.

    Parameters
    ----------
    x : FloatVector
        Mole fractions of all components. Unit = Any.
    visc : FloatVector
        Viscosities of all components, $\eta$. Unit = Any.
    hydrocarbons : bool
        Method selection. `True` for hydrocarbon mixtures, `False` for
        nonhydrocarbon mixtures.

    Returns
    -------
    float
        Mixture viscosity, $\eta_m$. Unit = [visc].
    """
    if hydrocarbons:
        result = np.dot(x, np.cbrt(visc))**3
    else:
        result = np.exp(np.dot(x, np.log(visc)))
    return result
