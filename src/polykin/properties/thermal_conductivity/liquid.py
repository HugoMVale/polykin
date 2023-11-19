# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector

import numpy as np

__all__ = ['KLMX2_li']


def KLMX2_li(w: FloatVector,
             k: FloatVector,
             rho: FloatVector,
             ) -> float:
    r"""Calculate the thermal conductivity of a liquid mixture from the
    thermal conductivities of pure components using the Li mixing rule.

    $$ k_m = \sum_i \sum_j \phi_i \phi_j k_{ij} $$

    with:

    $$ k_{ij} = \frac{2}{\frac{1}{k_i} + \frac{1}{k_j}} $$

    $$ \phi_i = \frac{\frac{w_i}{\rho_i}}{\sum_j \frac{w_j}{\rho_j}} $$

    where the meaning of the parameters is as defined below.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 565.

    Parameters
    ----------
    w : FloatVector
        Mass fractions of all components. Unit = Any.
    k : FloatVector
        Thermal conductivities of all components. Unit = Any.
    rho : FloatVector
        Densities of all components, $\rho$. Unit = Any.

    Returns
    -------
    float
        Mixture thermal conductivity, $k_m$. Unit = [k].
    """

    phi = w/rho
    phi /= phi.sum()

    # result = 0.
    # for i in range(len(phi)):
    #     for j in range(len(phi)):
    #         result += phi[i]*phi[j]*2/(1/k[i] + 1/k[j])

    return np.sum(np.outer(phi, phi) * 2 / (1/k + 1/k[:, np.newaxis]))
