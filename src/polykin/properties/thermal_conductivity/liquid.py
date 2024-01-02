# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np

from polykin.properties import quadratic_mixing
from polykin.types import FloatVector

__all__ = ['KLMX2_Li']

# %% Mixing rules


def KLMX2_Li(w: FloatVector,
             k: FloatVector,
             rho: FloatVector,
             ) -> float:
    r"""Calculate the thermal conductivity of a liquid mixture from the
    thermal conductivities of the pure components using the Li mixing rule.

    $$ \begin{aligned}
        k_m &= \sum_i \sum_j \phi_i \phi_j k_{ij} \\
        k_{ij} &= \frac{2}{\frac{1}{k_i} + \frac{1}{k_j}} \\
       \phi_i &= \frac{\frac{w_i}{\rho_i}}{\sum_j \frac{w_j}{\rho_j}}
    \end{aligned} $$

    !!! note

        In this equation, the units of mass fraction $w_i$ and density $\rho_i$
        are arbitrary, as they cancel out when considering the ratio of the
        numerator to the denominator.

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
    K = 2 / (1/k + 1/k[:, np.newaxis])
    return quadratic_mixing(phi, K)
