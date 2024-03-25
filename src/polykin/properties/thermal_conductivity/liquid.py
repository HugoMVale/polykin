# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np

from polykin.properties import quadratic_mixing
from polykin.utils.types import FloatVectorLike

__all__ = ['KLMX2_Li']

# %% Mixing rules


def KLMX2_Li(w: FloatVectorLike,
             k: FloatVectorLike,
             rho: FloatVectorLike,
             ) -> float:
    r"""Calculate the thermal conductivity of a liquid mixture from the
    thermal conductivities of the pure components using the Li mixing rule.

    $$ \begin{aligned}
        k_m &= \sum_{i=1}^N \sum_{j=1}^N \phi_i \phi_j k_{ij} \\
        k_{ij} &= \frac{2}{\frac{1}{k_i} + \frac{1}{k_j}} \\
        \phi_i &= \frac{\frac{w_i}{\rho_i}}
                       {\sum_{j=1}^N \frac{w_j}{\rho_j}}
    \end{aligned} $$

    !!! note

        In this equation, the units of mass fraction $w_i$ and density $\rho_i$
        are arbitrary, as they cancel out when considering the ratio of the
        numerator to the denominator.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 565.

    Parameters
    ----------
    w : FloatVectorLike
        Mass fractions of all components. Unit = Any.
    k : FloatVectorLike
        Thermal conductivities of all components. Unit = Any.
    rho : FloatVectorLike
        Densities of all components, $\rho$. Unit = Any.

    Returns
    -------
    float
        Mixture thermal conductivity, $k_m$. Unit = [k].

    Examples
    --------
    Estimate the thermal conductivity of a 50 wt% styrene/isoprene liquid
    mixture at 20°C.
    >>> from polykin.properties.thermal_conductivity import KLMX2_Li
    >>> import numpy as np
    >>>
    >>> w = [0.5, 0.5]
    >>> k = [0.172, 0.124]    # W/(m·K), from literature
    >>> rho = [0.909, 0.681]  # kg/L
    >>>
    >>> k_mix = KLMX2_Li(w, k, rho)
    >>>
    >>> print(f"{k_mix:.2e} W/(m·K)")
    1.43e-01 W/(m·K)
    """

    w = np.asarray(w)
    k = np.asarray(k)
    rho = np.asarray(rho)

    phi = w/rho
    phi /= phi.sum()
    K = 2 / (1/k + 1/k[:, np.newaxis])

    return quadratic_mixing(phi, K)
