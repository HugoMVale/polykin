# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from scipy.constants import R, N_A

__all__ = ['KVPC_stiel_thodos']


def KVPC_stiel_thodos(rhor: float,
                      M: float,
                      Tc: float,
                      Pc: float,
                      Zc: float
                      ) -> float:
    r"""Calculate the effect of pressure (or density) on gas thermal
    conductivity using the method of Stiel and Thodos for nonpolar gases.

    $$ \left( k-k_0 \right) \Gamma Z_c^5 = f(\rho_r) $$

    where $k$ is the dense gas thermal conductivity, $k^\circ$ is the
    low-pressure thermal conductivtiy, $\Gamma$ is a group of constants, $Z_c$
    is the critical compressibility factor, and
    $\rho_r = \rho / \rho_c = V_c / V$ is the reduced gas density.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 521.

    Parameters
    ----------
    rhor : float
        Reduced gas density, $\rho_r$.
    M : float
        Molar mass. Unit = kg/mol.
    Tc : float
        Critical temperature. Unit = K.
    Pc : float
        Critical pressure. Unit = Pa.
    Zc : float
        Critical compressibility factor.

    Returns
    -------
    float
        Residual thermal conductivity, $(k - k_0)$. Unit = W/(m·K).
    """

    gamma = ((Tc * M**3 * N_A**2)/(R**5 * Pc**4))**(1/6)

    if rhor < 0.5:
        a = 1.22e-2*(np.exp(0.535*rhor) - 1)
    elif rhor >= 0.5 and rhor < 2.0:
        a = 1.14e-2*(np.exp(0.67*rhor) - 1.069)
    elif rhor >= 2.0 and rhor < 2.8:
        a = 2.60e-3*(np.exp(1.155*rhor) + 2.016)
    else:
        raise ValueError("Invalid `rhor` input. Valid range: `rhor` < 2.8.")

    return a/(gamma * Zc**5)
