# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector

import numpy as np
from scipy.constants import R, N_A

__all__ = ['KVPC_stiel_thodos',
           'KVMXPC_stiel_thodos',
           'KVMX2_wassilijewa']

# %% Pressure correction


def KVPC_stiel_thodos(V: float,
                      M: float,
                      Tc: float,
                      Pc: float,
                      Zc: float
                      ) -> float:
    r"""Calculate the effect of pressure (or density) on the thermal
    conductivity of pure gases using the method of Stiel and Thodos for
    nonpolar components.

    $$ \left( k-k^{\circ} \right) \Gamma Z_c^5 = f(\rho_r) $$

    where $k$ is the dense gas thermal conductivity, $k^\circ$ is the
    low-pressure thermal conductivtiy, $\Gamma$ is a group of constants, $Z_c$
    is the critical compressibility factor, and $\rho_r = V_c / V$ is the
    reduced gas density. This equation is valid in the range
    $0 \leq \rho_r < 2.8$.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 521.

    Parameters
    ----------
    V : float
        Gas molar volume. Unit = m³/mol.
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
        Residual thermal conductivity, $(k - k^{\circ})$. Unit = W/(m·K).
    """

    gamma = ((Tc * M**3 * N_A**2)/(R**5 * Pc**4))**(1/6)
    Vc = Zc*R*Tc/Pc
    rhor = Vc/V

    if rhor < 0.5:
        a = 1.22e-2*(np.exp(0.535*rhor) - 1)
    elif rhor < 2.0:
        a = 1.14e-2*(np.exp(0.67*rhor) - 1.069)
    elif rhor < 2.8:
        a = 2.60e-3*(np.exp(1.155*rhor) + 2.016)
    else:
        raise ValueError("Invalid `rhor` input. Valid range: `rhor` < 2.8.")

    return a/(gamma * Zc**5)


def KVMXPC_stiel_thodos(V: float,
                        y: FloatVector,
                        M: FloatVector,
                        Tc: FloatVector,
                        Pc: FloatVector,
                        Zc: FloatVector,
                        w: FloatVector
                        ) -> float:
    r"""Calculate the effect of pressure (or density) on the thermal
    conductivity of gas mixtures using the method of Stiel and Thodos for
    nonpolar components, combined with the mixing rules of Yorizane.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 536.

    Parameters
    ----------
    V : float
        Gas molar volume. Unit = m³/mol.
    y : FloatVector
        Mole fractions of all components. Unit = mol/mol.
    M : FloatVector
        Molar masses of all components. Unit = kg/mol.
    Tc : FloatVector
        Critical temperatures of all components. Unit = K.
    Pc : FloatVector
        Critical pressures of all components. Unit = Pa.
    Zc : FloatVector
        Critical compressibility factors of all components.
    w : FloatVector
        Acentric factors of all components.

    Returns
    -------
    float
        Residual thermal conductivity, $(k - k^{\circ})$. Unit = W/(m·K).
    """

    Vc = Zc*R*Tc/Pc

    # The loop could be simplified because
    # sum_i sum_j y_i y_j V_{ij} = sum_i y_i^2 V_{ii} + 2 sum_i sum_{j>i} y_i y_j V_ij
    Vc_mix = 0.
    Tc_mix = 0.
    N = len(y)
    for i in range(N):
        for j in range(N):
            if i == j:
                vc = Vc[i]
                tc = Tc[i]
            else:
                vc = (1/8)*(Vc[i]**(1/3) + Vc[j]**(1/3))**3
                tc = np.sqrt(Tc[i]*Tc[j])
            term = y[i]*y[j]*vc
            Vc_mix += term
            Tc_mix += term*tc
    Tc_mix /= Vc_mix

    w_mix = np.dot(y, w)
    Zc_mix = 0.291 - 0.08*w_mix
    Pc_mix = Zc_mix*R*Tc_mix/Vc_mix
    M_mix = np.dot(y, M)

    return KVPC_stiel_thodos(V, M_mix, Tc_mix, Pc_mix, Zc_mix)

# %% Mixing rules


def KVMX2_wassilijewa(y: FloatVector,
                      k: FloatVector,
                      M: FloatVector
                      ) -> float:
    r"""Calculate the thermal conductivity of a gas mixture from the thermal
    conductivities of the pure components using the mixing rule of Wassilijewa,
    with the simplification of Herning and Zipperer.

    $$ k_m = \frac{\sum _i y_i M_i^{1/2} k_i}{\sum _i y_i M_i^{1/2}} $$

    where the meaning of the parameters is as defined below.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, pp. 410, 531.

    Parameters
    ----------
    y : FloatVector
        Mole fractions of all components. Unit = Any.
    k : FloatVector
        Thermal conductivities of all components. Unit = Any.
    M : FloatVector
        Molar masses of all components. Unit = Any.

    Returns
    -------
    float
        Mixture thermal conductivity, $k_m$. Unit = [k].
    """
    a = y*np.sqrt(M)
    a *= k/np.sum(a)
    return np.sum(a, dtype=np.float64)
