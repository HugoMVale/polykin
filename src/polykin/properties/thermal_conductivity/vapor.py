# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np
from numpy import dot, exp, sqrt
from scipy.constants import N_A, R

from polykin.utils.types import FloatVectorLike

__all__ = ['KVPC_Stiel_Thodos',
           'KVMXPC_Stiel_Thodos',
           'KVMX2_Wassilijewa']

# %% Pressure correction


def KVPC_Stiel_Thodos(v: float,
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
    is the critical compressibility factor, and $\rho_r = v_c / v$ is the
    reduced gas density. This equation is valid in the range
    $0 \leq \rho_r < 2.8$.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 521.

    Parameters
    ----------
    v : float
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

    Examples
    --------
    Estimate the residual thermal conductivity of ethylene at 350 K and
    100 bar.
    >>> from polykin.properties.thermal_conductivity import KVPC_Stiel_Thodos
    >>>
    >>> v = 1.84e-4 # m³/mol, with Peng-Robinson
    >>>
    >>> k_residual = KVPC_Stiel_Thodos(v=v, M=28.05e-3,
    ...                                Tc=282.4, Pc=50.4e5, Zc=0.280)
    >>>
    >>> print(f"{k_residual:.2e} W/(m·K)")
    1.69e-02 W/(m·K)

    """

    gamma = ((Tc * M**3 * N_A**2)/(R**5 * Pc**4))**(1/6)
    vc = Zc*R*Tc/Pc
    rhor = vc/v

    if rhor < 0.5:
        a = 1.22e-2*(exp(0.535*rhor) - 1)
    elif rhor < 2.0:
        a = 1.14e-2*(exp(0.67*rhor) - 1.069)
    elif rhor < 2.8:
        a = 2.60e-3*(exp(1.155*rhor) + 2.016)
    else:
        raise ValueError("Invalid `rhor` input. Valid range: `rhor` < 2.8.")

    return a/(gamma * Zc**5)


def KVMXPC_Stiel_Thodos(v: float,
                        y: FloatVectorLike,
                        M: FloatVectorLike,
                        Tc: FloatVectorLike,
                        Pc: FloatVectorLike,
                        Zc: FloatVectorLike,
                        w: FloatVectorLike
                        ) -> float:
    r"""Calculate the effect of pressure (or density) on the thermal
    conductivity of gas mixtures using the method of Stiel and Thodos for
    nonpolar components, combined with the mixing rules of Yorizane.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 536.

    Parameters
    ----------
    v : float
        Gas molar volume. Unit = m³/mol.
    y : FloatVectorLike
        Mole fractions of all components. Unit = mol/mol.
    M : FloatVectorLike
        Molar masses of all components. Unit = kg/mol.
    Tc : FloatVectorLike
        Critical temperatures of all components. Unit = K.
    Pc : FloatVectorLike
        Critical pressures of all components. Unit = Pa.
    Zc : FloatVectorLike
        Critical compressibility factors of all components.
    w : FloatVectorLike
        Acentric factors of all components.

    Returns
    -------
    float
        Residual thermal conductivity, $(k_m - k_m^{\circ})$. Unit = W/(m·K).

    Examples
    --------
    Estimate the residual thermal conductivity of a 50 mol% ethylene/propylene
    mixture at 350 K and 100 bar.
    >>> from polykin.properties.thermal_conductivity import KVMXPC_Stiel_Thodos
    >>> import numpy as np
    >>>
    >>> v = 1.12e-4  # m³/mol, with Peng-Robinson
    >>>
    >>> y = [0.5, 0.5]
    >>> M = [28.05e-3, 42.08e-3]  # kg/mol
    >>> Pc = [50.4e5, 46.0e5]     # Pa
    >>> Tc = [282.4, 364.9]       # K
    >>> Zc = [0.280, 0.274]
    >>> w = [0.089, 0.144]
    >>>
    >>> k_residual = KVMXPC_Stiel_Thodos(v, y, M, Tc, Pc, Zc, w)
    >>>
    >>> print(f"{k_residual:.2e} W/(m·K)")
    3.82e-02 W/(m·K)

    """

    y = np.asarray(y)
    M = np.asarray(M)
    Tc = np.asarray(Tc)
    Pc = np.asarray(Pc)
    Zc = np.asarray(Zc)
    w = np.asarray(w)

    vc = Zc*R*Tc/Pc

    # The loop could be simplified because
    # sum_i sum_j y_i y_j V_{ij} = sum_i y_i^2 V_{ii} + 2 sum_i sum_{j>i} y_i y_j V_ij
    vc_mix = 0.
    Tc_mix = 0.
    N = len(y)
    for i in range(N):
        for j in range(N):
            if i == j:
                vc_ = vc[i]
                Tc_ = Tc[i]
            else:
                vc_ = (1/8)*(vc[i]**(1/3) + vc[j]**(1/3))**3
                Tc_ = sqrt(Tc[i]*Tc[j])
            vc_term = y[i]*y[j]*vc_
            vc_mix += vc_term
            Tc_mix += vc_term*Tc_
    Tc_mix /= vc_mix

    w_mix = dot(y, w)
    Zc_mix = 0.291 - 0.08*w_mix
    Pc_mix = Zc_mix*R*Tc_mix/vc_mix
    M_mix = dot(y, M)

    return KVPC_Stiel_Thodos(v, M_mix, Tc_mix, Pc_mix, Zc_mix)

# %% Mixing rules


def KVMX2_Wassilijewa(y: FloatVectorLike,
                      k: FloatVectorLike,
                      M: FloatVectorLike
                      ) -> float:
    r"""Calculate the thermal conductivity of a gas mixture from the thermal
    conductivities of the pure components using the mixing rule of Wassilijewa,
    with the simplification of Herning and Zipperer.

    $$ k_m = \frac{\displaystyle \sum_{i=1}^N y_i M_i^{1/2} k_i}
                  {\displaystyle \sum_{i=1}^N y_i M_i^{1/2}} $$

    !!! note

        In this equation, the units of mole fraction $y_i$ and molar mass
        $M_i$ are arbitrary, as they cancel out when considering the ratio of
        the numerator to the denominator.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, pp. 410, 531.

    Parameters
    ----------
    y : FloatVectorLike
        Mole fractions of all components. Unit = Any.
    k : FloatVectorLike
        Thermal conductivities of all components. Unit = Any.
    M : FloatVectorLike
        Molar masses of all components. Unit = Any.

    Returns
    -------
    float
        Mixture thermal conductivity, $k_m$. Unit = [k].

    Examples
    --------
    Estimate the thermal conductivity of a 50 mol% styrene/ethyl-benzene gas
    mixture at 25°C and 0.1 bar.
    >>> from polykin.properties.thermal_conductivity import KVMX2_Wassilijewa
    >>>
    >>> y = [0.5, 0.5]
    >>> k = [1.00e-2, 1.55e-2] # W/(m·K), from literature
    >>> M = [104.15, 106.17]   # g/mol
    >>>
    >>> k_mix = KVMX2_Wassilijewa(y, k, M)
    >>>
    >>> print(f"{k_mix:.2e} W/(m·K)")
    1.28e-02 W/(m·K)

    """
    y = np.asarray(y)
    k = np.asarray(k)
    M = np.asarray(M)

    a = y*sqrt(M)
    a /= a.sum()

    return dot(a, k)
