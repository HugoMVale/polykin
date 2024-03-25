# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Union

import numpy as np
from numpy import abs, dot, exp, sqrt
from scipy.constants import R

from polykin.properties.mixing_rules import pseudocritical_properties
from polykin.utils.types import FloatArray, FloatVectorLike

__all__ = ['MUVMX2_Herning_Zipperer',
           'MUVPC_Jossi',
           'MUVMXPC_Dean_Stiel',
           'MUV_Lucas',
           'MUVMX_Lucas'
           ]

# %% Mixing rules


def MUVMX2_Herning_Zipperer(y: FloatVectorLike,
                            mu: FloatVectorLike,
                            M: FloatVectorLike
                            ) -> float:
    r"""Calculate the viscosity of a gas mixture from the viscosities of the
    pure components using the mixing rule of Wilke with the approximation of
    Herning and Zipperer.

    $$ \mu_m = \frac{\displaystyle \sum_{i=1}^N y_i M_i^{1/2} \mu_i}
                    {\displaystyle \sum_{i=1}^N y_i M_i^{1/2}} $$

    !!! note

        In this equation, the units of mole fraction $y_i$ and molar mass
        $M_i$ are arbitrary, as they cancel out when considering the ratio of
        the numerator to the denominator.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 410.

    Parameters
    ----------
    y : FloatVectorLike
        Mole fractions of all components. Unit = Any.
    mu : FloatVectorLike
        Viscosities of all components, $\mu$. Unit = Any.
    M : FloatVectorLike
        Molar masses of all components. Unit = Any.

    Returns
    -------
    float
        Mixture viscosity, $\mu_m$. Unit = [mu].

    Examples
    --------
    Estimate the viscosity of a 50 mol% ethylene/1-butene gas mixture at 120°C
    and 1 bar.
    >>> from polykin.properties.viscosity import MUVMX2_Herning_Zipperer
    >>>
    >>> y = [0.5, 0.5]
    >>> mu = [130e-7, 100e-7] # Pa.s, from literature
    >>> M = [28.e-3, 56.e-3]  # kg/mol
    >>>
    >>> mu_mix = MUVMX2_Herning_Zipperer(y, mu, M)
    >>>
    >>> print(f"{mu_mix:.2e} Pa·s")
    1.12e-05 Pa·s

    """
    y = np.asarray(y)
    mu = np.asarray(mu)
    M = np.asarray(M)

    a = y*sqrt(M)
    a /= a.sum()
    return dot(a, mu)

# %% Pressure correction


def MUVPC_Jossi(rhor: float,
                M: float,
                Tc: float,
                Pc: float
                ) -> float:
    r"""Calculate the effect of pressure (or density) on gas viscosity using
    the method of Jossi, Stiel and Thodos for nonpolar gases.

    $$ \left[(\mu -\mu^\circ)\xi + 1\right]^{1/4} = 1.0230 + 0.23364\rho_r
       + 0.58533\rho_r^2 - 0.40758\rho_r^3 + 0.093324\rho_r^4 $$

    where $\mu$ is the dense gas viscosity, $\mu^\circ$ is the is the
    low-pressure viscosity, $\xi$ is a group of constants, and
    $\rho_r = v_c / v$ is the reduced gas density. This equation is valid in
    the range $0.1 < \rho_r < 3.0$.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 424.

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

    Returns
    -------
    float
        Residual viscosity, $(\mu - \mu^\circ)$. Unit = Pa·s.

    Examples
    --------
    Estimate the residual viscosity of ethylene at 350 K and 100 bar.
    >>> from polykin.properties.viscosity import MUVPC_Jossi
    >>>
    >>> vc = 130. # cm³/mol
    >>> v  = 184. # cm³/mol, with Peng-Robinson
    >>> rhor = vc/v
    >>>
    >>> mu_residual = MUVPC_Jossi(rhor=rhor, Tc=282.4, Pc=50.4e5, M=28.05e-3)
    >>>
    >>> print(f"{mu_residual:.2e} Pa·s")
    6.76e-06 Pa·s

    """
    a = 1.0230 + 0.23364*rhor + 0.58533*rhor**2 - 0.40758*rhor**3 \
        + 0.093324*rhor**4
    # 1e7*(1/((1e3)**3 * (1/101325)**4))**(1/6)
    xi = 6.872969367e8*(Tc/(M**3 * Pc**4))**(1/6)
    return (a**4 - 1.)/xi


def MUVMXPC_Dean_Stiel(v: float,
                       y: FloatVectorLike,
                       M: FloatVectorLike,
                       Tc: FloatVectorLike,
                       Pc: FloatVectorLike,
                       Zc: FloatVectorLike,
                       ) -> float:
    r"""Calculate the effect of pressure (or density) on the viscosity of
    gas mixtures using the method of Dean and Stiel for nonpolar components.

    $$ (\mu_m -\mu_m^\circ)\xi = f(\rho_r) $$

    where $\mu_m$ is the dense gas mixture viscosity, $\mu_m^\circ$ is the
    low-pressure gas mixture viscosity, $\xi$ is a group of constants, and
    $\rho_r = v_c / v$ is the reduced gas density. This
    equation is valid in the range $0 \le \rho_r < 2.5$.

    **References**

    *   Dean, D., and Stiel, L. "The viscosity of nonpolar gas mixtures at
        moderate and high pressures." AIChE Journal 11.3 (1965): 526-532.

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

    Returns
    -------
    float
        Residual viscosity, $(\mu_m - \mu_m^\circ)$. Unit = Pa·s.

    Examples
    --------
    Estimate the residual viscosity of a 50 mol% ethylene/propylene mixture
    t 350 K and 100 bar.

    >>> from polykin.properties.viscosity import MUVMXPC_Dean_Stiel
    >>>
    >>> v = 1.12e-4  # m³/mol, with Peng-Robinson
    >>>
    >>> y = [0.5, 0.5]
    >>> M = [28.05e-3, 42.08e-3] # kg/mol
    >>> Tc = [282.4, 364.9]      # K
    >>> Pc = [50.4e5, 46.0e5]    # Pa
    >>> Zc = [0.280, 0.274]
    >>>
    >>> mu_residual = MUVMXPC_Dean_Stiel(v, y, M, Tc, Pc, Zc)
    >>>
    >>> print(f"{mu_residual:.2e} Pa·s")
    2.32e-05 Pa·s

    """

    y = np.asarray(y)
    M = np.asarray(M)
    Tc = np.asarray(Tc)
    Pc = np.asarray(Pc)
    Zc = np.asarray(Zc)

    # Mixing rules recommended in paper
    M_mix = dot(y, M)
    Tc_mix, Pc_mix, vc_mix, _, _ = pseudocritical_properties(y, Tc, Pc, Zc)

    rhor = vc_mix/v
    # xi = 1e3*Tc_mix**(1/6)/(sqrt(M_mix*1e3)*(Pc_mix/101325)**(2/3))
    xi = 6.87e4*Tc_mix**(1/6)/(sqrt(M_mix)*(Pc_mix)**(2/3))
    a = 10.8e-5*(exp(1.439*rhor) - exp(-1.111*rhor**1.858))

    return a/xi


# %% Estimation methods

def MUV_Lucas(T: float,
              P: float,
              M: float,
              Tc: float,
              Pc: float,
              Zc: float,
              dm: float
              ) -> float:
    r"""Calculate the viscosity of a pure gas at a given temperature and
    pressure using the method of Lucas.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 421.

    Parameters
    ----------
    T : float
        Temperature. Unit = K.
    P : float
        Pressure. Unit = Pa.
    M : float
        Molar mass. Unit = kg/mol.
    Tc : float
        Critical temperature. Unit = K.
    Pc : float
        Critical pressure. Unit = Pa.
    Zc : float
        Critical compressibility factor.
    dm : float
        Dipole moment. Unit = debye.

    Returns
    -------
    float
        Gas viscosity, $\mu$. Unit = Pa·s.

    Examples
    --------
    Estimate the viscosity of ethylene at 350 K and 10 bar.

    >>> from polykin.properties.viscosity import MUV_Lucas
    >>> mu = MUV_Lucas(T=350., P=10e5, M=28.05e-3,
    ...                 Tc=282.4, Pc=50.4e5, Zc=0.280, dm=0.)
    >>> print(f"{mu:.2e} Pa·s")
    1.20e-05 Pa·s

    """
    Tr = T/Tc
    Pr = P/Pc
    FP0 = _MUV_Lucas_FP0(Tr, Tc, Pc, Zc, dm)
    mu = _MUV_Lucas_mu(Tr, Pr, M, Tc, Pc, FP0)  # type: ignore
    return mu


def MUVMX_Lucas(T: float,
                P: float,
                y: FloatVectorLike,
                M: FloatVectorLike,
                Tc: FloatVectorLike,
                Pc: FloatVectorLike,
                Zc: FloatVectorLike,
                dm: FloatVectorLike
                ) -> float:
    r"""Calculate the viscosity of a gas mixture at a given temperature and
    pressure using the method of Lucas.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 431.

    Parameters
    ----------
    T : float
        Temperature. Unit = K.
    P : float
        Pressure. Unit = Pa.
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
    dm : FloatVectorLike
        Dipole moments of all components. Unit = debye.

    Returns
    -------
    float
        Gas mixture viscosity, $\mu_m$. Unit = Pa·s.

    Examples
    --------
    Estimate the viscosity of a 60 mol% ethylene/nitrogen gas mixture at 350 K
    and 10 bar.

    >>> from polykin.properties.viscosity import MUVMX_Lucas
    >>>
    >>> y = [0.6, 0.4]
    >>> M = [28.e-3, 28.e-3]   # kg/mol
    >>> Tc = [282.4, 126.2]    # K
    >>> Pc = [50.4e5, 33.9e5]  # Pa
    >>> Zc = [0.280, 0.290]
    >>> dm = [0., 0.]
    >>>
    >>> mu_mix = MUVMX_Lucas(T=350., P=10e5, y=y, M=M,
    ...                      Tc=Tc, Pc=Pc, Zc=Zc, dm=dm)
    >>>
    >>> print(f"{mu_mix:.2e} Pa·s")
    1.45e-05 Pa·s

    """
    y = np.asarray(y)
    M = np.asarray(M)
    Tc = np.asarray(Tc)
    Pc = np.asarray(Pc)
    Zc = np.asarray(Zc)
    dm = np.asarray(dm)

    Tc_mix = dot(y, Tc)
    M_mix = dot(y, M)
    Pc_mix = R*Tc_mix*dot(y, Zc)/dot(y, R*Tc*Zc/Pc)
    FP0_mix = dot(y, _MUV_Lucas_FP0(T/Tc, Tc, Pc, Zc, dm))
    mu = _MUV_Lucas_mu(
        T/Tc_mix, P/Pc_mix, M_mix, Tc_mix, Pc_mix, FP0_mix)

    return mu


def _MUV_Lucas_mu(Tr: float,
                  Pr: float,
                  M: float,
                  Tc: float,
                  Pc: float,
                  FP0: float
                  ) -> float:
    """Calculate the viscosity of a pure gas or gas mixture at a given
    temperature and pressure using the method of Lucas.
    """
    # Z1
    Z1 = (0.807*Tr**0.618 - 0.357*exp(-0.449*Tr) +
          0.340*exp(-4.058*Tr) + 0.018)*FP0

    # Z2
    if Tr <= 1. and Pr < 1.:
        alpha = 3.262 + 14.98*Pr**5.508
        beta = 1.390 + 5.746*Pr
        Z2 = 0.600 + 0.760*Pr**alpha + (6.990*Pr**beta - 0.6)*(1 - Tr)
    else:
        a1 = 1.245e-3
        a2 = 5.1726
        gamma = -0.3286
        b1 = 1.6553
        b2 = 1.2723
        c1 = 0.4489
        c2 = 3.0578
        delta = -37.7332
        d1 = 1.7368
        d2 = 2.2310
        epsilon = -7.6351
        f1 = 0.9425
        f2 = -0.1853
        zeta = 0.4489
        a = a1/Tr*exp(a2*Tr**gamma)
        b = a*(b1*Tr - b2)
        c = c1/Tr*exp(c2*Tr**delta)
        d = d1/Tr*exp(d2*Tr**epsilon)
        e = 1.3088
        f = f1*exp(f2*Tr**zeta)
        Z2 = Z1*(1 + a*Pr**e/(b*Pr**f + 1/(1. + c*Pr**d)))

    # FP
    Y = Z2/Z1
    FP = (1. + (FP0 - 1.)/Y**3)/FP0

    # ξ
    xi = 0.176e7*(Tc/((M*1e3)**3 * (Pc/1e5)**4))**(1/6)

    return Z2*FP/xi


def _MUV_Lucas_FP0(Tr: Union[float, FloatArray],
                   Tc: Union[float, FloatArray],
                   Pc: Union[float, FloatArray],
                   Zc: Union[float, FloatArray],
                   dm: Union[float, FloatArray]
                   ) -> Union[float, FloatArray]:
    "Compute FP0 for Lucas method of estimating gas viscosity."
    dmr = 52.46 * dm**2 * (Pc / 1e5) / Tc**2
    FP0 = np.where(dmr <= 0.022, 0.0, 30.55 * (0.292 - Zc) ** 1.72)
    FP0 = np.where(dmr >= 0.075, FP0*abs(0.96 + 0.1 * (Tr - 0.7)), FP0)
    FP0 += 1.0
    return FP0


# @np.vectorize
# def _visc_gas_lucas_FP0(T: float,
#                         Tc: float,
#                         Pc: float,
#                         Zc: float,
#                         dm: float
#                         ) -> float:
#     "Compute FP0 for Lucas method of gas viscosity estimation."
#     dmr = 52.46*dm**2*(Pc/1e5)/Tc**2
#     if dmr <= 0.022:
#         FP0 = 1.
#     else:
#         FP0 = 30.55*(0.292 - Zc)**1.72
#         if dmr >= 0.075:
#             FP0 *= abs(0.96 + 0.1*(T/Tc - 0.7))
#         FP0 += 1.
#     return FP0
