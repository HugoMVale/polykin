# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector, FloatOrArray

import numpy as np
from scipy.constants import R

__all__ = ['MUVMX2_herning',
           'MUVPC_jossi',
           'MUV_lucas',
           'MUVMX_lucas']


def MUVMX2_herning(y: FloatVector,
                   visc: FloatVector,
                   M: FloatVector
                   ) -> float:
    r"""Calculate the viscosity of a gas mixture from the viscosities of the
    pure components using the mixing rule of Herning and Zipperer.

    $$ \eta _m = \frac{\sum _i y_i M_i^{1/2} \eta_i}{\sum _i y_i M_i^{1/2}} $$

    where the meaning of the parameters is as defined below.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 410.

    Parameters
    ----------
    y : FloatVector
        Mole fractions of all components. Unit = Any.
    visc : FloatVector
        Viscosities of all components, $\eta$. Unit = Any.
    M : FloatVector
        Molar masses of all components. Unit = Any.

    Returns
    -------
    float
        Mixture viscosity, $\eta_m$. Unit = [mu].
    """
    a = y*np.sqrt(M)
    a *= visc/np.sum(a)
    return np.sum(a, dtype=np.float64)


def MUVPC_jossi(dr: float,
                Tc: float,
                Pc: float,
                M: float
                ) -> float:
    r"""Calculate the effect of pressure (or density) on gas viscosity using
    the method of Jossi, Stiel and Thodos for non-polar gases.

    $$ \left[(\eta -\eta^\circ)\xi + 1\right]^{1/4} = 1.0230 + 0.23364\rho_r
       + 0.58533\rho_r^2 - 0.40758\rho_r^3 + 0.093324\rho_r^4 $$

    where $\eta$ is the dense gas viscosity, $\eta^\circ$ is the is the
    low-pressure viscosity, $\xi$ is a group of constants, and
    $\rho_r = \rho / \rho_c = V_c / V$.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 424.

    Parameters
    ----------
    dr : float
        Reduced gas density, $\rho_r$.
    Tc : float
        Critical temperature. Unit = K.
    Pc : float
        Critical pressure. Unit = Pa.
    M : float
        Molar mass. Unit = kg/mol.

    Returns
    -------
    float
        Residual viscosity, $\eta - \eta_0$. Unit = Pa.s.
    """
    a = 1.0230 + 0.23364*dr + 0.58533*dr**2 - 0.40758*dr**3 + 0.093324*dr**4
    # 1e7*(1/((1e3)**3 * (1/101325)**4))**(1/6)
    xi = 6.872969367e8*(Tc/(M**3 * Pc**4))**(1/6)
    return (a**4 - 1)/xi


def MUV_lucas(T: float,
              P: float,
              M: float,
              Tc: float,
              Pc: float,
              Zc: float,
              mu: float
              ) -> float:
    r"""Calculate the viscosity of a pure gas at a given pressure using the
    method of Lucas.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
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
    mu : float
        Dipole moment. Unit = debye.

    Returns
    -------
    float
        Gas viscosity, $\eta$. Unit = Pa.s.
    """
    Tr = T/Tc
    Pr = P/Pc
    FP0 = _MUV_lucas_FP0(Tr, Tc, Pc, Zc, mu)
    eta = _MUV_lucas_eta(Tr, Pr, M, Tc, Pc, FP0)  # type: ignore
    return eta


def MUVMX_lucas(T: float,
                P: float,
                y: FloatVector,
                M: FloatVector,
                Tc: FloatVector,
                Pc: FloatVector,
                Zc: FloatVector,
                mu: FloatVector
                ) -> float:
    r"""Calculate the viscosity of a gas mixture at a given pressure using the
    method of Lucas.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 431.

    Parameters
    ----------
    T : float
        Temperature. Unit = K.
    P : float
        Pressure. Unit = Pa.
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
    mu : FloatVector
        Dipole moments of all components. Unit = debye.

    Returns
    -------
    float
        Gas mixture viscosity, $\eta_m$. Unit = Pa.s.
    """
    Tc_mix = np.dot(y, Tc)
    M_mix = np.dot(y, M)
    Pc_mix = R*Tc_mix*np.dot(y, Zc)/np.dot(y, R*Tc*Zc/Pc)
    FP0_mix = np.dot(y, _MUV_lucas_FP0(T/Tc, Tc, Pc, Zc, mu))
    eta = _MUV_lucas_eta(
        T/Tc_mix, P/Pc_mix, M_mix, Tc_mix, Pc_mix, FP0_mix)
    return eta


def _MUV_lucas_eta(Tr: float,
                   Pr: float,
                   M: float,
                   Tc: float,
                   Pc: float,
                   FP0: float
                   ) -> float:
    """Calculate the viscosity of a pure gas or gas mixture at a given pressure
    using the method of Lucas.
    """
    # Z1
    Z1 = (0.807*Tr**0.618 - 0.357*np.exp(-0.449*Tr) +
          0.340*np.exp(-4.058*Tr) + 0.018)*FP0

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
        a = a1/Tr*np.exp(a2*Tr**gamma)
        b = a*(b1*Tr - b2)
        c = c1/Tr*np.exp(c2*Tr**delta)
        d = d1/Tr*np.exp(d2*Tr**epsilon)
        e = 1.3088
        f = f1*np.exp(f2*Tr**zeta)
        Z2 = Z1*(1 + a*Pr**e/(b*Pr**f + 1/(1. + c*Pr**d)))

    # FP
    Y = Z2/Z1
    FP = (1. + (FP0 - 1.)/Y**3)/FP0

    # ξ
    xi = 0.176e7*(Tc/((M*1e3)**3 * (Pc/1e5)**4))**(1/6)

    return Z2*FP/xi


def _MUV_lucas_FP0(Tr: FloatOrArray,
                   Tc: FloatOrArray,
                   Pc: FloatOrArray,
                   Zc: FloatOrArray,
                   mu: FloatOrArray
                   ) -> FloatOrArray:
    "Compute FP0 for Lucas method of estimating gas viscosity."
    mur = 52.46 * mu**2 * (Pc / 1e5) / Tc**2
    FP0 = np.where(mur <= 0.022, 0.0, 30.55 * (0.292 - Zc) ** 1.72)
    FP0 = np.where(mur >= 0.075, FP0 * np.abs(0.96 + 0.1 * (Tr - 0.7)), FP0)
    FP0 += 1.0
    return FP0

# @np.vectorize
# def _visc_gas_lucas_FP0(T: float,
#                         Tc: float,
#                         Pc: float,
#                         Zc: float,
#                         mu: float
#                         ) -> float:
#     "Compute FP0 for Lucas method of gas viscosity estimation."
#     mur = 52.46*mu**2*(Pc/1e5)/Tc**2
#     if mur <= 0.022:
#         FP0 = 1.
#     else:
#         FP0 = 30.55*(0.292 - Zc)**1.72
#         if mur >= 0.075:
#             FP0 *= np.abs(0.96 + 0.1*(T/Tc - 0.7))
#         FP0 += 1.
#     return FP0
