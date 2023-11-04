# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

""" This module provides correlations to estimate the enthalpy of vaporization
of pure components.
"""

from polykin.utils import FloatOrArray

import numpy as np
from scipy.constants import R
from typing import Optional, Literal

__all__ = ['hvap_pitzer', 'hvapb_vetere', 'hvap_watson',
           'hvapb_kistiakowsky_vetere']


def hvap_pitzer(T: FloatOrArray,
                Tc: float,
                w: float
                ) -> FloatOrArray:
    r"""Estimate the enthalpy of vaporization of a pure compound at a given
    temperature, $\Delta H_v(T)$, using the Pitzer acentric factor method.

    $$
    \frac{\Delta H_v}{R T_c}=7.08(1-T_r)^{0.354} + 10.95 \omega (1-T_r)^{0.456}
    $$

    where $T_c$ is the critical temperature and $T_r=T/T_c$ is the reduced
    temperature, and $\omega$ is the acentric factor. The equation is valid in
    the range $0.6<T_r<1$.

    Parameters
    ----------
    T : FloatOrArray
        Temperature. Unit = K.
    Tc : float
        Critical temperature. Unit = K.
    w : float
        Acentric factor.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 657. 1986.

    Returns
    -------
    FloatOrArray
        Vaporization enthalpy. Unit = J/mol.
    """
    Tr = T/Tc
    return R*Tc*(7.08*(1 - Tr)**0.354 + 10.95*w*(1 - Tr)**0.456)


def hvapb_vetere(Tb: float,
                 Tc: float,
                 Pc: float,
                 ) -> float:
    r"""Estimate the enthalpy of vaporization of a pure compound at the normal
    boiling point, $\Delta H_{vb}$, using the Vetere method.

    $$
    \Delta H_{vb} = R T_c T_{br} \frac{0.4343\ln(P_c/10^5)-0.69431
    + 0.89584 T_{br}}{0.37691-0.37306 T_{br}
    + 0.15075 (P_c/10^5)^{-1} T_{br}^{-2}}
    $$

    where $P_c$ is the critical pressure, $T_c$ is the critical temperature,
    $T_b$ is the normal boiling temperature, and $T_{br}=T_b/T_c$ is the
    reduced normal boiling temperature.

    Parameters
    ----------
    Tb : float
        Normal boiling point temperature. Unit = K.
    Tc : float
        Critical temperature. Unit = K.
    Pc : float
        Critical pressure. Unit = Pa.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 657. 1986.

    Returns
    -------
    float
        Vaporization enthalpy at the normal boiling point. Unit = J/mol.
    """
    Tbr = Tb/Tc
    return R*Tc*Tbr*(0.4343*np.log(Pc/1e5) - 0.69431 + 0.89584*Tbr) \
        / (0.37691 - 0.37306*Tbr + 0.15075/(Pc/1e5)/Tbr**2)


def hvap_watson(hvap1: float,
                T1: float,
                T2: float,
                Tc: float
                ) -> float:
    r"""Estimate the variation of the vaporization enthalpy of a pure compound
    with temperature using the Watson method.

    $$ \Delta H_{v,2} = \Delta H_{v,1}
       \left(\frac{1-T_{r,2}}{1-T_{r,1}}\right)^{0.38} $$

    where $T_{r,i}=T_i/T_c$.

    Parameters
    ----------
    hvap1 : float
        Vaporization enthalpy at `T1`. Unit = Any.
    T1 : float
        Temperature corresponding to `hvap1`. Unit = K.
    T2 : float
        Temperature at which the vaporization temperature is to be computed.
        Unit = K.
    Tc : float
        Critical temperature. Unit = K.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 657. 1986.

    Returns
    -------
    float
        Vaporization temperature at `T2`. Unit = [hvap1].
    """
    return hvap1*((1 - T2/Tc)/(1 - T1/Tc))**0.38


def hvapb_kistiakowsky_vetere(
        Tb: float,
        M: Optional[float] = None,
        kind: Literal['any', 'acid_alcohol', 'ester',
                      'hydrocarbon', 'polar'] = 'any') -> float:
    r"""Estimate the enthalpy of vaporization of a pure compound at the normal
    boiling point, $\Delta H_{vb}$, using Vetere's modification of the
    Kistiakowsky method.

    $$ \Delta H_{vb} = T_b \Delta S_{vb}(T_b, M, kind) $$

    where $T_b$ is the normal boiling point temperature, $M$ is the molar mass,
    and $\Delta S_{vb}(...)$ is the entropy of vaporization, given by one of
    five empirical correlations depending on the kind of compound (see
    Parameter description).

    Parameters
    ----------
    Tb : float
        Normal boiling point temperature. Unit = K.
    M : float | None
        Molar mass. Unit = kg/mol.
    kind : Literal['any', 'acid_alcohol', 'ester', 'hydrocarbon', 'polar']
        Type of compound. `any` corresponds ot the original correlation
        proposed by Kistiakowsky.

    Returns
    -------
    float
        Vaporization enthalpy at the normal boiling point. Unit = J/mol.
    """

    if kind == 'any':
        DSvb = 30.6 + R*np.log(Tb)
        return DSvb*Tb

    if M is not None:
        M = M*1e3
    else:
        raise ValueError("`M` can't be `None` with selected compound kind.")

    if kind == 'hydrocarbon':
        DSvb = 58.20 + 13.7*np.log10(M) + 6.49/M*(Tb - (263*M)**0.581)**1.037
    elif kind == 'polar' or kind == 'ester':
        DSvb = 44.367 + 15.33 * \
            np.log10(Tb) + 0.39137*Tb/M + 4.330e-3/M*Tb**2 - 5.627e-6/M*Tb**3
        if kind == 'ester':
            DSvb *= 1.03
    elif kind == 'acid_alcohol':
        DSvb = 81.119 + 13.083 * \
            np.log10(Tb) - 25.769*Tb/M + 0.146528/M*Tb**2 - 2.1362e-4/M*Tb**3
    else:
        raise ValueError("Invalid compound `kind`.")
    return DSvb*Tb
