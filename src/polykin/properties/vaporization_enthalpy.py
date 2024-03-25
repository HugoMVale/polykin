# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024


from typing import Literal, Optional, Union

import numpy as np
from numpy import log, log10
from scipy.constants import R

from polykin.utils.types import FloatArray, FloatArrayLike

__all__ = ['DHVL_Pitzer',
           'DHVL_Vetere',
           'DHVL_Watson',
           'DHVL_Kistiakowsky_Vetere'
           ]


def DHVL_Pitzer(T: Union[float, FloatArrayLike],
                Tc: float,
                w: float
                ) -> Union[float, FloatArray]:
    r"""Calculate the enthalpy of vaporization of a pure compound at a given
    temperature, $\Delta H_v(T)$, using the Pitzer acentric factor method.

    $$
    \frac{\Delta H_v}{R T_c}=7.08(1-T_r)^{0.354} + 10.95 \omega (1-T_r)^{0.456}
    $$

    where $T_c$ is the critical temperature and $T_r=T/T_c$ is the reduced
    temperature, and $\omega$ is the acentric factor. The equation is valid in
    the range $0.6<T_r<1$.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 220.

    Parameters
    ----------
    T : float | FloatArrayLike
        Temperature. Unit = K.
    Tc : float
        Critical temperature. Unit = K.
    w : float
        Acentric factor.

    Returns
    -------
    float | FloatArray
        Vaporization enthalpy. Unit = J/mol.

    !!! note annotate "See also"

        * [`DHVL_Kistiakowsky_Vetere`](DHVL_Kistiakowsky_Vetere.md): alternative method.
        * [`DHVL_Vetere`](DHVL_Vetere.md): alternative method.

    Examples
    --------
    Estimate the vaporization enthalpy of vinyl chloride at 50°C.
    >>> from polykin.properties.vaporization_enthalpy import DHVL_Pitzer
    >>> DHVL = DHVL_Pitzer(T=273.15+50, Tc=425., w=0.122)
    >>> print(f"{DHVL/1e3:.1f} kJ/mol")
    17.5 kJ/mol

    """
    T = np.asarray(T)
    Tr = T/Tc
    return R*Tc*(7.08*(1. - Tr)**0.354 + 10.95*w*(1. - Tr)**0.456)


def DHVL_Vetere(Tb: float,
                Tc: float,
                Pc: float,
                ) -> float:
    r"""Calculate the enthalpy of vaporization of a pure compound at the normal
    boiling point, $\Delta H_{vb}$, using the Vetere method.

    $$
    \Delta H_{vb} = R T_c T_{br} \frac{0.4343\ln(P_c/10^5)-0.69431
    + 0.89584 T_{br}}{0.37691-0.37306 T_{br}
    + 0.15075 (P_c/10^5)^{-1} T_{br}^{-2}}
    $$

    where $P_c$ is the critical pressure, $T_c$ is the critical temperature,
    $T_b$ is the normal boiling temperature, and $T_{br}=T_b/T_c$ is the
    reduced normal boiling temperature.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 227.

    Parameters
    ----------
    Tb : float
        Normal boiling point temperature. Unit = K.
    Tc : float
        Critical temperature. Unit = K.
    Pc : float
        Critical pressure. Unit = Pa.

    Returns
    -------
    float
        Vaporization enthalpy at the normal boiling point. Unit = J/mol.

    !!! note annotate "See also"

        * [`DHVL_Kistiakowsky_Vetere`](DHVL_Kistiakowsky_Vetere.md): alternative method.
        * [`DHVL_Pitzer`](DHVL_Pitzer.md): alternative method.

    Examples
    --------
    Estimate the vaporization enthalpy of vinyl chloride at the normal boiling
    temperature.
    >>> from polykin.properties.vaporization_enthalpy import DHVL_Vetere
    >>> DHVL = DHVL_Vetere(Tb=259.8, Tc=425., Pc=51.5e5)
    >>> print(f"{DHVL/1e3:.1f} kJ/mol")
    21.6 kJ/mol

    """
    Tbr = Tb/Tc
    return R*Tc*Tbr*(0.4343*log(Pc/1e5) - 0.69431 + 0.89584*Tbr) \
        / (0.37691 - 0.37306*Tbr + 0.15075/(Pc/1e5)/Tbr**2)


def DHVL_Kistiakowsky_Vetere(
        Tb: float,
        M: Optional[float] = None,
        kind: Literal['any', 'acid_alcohol', 'ester',
                      'hydrocarbon', 'polar'] = 'any') -> float:
    r"""Calculate the enthalpy of vaporization of a pure compound at the normal
    boiling point, using Vetere's modification of the Kistiakowsky method.

    $$ \Delta H_{vb} = T_b \Delta S_{vb}(T_b, M, kind) $$

    where $T_b$ is the normal boiling point temperature, $M$ is the molar mass,
    and $\Delta S_{vb}(...)$ is the entropy of vaporization, given by one of
    five empirical correlations depending on the kind of compound (see
    Parameter description).

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 231.

    Parameters
    ----------
    Tb : float
        Normal boiling point temperature. Unit = K.
    M : float | None
        Molar mass. Must be provided except if `kind = "any"`. Unit = kg/mol.
    kind : Literal['any', 'acid_alcohol', 'ester', 'hydrocarbon', 'polar']
        Type of compound. `any` corresponds ot the original correlation
        proposed by Kistiakowsky.

    Returns
    -------
    float
        Vaporization enthalpy at the normal boiling point. Unit = J/mol.

    !!! note annotate "See also"

        * [`DHVL_Pitzer`](DHVL_Pitzer.md): alternative method.
        * [`DHVL_Vetere`](DHVL_Vetere.md): alternative method.

    Examples
    --------
    Estimate the vaporization enthalpy of butadiene at the normal boiling
    temperature.
    >>> from polykin.properties.vaporization_enthalpy \
    ...      import DHVL_Kistiakowsky_Vetere
    >>> DHVL = DHVL_Kistiakowsky_Vetere(Tb=268.6, M=54.1e-3, kind='hydrocarbon')
    >>> print(f"{DHVL/1e3:.1f} kJ/mol")
    22.4 kJ/mol

    """

    if kind == 'any':
        DSvb = 30.6 + R*log(Tb)
        return DSvb*Tb

    if M is not None:
        M = M*1e3
    else:
        raise ValueError("`M` can't be `None` with selected compound kind.")

    if kind == 'hydrocarbon':
        DSvb = 58.20 + 13.7*log10(M) + 6.49/M*(Tb - (263*M)**0.581)**1.037
    elif kind == 'polar' or kind == 'ester':
        DSvb = 44.367 + 15.33 * \
            log10(Tb) + 0.39137*Tb/M + 4.330e-3/M*Tb**2 - 5.627e-6/M*Tb**3
        if kind == 'ester':
            DSvb *= 1.03
    elif kind == 'acid_alcohol':
        DSvb = 81.119 + 13.083 * \
            log10(Tb) - 25.769*Tb/M + 0.146528/M*Tb**2 - 2.1362e-4/M*Tb**3
    else:
        raise ValueError("Invalid compound `kind`.")
    return DSvb*Tb


def DHVL_Watson(hvap1: float,
                T1: float,
                T2: float,
                Tc: float
                ) -> float:
    r"""Calculate the variation of the vaporization enthalpy of a pure compound
    with temperature using the Watson method.

    $$ \Delta H_{v,2} = \Delta H_{v,1}
       \left(\frac{1-T_{r,2}}{1-T_{r,1}}\right)^{0.38} $$

    where $T_{r,i}=T_i/T_c$.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 228.

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

    Returns
    -------
    float
        Vaporization temperature at `T2`. Unit = [hvap1].

    Examples
    --------
    Estimate the vaporization enthalpy of vinyl chloride at 50°C from the known
    value at the normal boiling temperature.
    >>> from polykin.properties.vaporization_enthalpy import DHVL_Watson
    >>> DHVL = DHVL_Watson(hvap1=22.9, T1=258., T2=273.15+50, Tc=425.)
    >>> print(f"{DHVL:.1f} kJ/mol")
    19.0 kJ/mol

    """
    return hvap1*((1. - T2/Tc)/(1. - T1/Tc))**0.38
