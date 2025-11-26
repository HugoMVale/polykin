# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from numpy import exp, log

all = ['PL_Pizter']


def PL_Pitzer(
    T: float,
    Tb: float,
    Tc: float,
    Pc: float
) -> float:
    r"""Estimate the vapor pressure of a pure compound using the Lee-Kesler 
    form of the Pitzer equation.

    $$ \ln \frac{P_{vap}}{P_c} = f^{(0)}(T_r) + \omega f^{(1)}(T_r) $$

    where $P_{vap}$ is the vapor pressure, $P_c$ is the critical pressure,
    $\omega$ is the acentric factor, and $f^{(0)}$ and $f^{(1)}$ are empirical
    functions of the reduced temperature.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 207.

    Parameters
    ----------
    T : float
        Temperature. Unit = K.
    Tb : float
        Normal boiling temperature. Unit = K.
    Tc : float
        Critical temperature. Unit = K.
    Pc : float
        Critical pressure. Unit = Pa.

    Returns
    -------
    float
        Vapor pressure. Unit = Pa.

    Examples
    --------
    Estimate the vapor pressure of butyl acrylate at 25°C.
    >>> from polykin.properties.vaporization import PL_Pitzer
    >>> pvap = PL_Pitzer(298.15, Tb=420.0, Tc=644.0, Pc=45.40e5)
    >>> print(f"{pvap:.1e} Pa")
    6.6e+02 Pa
    """

    Tr = T/Tc
    f0 = 5.92714 - 6.09648/Tr - 1.28862*log(Tr) + 0.169347*Tr**6
    f1 = 15.2518 - 15.6875/Tr - 13.4721*log(Tr) + 0.43577*Tr**6

    Tbr = Tb/Tc
    a = - log(Pc/101325) - 5.92714 + 6.09648/Tbr + 1.28862*log(Tbr) - 0.169347*Tbr**6
    b = 15.2518 - 15.6875/Tbr - 13.4721*log(Tbr) + 0.43577*Tbr**6
    w = a/b

    return Pc*exp(f0 + w*f1)
