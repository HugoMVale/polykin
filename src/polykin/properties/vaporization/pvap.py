# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from numpy import exp, log

__all__ = ['PL_Lee_Kesler']


def PL_Lee_Kesler(
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
    >>> from polykin.properties.vaporization import PL_Lee_Kesler
    >>> pvap = PL_Lee_Kesler(298.15, Tb=420.0, Tc=644.0, Pc=45.40e5)
    >>> print(f"{pvap:.1e} Pa")
    6.6e+02 Pa
    """

    def fLK(t):
        f0 = 5.92714 - 6.09648/t - 1.28862*log(t) + 0.169347*t**6
        f1 = 15.2518 - 15.6875/t - 13.4721*log(t) + 0.43577*t**6
        return (f0, f1)

    f0, f1 = fLK(T/Tc)
    a, b = fLK(Tb/Tc)
    a = - log(Pc/101325) - a
    w = a/b

    return Pc*exp(f0 + w*f1)
