# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from numpy import exp, log

__all__ = [
    "PL_Lee_Kesler",
    "PL_Wilson",
    "PL_Ambrose_Walton",
]


def PL_Lee_Kesler(
    T: float,
    Tb: float,
    Tc: float,
    Pc: float,
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
        Temperature [K].
    Tb : float
        Normal boiling temperature [K].
    Tc : float
        Critical temperature [K].
    Pc : float
        Critical pressure [Pa].

    Returns
    -------
    float
        Vapor pressure [Pa].

    See Also
    --------
    * [`PL_Ambrose_Walton`](PL_Ambrose_Walton.md): alternative more accurate
      method.

    Examples
    --------
    Estimate the vapor pressure of butyl acrylate at 25°C.
    >>> from polykin.properties.vaporization import PL_Lee_Kesler
    >>> pvap = PL_Lee_Kesler(298.15, Tb=420.0, Tc=644.0, Pc=45.40e5)
    >>> print(f"{pvap:.1e} Pa")
    6.6e+02 Pa
    """

    def fLK(t):
        f0 = 5.92714 - 6.09648 / t - 1.28862 * log(t) + 0.169347 * t**6
        f1 = 15.2518 - 15.6875 / t - 13.4721 * log(t) + 0.43577 * t**6
        return (f0, f1)

    f0, f1 = fLK(T / Tc)
    a, b = fLK(Tb / Tc)
    a = -log(Pc / 101325) - a
    w = a / b

    return Pc * exp(f0 + w * f1)


def PL_Ambrose_Walton(
    T: float,
    Tc: float,
    Pc: float,
    w: float,
) -> float:
    r"""Estimate the vapor pressure of a pure compound using the Ambrose-Walton
    form of the Pitzer equation.

    $$ \ln \frac{P_{vap}}{P_c}
       = f^{(0)}(T_r) + \omega f^{(1)}(T_r) + \omega^2 f^{(2)}(T_r) $$

    where $P_{vap}$ is the vapor pressure, $P_c$ is the critical pressure,
    $\omega$ is the acentric factor, and $f^{(0)}$, $f^{(1)}$ and $f^{(2)}$ are
    empirical functions of the reduced temperature.

    **References**

    *   BE Poling, JM Prausniz, and JP O'Connell. The properties of gases &
        liquids, 5th edition, 2001, p. 235.

    Parameters
    ----------
    T : float
        Temperature [K].
    Tc : float
        Critical temperature [K].
    Pc : float
        Critical pressure [Pa].
    w : float
        Acentric factor.

    Returns
    -------
    float
        Vapor pressure [Pa].

    Examples
    --------
    Estimate the vapor pressure of butadiene at 268.7 K.
    >>> from polykin.properties.vaporization import PL_Ambrose_Walton
    >>> pvap = PL_Ambrose_Walton(268.7, Tc=425.0, Pc=43.3e5, w=0.195)
    >>> print(f"{pvap:.1e} Pa")
    1.0e+05 Pa
    """
    Tr = T / Tc
    t = 1.0 - Tr
    f0 = (-5.97616 * t + 1.29874 * t**1.5 - 0.60394 * t**2.5 - 1.06841 * t**5) / Tr
    f1 = (-5.03365 * t + 1.11505 * t**1.5 - 5.41217 * t**2.5 - 7.46628 * t**5) / Tr
    f2 = (-0.64771 * t + 2.41539 * t**1.5 - 4.26979 * t**2.5 + 3.25259 * t**5) / Tr
    return Pc * exp(f0 + w * f1 + (w**2) * f2)


def PL_Wilson(
    T: float,
    Tc: float,
    Pc: float,
    w: float,
) -> float:
    r"""Estimate the vapor pressure of a pure compound using the Wilson
    approximation.

    $$ \ln \frac{P_{vap}}{P_c} = 5.373(1 + \omega)(1 - 1/T_r) $$

    where $P_{vap}$ is the vapor pressure, $P_c$ is the critical pressure,
    $\omega$ is the acentric factor, and $T_r$ is the reduced temperature.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 207.

    Parameters
    ----------
    T : float
        Temperature [K].
    Tc : float
        Critical temperature [K].
    Pc : float
        Critical pressure [Pa].
    w : float
        Acentric factor.

    Returns
    -------
    float
        Vapor pressure [Pa].

    Examples
    --------
    Estimate the vapor pressure of butadiene at 268.7 K.
    >>> from polykin.properties.vaporization import PL_Wilson
    >>> pvap = PL_Wilson(268.7, Tc=425.0, Pc=43.3e5, w=0.195)
    >>> print(f"{pvap:.1e} Pa")
    1.0e+05 Pa
    """
    return Pc * exp(5.373 * (1 + w) * (1 - Tc / T))
