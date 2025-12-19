# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from scipy.constants import R

__all__ = ["VL_Rackett"]


def VL_Rackett(
    T: float,
    Tc: float,
    Pc: float,
    ZRA: float | None = None,
    w: float | None = None,
) -> float:
    r"""Calculate the saturated liquid molar volume of a pure component using
    the Rackett equation.

    $$ v^L = \frac{R T_c}{P_c} Z_{RA}^{1 + (1-T_r)^{2/7}} $$

    where $T_c$ is the critical temperature, $P_c$ is the critical pressure,
    $T_r=T/T_c$ is the reduced temperature, and $Z_{RA}$ is the Rackett
    compressibility factor.

    If $Z_{RA}$ is not known, it will be estimated from the acentric factor
    $\omega$ using the following approximation:

    $$ Z_{RA} = 0.29056 - 0.08775 \omega $$

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 67.

    Parameters
    ----------
    T : float
        Temperature [K].
    Tc : float
        Critical temperature [K].
    Pc : float
        Critical pressure [Pa].
    ZRA : float | None
        Rackett compressibility factor. If provided, it is used directly.
    w : float | None
        Acentric factor. Used to estimate `ZRA` if `ZRA` is not given.

    Returns
    -------
    float
        Saturated liquid molar volume [m³/mol].

    Examples
    --------
    Estimate the molar saturated liquid volume of butadiene at 350 K.

    >>> from polykin.properties.pvt import VL_Rackett
    >>> vL = VL_Rackett(350.0, 425.0, 43.3e5, w=0.195)
    >>> print(f"{vL:.2e} m³/mol")
    1.01e-04 m³/mol
    """
    if (ZRA is None) == (w is None):
        raise ValueError("Provide exactly one of `ZRA` or `w`.")

    if ZRA is None and w is not None:
        ZRA = 0.29056 - 0.08775 * w

    Tr = T / Tc
    return (R * Tc / Pc) * ZRA ** (1 + (1 - Tr) ** (2 / 7))
