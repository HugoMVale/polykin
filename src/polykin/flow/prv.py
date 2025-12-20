# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from dataclasses import dataclass

from numpy import log as ln
from numpy import nan, sqrt

__all__ = [
    "area_relief_gas",
    "area_relief_liquid",
    "area_relief_2phase",
    "area_relief_2phase_subcooled",
    "PRVResult",
]


@dataclass(frozen=True)
class PRVResult:
    """Dataclass for pressure relief device results.

    Attributes
    ----------
    Pcf : float
        Critical flow pressure [bara].
    critical_flow : bool
        Flag indicating if the flow is sonic (`True`) or subsonic (`False`).
    A : float
        Required effective discharge area [mm²].
    """

    Pcf: float
    critical_flow: bool
    A: float

    def __repr__(self):
        """Return a string representation of the PRVResult."""
        return f"PRVResult(Pcf={self.Pcf:.2e} bara, critical_flow={self.critical_flow}, A={self.A:.2e} mm²)"


def area_relief_gas(
    W: float,
    P1: float,
    P2: float,
    T: float,
    k: float,
    M: float,
    *,
    Z: float = 1.0,
    steam: bool = False,
    Kd: float = 0.975,
    Kb: float = 1.0,
    Kc: float = 1.0,
    KSH: float = 1.0,
) -> PRVResult:
    r"""Calculate the required effective discharge area of a pressure relief
    device in gas or steam service (single-phase flow).

    The calculation is done according to the API standard 520. No distinction
    is made between gas and vapor (synomyms in this context).

    The calculation method relies on assumptions whose validity diminishes as
    the gas approaches the critical point. Typically, the reduced volume of the
    relieving gas should be greater than 2.0.

    **References**

    * Sizing, Selection, and Installation of Pressure-relieving Devices in
      Refineries: Part I—Sizing and Selection, API Standard 520, 10th ed., 2020.

    Parameters
    ----------
    W : float
        Required relieving mass flow rate [kg/h].
    P1 : float
        Upstream relieving pressure, absolute [bara].
    P2 : float
        Downstream back pressure, absolute [bara].
    T : float
        Absolute relieving temperature of the gas at the valve inlet [K].
    k : float
        Ideal gas specific heat ratio at relieving temperature.
    M : float
        Molar mass of gas [g/mol].
    Z : float
        Compressibility factor of the gas at relieving pressure and temperature.
        If a calculated value is not available, use `Z=1.0`, which leads to a
        conservative estimate of the discharge area.
    steam : bool
        Flag for steam service. A special calculation is used for devices in
        steam service that operate at critical (sonic) flow conditions.
    Kd: float
        Effective discharge coefficient. Use `Kd=0.975` when sizing a PRV and
        `Kd=0.62` when sizing a rupture disk without PRV.
    Kb : float
        Backpressure correction factor for balanced bellows valves. The value
        can be obtained from the manufacturer or estimated with the help of
        Figure 31 of the API standard 520.
    Kc : float
        Combination correction factor for installations with a rupture disk
        upstream of the PRV. Use `Kc=1.0` if there is no rupture disk, and
        `Kc=0.9` if there is a rupture disk.
    KSH : float
        Steam superheat correction factor. Use `KSH=1.0` for saturated steam.
        For superheated steam, the value can be obtained from Tables 12 or 13
        of the API standard 520.

    Returns
    -------
    PRVResult
        Dataclass containing the results of the calculation.

    Examples
    --------
    Estimate the required discharge area of a pressure relief valve in gas
    service, according to the API standard 520. The required flow is 24270 kg/h,
    the relieving pressure is 6.7 bara, the back pressure is 0 barg, the
    relieving temperature is 348 K, the gas specific heat ratio is 1.11, the
    gas molar mass is 51 g/mol, and the compressibility factor is 0.9.
    >>> from polykin.flow import area_relief_gas
    >>> area_relief_gas(W=24270, P1=6.7, P2=1.01325, T=348, k=1.11, M=51, Z=0.9)
    PRVResult(Pcf=3.90e+00 bara, critical_flow=True, A=3.70e+03 mm²)
    """
    # Convert pressures from bar to kPa
    P1 *= 1e2
    P2 *= 1e2

    # Critical flow nozzle pressure
    Pcf = P1 * (2 / (k + 1)) ** (k / (k - 1))
    critical_flow = P2 <= Pcf

    if critical_flow:
        if steam:
            if P1 <= 10339.0:
                KN = 1.0
            elif P1 < 22057.0:
                KN = (0.02764 * P1 - 1000) / (0.03324 * P1 - 1061)
            else:
                raise ValueError("`P1` out of range.")
            A = 190.5 * W / (P1 * Kd * Kb * Kc * KN * KSH)
        else:
            C = 0.03948 * sqrt(k * (2 / (k + 1)) ** ((k + 1) / (k - 1)))
            A = W / (C * Kd * P1 * Kb * Kc) * sqrt(T * Z / M)
    else:
        r = P2 / P1
        F2 = sqrt((k / (k - 1)) * r ** (2 / k) * ((1 - r ** ((k - 1) / k)) / (1 - r)))
        A = 17.9 * W / (F2 * Kd * Kc) * sqrt(T * Z / (M * P1 * (P1 - P2)))

    return PRVResult(Pcf / 1e2, critical_flow, A)


def area_relief_liquid(
    Q: float,
    P1: float,
    P2: float,
    mu: float,
    Gl: float,
    *,
    Kd: float = 0.65,
    Kw: float = 1.0,
    Kc: float = 1.0,
) -> float:
    r"""Calculate the required effective discharge area of a pressure relief
    device in liquid service (single-phase flow).

    The calculation is done according to the API standard 520, assuming the
    capacity of the valves is certified, as required by the ASME code.

    **References**

    * Sizing, Selection, and Installation of Pressure-relieaving Devices in
      Refineries: Part I—Sizing and Selection, API Standard 520, 10th ed., 2020.

    Parameters
    ----------
    Q : float
        Required relieving volume flow rate [L/min].
    P1 : float
        Upstream relieving pressure [bar]. It can be absolute or gauge, as long
        as it is consistent with `P2`.
    P2 : float
        Downstream back pressure [bar]. It can be absolute or gauge, as long as
        it is consistent with `P1`.
    mu : float
        Viscosity at the relieving temperature [cP].
    Gl : float
        Relative density of the liquid at the relieving temperature with respect
        to water at standard conditions.
    Kd : float
        Effective discharge coefficient. Use `Kd=0.65` when sizing a PRV and
        `Kd=0.62` when sizing a rupture disk without PRV.
    Kw : float
        Backpressure correction factor for balanced bellows valves. The value
        can be obtained from the manufacturer or estimated with the help of
        Figure 32 of the API standard 520.
    Kc : float
        Combination correction factor for installations with a rupture disk
        upstream of the PRV. Use `Kc=1.0` if there is no rupture disk, and
        `Kc=0.9` if there is a rupture disk.

    Returns
    -------
    float
        Effective discharge area [mm²].

    Examples
    --------
    Estimate the required discharge area of a pressure relief valve in liquid
    service, according to the API standard 520. The required flow is 6814 L/min,
    the relieving pressure is 18.96 barg, the back pressure is 3.45 barg, the
    liquid relative density is 0.9, the liquid viscosity is 396 cP, and the valve
    back pressure correction factor is 0.97.
    >>> from polykin.flow import area_relief_liquid
    >>> A = area_relief_liquid(Q=6814, P1=18.96, P2=3.45, mu=396, Gl=0.9, Kw=0.97)
    >>> print(f"Effective discharge area: {A:.0f} mm²")
    Effective discharge area: 3172 mm²
    """
    # Convert pressures from bar to kPa
    P1 *= 1e2
    P2 *= 1e2

    A = 0.0
    A_old = 0.0
    converged = False
    MAX_ITER = 50
    for i in range(MAX_ITER):

        if i == 0:
            Kv = 1.0
        else:
            Re = 18800 * Q * Gl / (mu * sqrt(A))
            Kv = 1 / (0.9935 + 2.878 / Re**0.5 + 342.75 / Re**1.5)

        A = 11.78 * Q / (Kd * Kw * Kc * Kv) * sqrt(Gl / (P1 - P2))

        if abs((A - A_old) / A) < 1e-5:
            converged = True
            break
        else:
            A_old = A

    if not converged:
        raise ValueError(f"Failed to converge after {MAX_ITER} iterations.")

    return A


def area_relief_2phase(
    W: float,
    P1: float,
    P2: float,
    v1: float,
    v9: float,
    *,
    Kd: float = 0.85,
    Kb: float = 1.0,
    Kc: float = 1.0,
    Kv: float = 1.0,
) -> PRVResult:
    r"""Calculate the required effective discharge area of a pressure relief
    device for two-phase flashing or non-flashing flow using the omega method.

    The calculation is done according to the API standard 520, Appendix C.2.2.

    This method can also be used for liquids that are saturated (but not
    subcooled) as they enter the relief device.

    **References**

    * Sizing, Selection, and Installation of Pressure-relieving Devices in
      Refineries: Part I—Sizing and Selection, API Standard 520, 10th ed., 2020.

    Parameters
    ----------
    W : float
        Required relieving mass flow rate [kg/h].
    P1 : float
        Upstream relieving pressure, absolute [bara].
    P2 : float
        Downstream back pressure, absolute [bara].
    v1 : float
        Overall specific volume of the two-phase mixture at upstream relieving
        conditions [m³/kg].
    v9 : float
        Overall specific volume of the two-phase mixture evaluated at 90% of
        the upstream pressure [m³/kg]. The flash calculation should be carried
        out isentropically, but an isenthalpic (adiabatic) flash is sufficient
        for low-vapor-content mixtures far from the thermodynamic critical point.
    Kd : float
        Effective discharge coefficient. Use `Kd=0.85` for a preliminary size
        estimation.
    Kb : float
        Backpressure correction factor for balanced bellows valves. The value
        can be obtained from the manufacturer or estimated with the help of
        Figure 31 of the API standard 520.
    Kc : float
        Combination correction factor for installations with a rupture disk
        upstream of the PRV. Use `Kc=1.0` if there is no rupture disk, and
        `Kc=0.9` if there is a rupture disk.
    Kv : float
        Viscosity correction factor. Use `Kv=1.0` if the liquid phase has a
        viscosity less or equal than 0.1 Pa·s.

    Returns
    -------
    PRVResult
        Dataclass containing the results of the calculation.

    Examples
    --------
    Estimate the required discharge area of a pressure relief device handling a
    two-phase mixture, according to the API standard 520. The required flow is
    216560 kg/h, the upstream relieving pressure is 5.564 bara, the downstream
    back pressure is 2.045 bara, the overall specific volume of the two-phase
    mixture at upstream relieving conditions is 0.01945 m³/kg, and the overall
    specific volume of the two-phase mixture evaluated at 90% of the upstream
    pressure is 0.02265 m³/kg.
    >>> from polykin.flow import area_relief_2phase
    >>> area_relief_2phase(W=216560, P1=5.564, P2=2.045, v1=0.01945, v9=0.02265)
    PRVResult(Pcf=3.65e+00 bara, critical_flow=True, A=2.45e+04 mm²)
    """
    # Convert pressures from bar to Pa
    P1 *= 1e5
    P2 *= 1e5

    # Omega parameter
    w = 9 * (v9 / v1 - 1)

    # Critical pressure ratio
    ηc = (1 + (1.0446 - 0.0093431 * sqrt(w)) * w ** (-0.56261)) ** (
        -0.70356 + 0.014685 * ln(w)
    )

    # Critical pressure
    Pcf = P1 * ηc
    critical_flow = P2 <= Pcf

    # Mass flux [kg/s.m²]
    if critical_flow:
        G = ηc * sqrt(P1 / (v1 * w))
    else:
        ηa = P2 / P1
        G = (
            sqrt(-2 * (w * ln(ηa) + (w - 1) * (1 - ηa)))
            * sqrt(P1 / v1)
            / (w * (1 / ηa - 1) + 1)
        )

    # Area [mm²]
    A = 277.8 * W / (Kd * Kb * Kc * Kv * G)

    return PRVResult(Pcf / 1e5, critical_flow, A)


def area_relief_2phase_subcooled(
    Q: float,
    P1: float,
    P2: float,
    Ps: float,
    rho1: float,
    rho9: float,
    *,
    Kd: float = 0.65,
    Kb: float = 1.0,
    Kc: float = 1.0,
    Kv: float = 1.0,
) -> PRVResult:
    r"""Calculate the required effective discharge area of a pressure relief
    device for subcooled liquid flow using the omega method.

    The calculation is done according to the API standard 520, Appendix C.2.3.

    This method can also be used for liquids that are saturated as they enter
    the relief device, but no condensable vapor or noncondensable gas should be
    present at the inlet.

    **References**

    * Sizing, Selection, and Installation of Pressure-relieving Devices in
      Refineries: Part I—Sizing and Selection, API Standard 520, 10th ed., 2020.

    Parameters
    ----------
    Q : float
        Required relieving volume flow rate [L/min].
    P1 : float
        Upstream relieving pressure, absolute [bara].
    P2 : float
        Downstream back pressure, absolute [bara].
    Ps : float
        Saturation (bubble) pressure at upstream relieving temperature [bara].
    rho1 : float
        Liquid density at upstream relieving conditions [m³/kg].
    rho9 : float
        Overall density evaluated at 90% of the saturation pressure `Ps` [m³/kg].
        The flash calculation should be carried out isentropically, but an
        isenthalpic (adiabatic) flash is sufficient for low-vapor-content mixtures
        far from the thermodynamic critical point.
    Kd : float
        Effective discharge coefficient. For a preliminary size estimation, use
        `Kd=0.65` for subcooled liquids, and `Kd=0.85` for saturated liquids.
    Kb : float
        Backpressure correction factor for balanced bellows valves. The value
        can be obtained from the manufacturer or estimated with the help of
        Figure C.3 of the API standard 520.
    Kc : float
        Combination correction factor for installations with a rupture disk
        upstream of the PRV. Use `Kc=1.0` if there is no rupture disk, and
        `Kc=0.9` if there is a rupture disk.
    Kv : float
        Viscosity correction factor. Use `Kv=1.0` if the liquid phase has a
        viscosity less or equal than 0.1 Pa·s.

    Returns
    -------
    PRVResult
        Dataclass containing the results of the calculation.

    Examples
    --------
    Estimate the required discharge area of a pressure relief device handling a
    subcooled liquid, according to the API standard 520. The required flow is
    378.5 L/min, the upstream relieving pressure is 20.733 bara, the downstream
    back pressure is 1.703 bara, the saturation pressure at the relieving
    temperature is 7.419 bara, the density of the liquid is 511.3 kg/m³ and the
    density of the two-phase mixture evaluated at 90% of the saturation pressure
    is 262.7 kg/m³.
    >>> from polykin.flow import area_relief_2phase_subcooled
    >>> area_relief_2phase_subcooled(Q=378.5, P1=20.733, P2=1.703, Ps=7.419,
    ...                              rho1=511.3, rho9=262.7)
    PRVResult(Pcf=nan bara, critical_flow=True, A=1.35e+02 mm²)
    """
    # Convert pressures from bar to Pa
    P1 *= 1e5
    P2 *= 1e5
    Ps *= 1e5

    # Omega parameter for saturated liquid
    ws = 9 * (rho1 / rho9 - 1)

    # Transition saturation pressure ratio
    ηst = 2 * ws / (1 + 2 * ws)

    # High/low subcooling region
    ηs = Ps / P1
    ηa = P2 / P1
    high_subcooling = ηs < ηst

    # Mass flux [kg/s.m²]
    if high_subcooling:
        critical_flow = P2 <= Ps
        P = Ps if critical_flow else P2
        Pcf = nan
        G = 1.414 * sqrt(rho1 * (P1 - P))
    else:
        ηc_ = (
            ηs
            * (2 * ws / (2 * ws - 1))
            * (1 - sqrt(1 - 1 / ηs * ((2 * ws - 1) / (2 * ws))))
        )
        ηc = ηc_ if ηs > ηst else ηs
        Pcf = P1 * ηc
        critical_flow = P2 <= Pcf
        η = ηc if critical_flow else ηa
        G = (
            sqrt(2 * (1 - ηs) + 2 * (ws * ηs * ln(ηs / η) - (ws - 1) * (ηs - η)))
            * sqrt(P1 * rho1)
            / (ws * (ηs / η - 1) + 1)
        )

    # Area [mm²]
    A = 16.67 * Q * rho1 / (Kd * Kb * Kc * Kv * G)

    return PRVResult(Pcf / 1e5, critical_flow, A)
