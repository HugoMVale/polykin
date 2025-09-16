# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from math import sqrt

__all__ = ['area_relief_gas',
           'area_relief_liquid']


def area_relief_gas(W: float,
                    P1: float,
                    P2: float,
                    T: float,
                    k: float,
                    M: float,
                    Z: float = 1.0,
                    steam: bool = False,
                    Kd: float = 0.975,
                    Kb: float = 1.0,
                    Kc: float = 1.0,
                    KSH: float = 1.0
                    ) -> dict:
    r"""Calculate the required effective discharge area of a pressure relief
    device in gas (or vapor) or steam service. 

    The calculation is done according to the API standard 520.

    **References**

    * Sizing, Selection, and Installation of Pressure-relieving Devices in
      Refineries: Part I—Sizing and Selection, API Standard 520, 8th ed., 2008.

    Parameters
    ----------
    W : float
        Required relieving mass flow rate (kg/h).
    P1 : float
        Relieving pressure, absolute (bara).
    P2 : float
        Back pressure, absolute (bara).
    T : float
        Absolute relieving temperature of the gas at the valve inlet (K).
    k : float
        Ideal gas specific heat ratio at relieving temperature.
    M : float
        Molar mass of gas (g/mol).
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
        Backpressure correction factor for balanced bellows valves.
    Kc : float
        Combination correction factor for installations with a rupture disk
        upstream of the PRV. Use `Kc=1.0` if there is no rupture disk, and
        `Kc=0.9` if there is a rupture disk.
    KSH : float
        Steam superheat correction factor, equal to 1.0 for saturated steam.

    Returns
    -------
    dict
        Dictionary of results, including the critical flow nozzle pressure 
        `Pcf` (bara), the flow condition `critical_flow` (bool), and the 
        effective discharge area `A` (mm²).

    Examples
    --------
    Estimate the required discharge area of a pressure relief valve in gas 
    service, using the API standard 520. The required flow is 24270 kg/h, the
    relieving pressure is 6.7 bara, the back pressure is 0 barg, the relieving
    temperature is 348 K, the gas specific heat ratio is 1.11, the gas molar
    mass is 51 g/mol, and the compressibility factor is 0.9.
    >>> from polykin.flow import area_relief_gas
    >>> res = area_relief_gas(W=24270, P1=6.7, P2=1.01325,
    ...                       T=348, k=1.11, M=51, Z=0.9)
    >>> print(f"Effective discharge area: {res['A']:.0f} mm²")
    Effective discharge area: 3699 mm²
    """

    # Convert pressures from bar to kPa
    P1 *= 1e2
    P2 *= 1e2

    # Critical flow nozzle pressure
    Pcf = P1 * (2/(k + 1))**(k/(k - 1))
    critical_flow = (P2 <= Pcf)

    if critical_flow:
        if steam:
            if P1 <= 10339.0:
                KN = 1.0
            elif P1 < 22057.0:
                KN = (0.02764*P1 - 1000)/(0.03324*P1 - 1061)
            else:
                raise ValueError('`P1` out of range.')
            A = 190.5*W/(P1*Kd*Kb*Kc*KN*KSH)
        else:
            C = 0.03948*sqrt(k*(2/(k + 1))**((k + 1)/(k - 1)))
            A = W/(C*Kd*P1*Kb*Kc)*sqrt(T*Z/M)
    else:
        r = P2/P1
        F2 = sqrt((k/(k - 1))*r**(2/k) * ((1 - r**((k - 1)/k))/(1 - r)))
        A = 17.9*W/(F2*Kd*Kc)*sqrt(T*Z/(M*P1*(P1 - P2)))

    result = {
        'Pcf': Pcf,  # kPa
        'critical_flow': critical_flow,
        'A': A  # mm²
    }

    return result


def area_relief_liquid(Q: float,
                       P1: float,
                       P2: float,
                       mu: float,
                       Gl: float,
                       Kd: float = 0.65,
                       Kw: float = 1.0,
                       Kc: float = 1.0,
                       ) -> float:
    r"""Calculate the required effective discharge area of a pressure relief
    device in liquid service. 

    The calculation is done according to the API standard 520, assuming the
    valves are designed in accordance with the ASME code.

    **References**

    * Sizing, Selection, and Installation of Pressure-relieaving Devices in
      Refineries: Part I—Sizing and Selection, API Standard 520, 8th ed., 2008.

    Parameters
    ----------
    Q : float
        Required relieving volume flow rate (L/min).
    P1 : float
        Relieving pressure (bar). Can be absolute or gauge, as long as it is 
        consistent with `P2`.
    P2 : float
        Back pressure (bar). Can be absolute or gauge, as long as it is 
        consistent with `P1`.
    mu : float
        Viscosity at relieving temperature (cP).
    Gl : float
        Relative density of the liquid at relieving temperature with respect to
        water at standard conditions.
    Kd : float
        Effective discharge coefficient. Use `Kd=0.65` when sizing a PRV and
        `Kd=0.62` when sizing a rupture disk without PRV.
    Kw : float
        Backpressure correction factor for balanced bellows valves.
    Kc : float
        Combination correction factor for installations with a rupture disk
        upstream of the PRV. Use `Kc=1.0` if there is no rupture disk, and
        `Kc=0.9` if there is a rupture disk.

    Returns
    -------
    float
        Effective discharge area (mm²).

    Examples
    --------
    Estimate the required discharge area of a pressure relief valve in liquid 
    service, using the API standard 520. The required flow is 6814 L/min, the
    relieving pressure is 18.96 barg, the back pressure is 3.45 barg, the liquid
    relative density is 0.9, the liquid viscosity is 396 cP, and the valve back 
    pressure correction factor is 0.97.
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
            Re = 18800*Q*Gl/(mu*sqrt(A))
            Kv = 1/(0.9935 + 2.878/Re**0.5 + 342.75/Re**1.5)

        A = 11.78*Q/(Kd*Kw*Kc*Kv)*sqrt(Gl/(P1 - P2))

        if abs((A - A_old)/A) < 1e-5:
            converged = True
            break
        else:
            A_old = A

    if not converged:
        raise ValueError(f"Failed to converge after {MAX_ITER} iterations.")

    return A


def area_relief_biphasic():
    pass
