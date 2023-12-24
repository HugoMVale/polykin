# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""This module implements methods to estimate the binary diffusion
coefficient in gas solutions.
"""

from typing import Optional

from numpy import exp, sqrt

__all__ = ['DV_wilke_lee']


def DV_wilke_lee(T: float,
                 P: float,
                 MA: float,
                 MB: float,
                 rhoA: float,
                 rhoB: Optional[float],
                 TA: float,
                 TB: Optional[float]
                 ) -> float:
    r"""Estimate the mutual diffusion coefficient of a binary gas mixture,
    $D_{AB}$, using the Wilke-Lee method.

    !!! note
        If air is one of the components of the mixture, arguments `TB` and
        `rhoB` should both be set to `None`.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 587.

    Parameters
    ----------
    T : float
        Temperature. Unit = K.
    P : float
        Pressure. Unit = Pa.
    MA : float
        Molar mass of component A. Unit = kg/mol.
    MA : float
        Molar mass of component B. Unit = kg/mol.
    rhoA : float
        Density of component A at the normal boiling point. Unit = kg/m³.
    rhoB : float | None
        Density of component B at the normal boiling point. If `None`,
        air is assumeed to be the component B. Unit = kg/m³.
    TA : float
        Normal boiling temperature of component A. Unit = K.
    TB : float | None
        Normal boiling temperature of component B. If `None`, air is assumeed
        to be the component B. Unit = K.

    Returns
    -------
    float
        Binary diffusion coefficient. Unit = m²/s.
    """

    MAB = 1e3*2/(1/MA + 1/MB)

    # ϵ_AB and σ_AB
    eA = 1.15*TA
    sA = 1.18*(1e6*MA/rhoA)**(1/3)
    if rhoB is None and TB is None:
        # values for air
        sB = 3.62
        eB = 97.
    elif rhoB is not None and TB is not None:
        sB = 1.18*(1e6*MB/rhoB)**(1/3)
        eB = 1.15*TB
    else:
        raise ValueError(
            "Invalid input. `rhoB` and `TB` must both be None or floats.")

    eAB = sqrt(eA*eB)
    sAB = (sA + sB)/2

    # Ω_D
    Ts = T/eAB
    A = 1.06036
    B = 0.15610
    C = 0.19300
    D = 0.47635
    E = 1.03587
    F = 1.52996
    G = 1.76474
    H = 3.89411
    omegaD = A/Ts**B + C/exp(D*Ts) + E/exp(F*Ts) + G/exp(H*Ts)

    DAB = 1e-2*(3.03 - 0.98/sqrt(MAB))*T**1.5 / (P*sqrt(MAB)*sAB**2*omegaD)
    return DAB
