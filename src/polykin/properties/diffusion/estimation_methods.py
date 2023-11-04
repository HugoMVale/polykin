# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

"""This module provides functions to estimate the binary diffusion coefficient
in gas and liquid solutions.
"""

import numpy as np
from typing import Literal, Optional
from math import sqrt

__all__ = ['wilke_chang', 'hayduk_minhas', 'wilke_lee']

# %% Liquid phase: infinite-dilution


def wilke_chang(T: float,
                MA: float,
                MB: float,
                rhoA: float,
                viscB: float,
                phi: float = 1.0
                ) -> float:
    r"""Estimate the infinite-dilution coefficient of a solute A in a liquid
    solvent B, $D^0_{AB}$, using the Wilke-Chang method.

    $$
    D^0_{AB} = 5.9\times 10^{-17}
        \frac{(\phi M_B)^{1/2} T}{\eta_B (M_A/\rho_A)^{0.6}}
    $$

    where the meaning of all symbol is as described below in the parameters
    section. The numerical factor has been adjusted to convert the equation to
    SI units.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 657. 1986.

    Parameters
    ----------
    T : float
        Temperature. Unit = K.
    MA : float
        Molar mass of solute A. Unit = kg/mol.
    MB : float
        Molar mass of solvent B. Unit = kg/mol.
    rhoA : float
        Density of solute A at the normal boiling point, $\rho_A$.
        Unit = kg/m³.
    viscB : float
        Viscostity of solvent B, $\eta_B$. Unit = Pa.s.
    phi : float
        Association factor of solvent B, $\phi$. The following values are
        recomended: {water: 2.6, methanol: 1.9, ethanol: 1.5,
        unassociated: 1.0}.

    Returns
    -------
    float
        Diffusion coefficient of A in B at infinite dilution. Unit = m²/s.
    """
    return 7.4e-12*sqrt(phi*MB*1e3)*T/((viscB*1e3)*(1e6*MA/rhoA)**0.6)


def hayduk_minhas(T: float,
                  method: Literal['paraffin', 'aqueous'],
                  MA: float,
                  rhoA: float,
                  viscB: float,
                  ) -> float:
    r"""Estimate the infinite-dilution coefficient of a solute A in a liquid
    solvent B, $D^0_{AB}$, using the Hayduk-Minhas method.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 657. 1986.

    Parameters
    ----------
    T : float
        Temperature. Unit = K.
    method : Literal['paraffin', 'aqueous']
        Method selection. Chose `paraffin` for normal paraffin solutions and
        `aqueous` for solutes in aqueous solutions.
    MA : float
        Molar mass of solute A. Unit = kg/mol.
    rhoA : float
        Density of solute A at the normal boiling point. Unit = kg/m³.
    viscB : float
        Viscostity of solvent B. Unit = Pa.s.

    Returns
    -------
    float
        Diffusion coefficient of A in B at infinite dilution. Unit = m²/s.
    """
    VA = 1e6*MA/rhoA
    if method == 'paraffin':
        epsilon = 10.2/VA - 0.791
        DAB0 = 13.3e-12*T**1.47*(viscB*1e3)**epsilon/VA**0.71
    elif method == 'aqueous':
        epsilon = 9.58/VA - 1.12
        DAB0 = 1.25e-12*(VA**-0.19 - 0.292)*T**1.52*(viscB*1e3)**epsilon
    else:
        raise ValueError(f"Invalid `method` input: {method}")
    return DAB0

# %% Gas-phase: binary diffusion coefficient


def wilke_lee(T: float,
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
    4th edition, 657. 1986.

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
    omegaD = A/Ts**B + C/np.exp(D*Ts) + E/np.exp(F*Ts) + G/np.exp(H*Ts)

    DAB = 1e-7*(3.03 - 0.98/sqrt(MAB))*T**1.5 / \
        ((P*1e-5)*sqrt(MAB)*sAB**2*omegaD)
    return DAB
