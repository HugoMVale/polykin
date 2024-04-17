# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from math import sqrt
from typing import Literal

__all__ = ['DL_Wilke_Chang',
           'DL_Hayduk_Minhas',
           ]

# %% Liquid phase: infinite-dilution


def DL_Wilke_Chang(T: float,
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

    where the meaning of all symbols is as described below in the parameters
    section. The numerical factor has been adjusted to convert the equation to
    SI units.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 598.

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

    See also
    --------
    * [`DL_Hayduk_Minhas`](DL_Hayduk_Minhas.md): alternative method.

    Examples
    --------
    Estimate the diffusion coefficient of vinyl chloride through liquid water.

    >>> from polykin.properties.diffusion import DL_Wilke_Chang
    >>> D = DL_Wilke_Chang(
    ...     T=298.,         # temperature
    ...     MA=62.5e-3,     # molar mass of vinyl chloride
    ...     MB=18.0e-3,     # molar mass of water
    ...     rhoA=910.,      # density of vinyl chloride at the boiling point
    ...     viscB=0.89e-3,  # viscosity of water at solution temperature
    ...     phi=2.6         # association factor for water (see docstring)
    ...     )
    >>> print(f"{D:.2e} m²/s")
    1.34e-09 m²/s

    """
    return 7.4e-12*sqrt(phi*MB*1e3)*T/((viscB*1e3)*(1e6*MA/rhoA)**0.6)


def DL_Hayduk_Minhas(T: float,
                     method: Literal['paraffin', 'aqueous'],
                     MA: float,
                     rhoA: float,
                     viscB: float,
                     ) -> float:
    r"""Estimate the infinite-dilution coefficient of a solute A in a liquid
    solvent B, $D^0_{AB}$, using the Hayduk-Minhas method.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 602.

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

    See also
    --------
    * [`DL_Wilke_Chang`](DL_Wilke_Chang.md): alternative method.

    Examples
    --------
    Estimate the diffusion coefficient of vinyl chloride through liquid water.

    >>> from polykin.properties.diffusion import DL_Hayduk_Minhas
    >>> D = DL_Hayduk_Minhas(
    ...     T=298.,           # temperature
    ...     method='aqueous', # equation for aqueous solutions
    ...     MA=62.5e-3,       # molar mass of vinyl chloride
    ...     rhoA=910.,        # density of vinyl chloride at the boiling point
    ...     viscB=0.89e-3     # viscosity of water at solution temperature
    ...     )
    >>> print(f"{D:.2e} m²/s")
    1.26e-09 m²/s

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
