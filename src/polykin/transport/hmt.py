# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from typing import Literal

import numpy as np
from numpy import cbrt, inf, sqrt

from polykin.transport.flow import fD_Haaland
from polykin.utils.tools import check_range_warn

__all__ = ['Nu_tube',
           'Nu_cylinder',
           'Nu_cylinder_bank',
           'Nu_cylinder_free',
           'Nu_sphere',
           'Nu_sphere_free',
           'Nu_drop',
           'Nu_plate',
           'Nu_plate_free',
           'Nu_tank',
           'Nu_combined',
           'U_plane_wall',
           'U_cylindrical_wall'
           ]


def Nu_tube(Re: float,
            Pr: float,
            D_L: float = 0.0,
            er: float = 0.0
            ) -> float:
    r"""Calculate the internal Nusselt number for flow through a circular tube.

    For laminar flow, the average Nusselt number $\overline{Nu}=\bar{h}D/k$ is
    estimated by the following expression:

    $$ \overline{Nu} = 3.66 + \frac{0.0668 (D/L) Re Pr}{1 + 0.04 [(D/L) Re Pr]^{2/3}} $$

    where $Re$ is the Reynolds number and $Pr$ is the Prandtl number. This
    correlation presumes constant surface temperature and a thermal entry length,
    thereby leading to conservative (underestimated) $\overline{Nu}$ values.

    For turbulent flow, the Nusselt number is estimated by the following
    expression:

    $$  Nu = \frac{(f_D/8)(Re - 1000)Pr}{1 + 12.7 (f_D/8)^{1/2} (Pr^{2/3} - 1)} $$

    \begin{bmatrix}
    3000 < Re < 5 \times 10^{6} \\
    0.5 < Pr < 2000 \\
    L/D \gtrsim 10
    \end{bmatrix}

    where $f_D$ is the Darcy friction factor. 

    In both flow regimes, the properties are to be evaluated at the mean fluid
    temperature.

    **References**

    * Gnielinski, Volker. "New equations for heat and mass transfer in
      turbulent pipe and channel flow", International Chemical Engineering
      16.2 (1976): 359-367.
    * Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
      mass transfer", 4th edition, 1996, p. 444, 445.

    Parameters
    ----------
    Re : float
        Reynolds number based on tube internal diameter.
    Pr : float
        Prandtl number.
    D_L : float
        Diameter-to-length ratio.
    er : float
        Relative pipe roughness.

    Returns
    -------
    float
        Nusselt number.

    See also
    --------
    * [`Nu_cylinder`](Nu_cylinder.md): related method for external cross flow.

    Examples
    --------
    Estimate the internal heat transfer coefficient for water flowing at 2 m/s
    through a long smooth tube with an internal diameter of 25 mm.
    >>> from polykin.transport import Nu_tube
    >>> rho = 1e3  # kg/m³
    >>> mu = 1e-3  # Pa.s
    >>> cp = 4.2e3 # J/kg/K
    >>> k = 0.6    # W/m/K
    >>> v = 2.0    # m/s
    >>> D = 25e-3  # m
    >>> Re = rho*v*D/mu
    >>> Pr = cp*mu/k
    >>> Nu = Nu_tube(Re, Pr, D_L=0, er=0)
    >>> h = Nu*k/D
    >>> print(f"h={h:.1e} W/m²·K")
    h=7.8e+03 W/m²·K
    """
    if Re < 2.3e3:
        return 3.66 + (0.0668*D_L*Re*Pr)/(1 + 0.04*(D_L*Re*Pr)**(2/3))
    else:
        fD = fD_Haaland(Re, er)
        return (fD/8)*(Re - 1e3)*Pr/(1 + 12.7*sqrt(fD/8)*(Pr**(2/3) - 1))


def Nu_plate(Re: float, Pr: float) -> float:
    r"""Calculate the Nusselt number for parallel flow over an isothermal flat
    plate.

    The average Nusselt number $\overline{Nu}=\bar{h}L/k$ is estimated by the
    following expressions:    

    $$ \overline{Nu} =
    \begin{cases}
    0.664 Re^{1/2} Pr^{1/3} ,& Re > 5 \times 10^5 \\
    (0.037 Re^{4/5} - 871)Pr^{1/3} ,& 5 \times 10^5 < Re \lesssim 10^8
    \end{cases} $$

    $$ [0.6 < Pr < 60] $$

    where $Re$ is the Reynolds number and $Pr$ is the Prandtl number.

    **References**

    * Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
      mass transfer", 4th edition, 1996, p. 354-356.

    Parameters
    ----------
    Re : float
        Reynolds number based on plate length.
    Pr : float
        Prandtl number.

    Returns
    -------
    float
        Nusselt number.
    """
    check_range_warn(Pr, 0.6, 60, 'Pr')

    Re_c = 5e5
    if Re < Re_c:
        return 0.664*Re**(1/2)*Pr**(1/3)
    else:
        A = 0.037*Re_c**(4/5) - 0.664*Re_c**(1/2)
        return (0.037*Re**(4/5) - A)*Pr**(1/3)


def Nu_cylinder(Re: float, Pr: float) -> float:
    r"""Calculate the Nusselt number for cross flow over a circular cylinder.

    The average Nusselt number $\overline{Nu}=\bar{h}D/k$ is estimated by the
    following expression:

    $$ \overline{Nu} = 0.3 + \frac{0.62 Re^{1/2} Pr^{1/3}}
    {\left[1 + (0.4/Pr)^{2/3}\right]^{1/4}} 
    \left[1 + \left(\frac{Re}{282 \times 10^3}\right)^{5/8}\right]^{4/5} $$

    $$ \left[  Re Pr > 0.2 \right] $$

    where $Re$ is the Reynolds number and $Pr$ is the Prandtl number. The 
    properties are to be evaluated at the film temperature.

    **References**

    * Churchill, S. W., and Bernstein, M. "A Correlating Equation for Forced
      Convection From Gases and Liquids to a Circular Cylinder in Crossflow",
      ASME. J. Heat Transfer. May 1977; 99(2): 300.
    * Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
      mass transfer", 4th edition, 1996, p. 370.

    Parameters
    ----------
    Re : float
        Reynolds number based on cylinder diameter.
    Pr : float
        Prandtl number.

    Returns
    -------
    float
        Nusselt number.

    See also
    --------
    - [`Nu_cylinder_bank`](Nu_cylinder_bank.md): specific method for a bank of
      tubes.
    - [`Nu_cylinder_free`](Nu_cylinder_free.md): related method for free
      convection.

    Examples
    --------
    Estimate the external heat transfer coefficient for water flowing at 2 m/s
    across a DN25 pipe.
    >>> from polykin.transport import Nu_cylinder
    >>> rho = 1e3   # kg/m³
    >>> mu = 1e-3   # Pa.s
    >>> cp = 4.2e3  # J/kg/K
    >>> k = 0.6     # W/m/K
    >>> v = 2.0     # m/s
    >>> D = 33.7e-3 # m (OD from pipe chart)
    >>> Re = rho*v*D/mu
    >>> Pr = cp*mu/k
    >>> Nu = Nu_cylinder(Re, Pr)
    >>> h = Nu*k/D
    >>> print(f"h={h:.1e} W/m²·K")
    h=7.0e+03 W/m²·K
    """
    check_range_warn(Re*Pr, 0.2, inf, 'Re*Pr')

    return 0.3 + 0.62*Re**(1/2)*Pr**(1/3)/(1 + (0.4/Pr)**(2/3))**(1/4) \
        * (1 + (Re/282e3)**(5/8))**(4/5)


def Nu_cylinder_free(Ra: float, Pr: float) -> float:
    r"""Calculate the Nusselt number for free convection on a horizontal 
    cylinder.

    The average Nusselt number $\overline{Nu}=\bar{h}D/k$ is estimated by the
    following expression:

    $$ \overline{Nu} = \left(0.6 + \frac{0.387 Ra^{1/6}}
       {[1 + (0.559/Pr)^{9/16}]^{8/27}}\right)^2 $$

    $$ \left[  Ra \lesssim 10^{12} \right] $$

    where $Ra$ is the Rayleigh number and $Pr$ is the Prandtl number. The 
    properties are to be evaluated at the film temperature.

    **References**

    * Churchill, Stuart W., and Humbert HS Chu. "Correlating equations for
      laminar and turbulent free convection from a horizontal cylinder", 
      International Journal of Heat and Mass Transfer 18, no. 9 (1975): 1049-1053.
    * Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
      mass transfer", 4th edition, 1996, p. 502.

    Parameters
    ----------
    Ra : float
        Rayleigh number based on cylinder diameter.
    Pr : float
        Prandtl number.

    Returns
    -------
    float
        Nusselt number.

    See also
    --------
    - [`Nu_cylinder`](Nu_cylinder.md): related method for forced convection.

    Examples
    --------
    Estimate the external heat transfer coefficient for a DN25 tube with a
    surface temperature of 330 K immersed in water with a bulk temperature of
    290 K.
    >>> from polykin.transport import Nu_cylinder_free
    >>> rho = 1.0e3   # kg/m³ (properties at 310 K)
    >>> mu = 0.70e-3  # Pa.s
    >>> cp = 4.2e3    # J/kg/K
    >>> k = 0.63      # W/m/K
    >>> beta = 362e-6 # 1/K
    >>> D = 33.7e-3   # m (OD from pipe chart)
    >>> g = 9.81      # m/s²
    >>> Pr = cp*mu/k
    >>> Gr = g*beta*(330-290)*D**3/(mu/rho)**2
    >>> Ra = Gr*Pr
    >>> Nu = Nu_cylinder_free(Ra, Pr)
    >>> h = Nu*k/D
    >>> print(f"h={h:.1e} W/m²·K")
    h=1.1e+03 W/m²·K
    """
    check_range_warn(Ra, 0, 1e12, 'Ra')

    return (0.6 + 0.387*Ra**(1/6) / (1 + (0.559/Pr)**(9/16))**(8/27))**2


def Nu_sphere(Re: float, Pr: float, mur: float) -> float:
    r"""Calculate the Nusselt number for flow around an isolated sphere.

    The average Nusselt number $\overline{Nu}=\bar{h}D/k$ is estimated by the
    following expression:

    $$ \overline{Nu} = 2 + ( 0.4 Re^{1/2} + 0.06 Re^{2/3} ) Pr^{0.4}
                           \left(\frac{\mu}{\mu_s}\right)^{1/4} $$

    \begin{bmatrix}
    3.5 < Re < 7.6 \times 10^{4} \\
    0.71 < Pr < 380 \\
    1.0 < (\mu/\mu_s) 3.2
    \end{bmatrix}

    where $Re$ is the sphere Reynolds number, $Pr$ is the Prandtl number, 
    $\mu$ is the bulk viscosity, and $\mu_s$ is the surface viscosity. All 
    properties are to be evaluated at the bulk temperature, except $\mu_s$.

    **References**

    * Whitaker, S. (1972), "Forced convection heat transfer correlations for
      flow in pipes, past flat plates, single cylinders, single spheres, and
      for flow in packed beds and tube bundles", AIChE J., 18: 361-371.
    * Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
      mass transfer", 4th edition, 1996, p. 374.

    Parameters
    ----------
    Re : float
        Reynolds number based on sphere diameter.
    Pr : float
        Prandtl number.
    mur : float
        Ratio of bulk viscosity to surface viscosity, $\mu/\mu_s$.

    Returns
    -------
    float
        Nusselt number.

    See also
    --------
    * [`Nu_drop`](Nu_drop.md): specific method for drops.
    """
    check_range_warn(Re, 3.5, 7.6e4, 'Re')
    check_range_warn(Pr, 0.71, 380, 'Pr')
    check_range_warn(mur, 1.0, 3.2, 'mur')

    return 2 + (0.4*Re**(1/2) + 0.06*Re**(2/3))*Pr**0.4*(mur)**(1/4)


def Nu_drop(Re: float, Pr: float) -> float:
    r"""Calculate the Nusselt number for a single freely falling drop.

    The average Nusselt number $\overline{Nu}=\bar{h}D/k$ is estimated by the
    following expression:

    $$ \overline{Nu} = 2 + 0.6 Re^{1/2} Pr^{1/3} $$

    \begin{bmatrix}
    0 < Re < 1000 \\
    0.7 < Pr < 100? \\
    \end{bmatrix}

    where $Re$ is the drop Reynolds number and $Pr$ is the Prandtl number. The
    exact range for $Pr$ in this context is unspecified. All properties are to
    be evaluated at the bulk temperature. This correlation was developed for use
    in spray drying applications.

    **References**

    * Ranz, W. and Marshall, W. (1952) "Evaporation from Drops", Chemical
      Engineering Progress, 48, 141-146.
    * Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
      mass transfer", 4th edition, 1996, p. 374.

    Parameters
    ----------
    Re : float
        Reynolds number based on drop diameter.
    Pr : float
        Prandtl number.

    Returns
    -------
    float
        Nusselt number.

    See also
    --------
    * [`Nu_sphere`](Nu_sphere.md): generic method for spheres.

    Examples
    --------
    Estimate the Nusselt number for a 1 mm styrene droplet falling in air.
    >>> from polykin.transport import Nu_drop
    >>> D = 1e-3    # m
    >>> vt = 3.8    # m/s (from vt_sphere)
    >>> rho = 1.2   # kg/m³
    >>> mu = 1.6e-5 # Pa.s
    >>> Pr = 0.7    # from table of air properties
    >>> Re = rho*vt*D/mu
    >>> Nu = Nu_drop(Re, Pr)
    >>> print(f"Nu={Nu:.1f}")
    Nu=11.0
    """
    check_range_warn(Re, 0, 1000, 'Re')
    check_range_warn(Pr, 0.7, 100, 'Pr')

    return 2 + 0.6*Re**(1/2)*Pr**(1/3)


def Nu_tank(
    surface: Literal['wall', 'bottom-head', 'helical-coil', 'harp-coil-0', 'harp-coil-45'],
    impeller: Literal['4BF', '4BP', '6BD', 'HE3', 'PROP', 'anchor', 'helical-ribbon'],
    Re: float,
    Pr: float,
    mur: float,
    D_T: float = 1/3,
    H_T: float = 1.0,
    L_Ls: float = 1.0,
    d_T: float = 0.04,
    P_D: float = 1.0,
    nb: int = 2
) -> float:
    r"""Calculate the Nusselt number for a stirred tank.

    This function calculates the Nusselt number based on impeller and surface 
    type, and fluid dynamics parameters for a stirred tank, according to
    the correlations in chapter 14.4 of the Handbook of Industrial Mixing.

    **References**

    * Penney, W. R. and Atiemo-Obeng, V. A. "Heat Transfer" in "Handbook of
      Industrial Mixing: Science and Practice", Wiley, 2004.

    Parameters
    ----------
    surface : Literal['wall', 'bottom-head', 'helical-coil', 'harp-coil-0', 'harp-coil-45']
        Heat transfer surface type.
    impeller : Literal['4BF', '4BP', '6BD', 'HE3', 'PROP', 'anchor', 'helical-ribbon']
        Impeller type.
    Re : float
        Impeller Reynolds number.
    Pr : float
        Prandtl number.
    mur : float
        Ratio of bulk viscosity to surface viscosity, $\mu/\mu_s$.
    D_T : float
        Ratio of impeller diameter to tank diameter, $D/T$.
    H_T : float
        Ratio of liquid height to tank diameter, $H/T$.
    L_Ls : float
        Ratio of height of impeller blade to standard value, $L/L_s$.
    d_T : float, optional
        Ratio of coil tube outer diameter to tank diameter, $d/T$.
    P_D : float, optional
        Ratio of impeller blade pitch to impeller diameter.
    nb : int, optional
        Number of baffles or vertical tubes acting as baffles.

    Returns
    -------
    float
        Nusselt number. Characteristic length depends on the surface type.

    Examples
    --------
    Estimate the internal heat transfer coefficient for a 2-m diameter stirred
    tank equiped with a HE3 impeller operated at 120 rpm. Assume water properties
    and default geometry.
    >>> from polykin.transport import Nu_tank
    >>> T = 2.0    # m
    >>> D = T/3    # m
    >>> rho = 1e3  # kg/m³
    >>> mu = 1e-3  # Pa.s
    >>> k  = 0.6   # W/m.K
    >>> cp = 4.2e3 # J/kg.K
    >>> Re = (120/60) * D**2 * rho / mu
    >>> Pr = mu*cp/k
    >>> mur = 1.0  # neglect temperature correction
    >>> Nu = Nu_tank('wall', 'HE3', Re, Pr, mur)
    >>> h = Nu*k/T
    >>> print(f"h={h:.1e} W/m²·K")
    h=1.6e+03 W/m²·K
    """

    # Default parameters
    K = 0.
    a = 2/3
    b = 1/3
    c = 0.14
    Gc = 1.

    impeller_error = False
    if surface == 'wall':
        if impeller == '6BD':
            K = 0.74
            Gc = H_T**-0.15 * L_Ls**0.2
        elif impeller == '4BF':
            K = 0.66
            Gc = H_T**-0.15 * L_Ls**0.2
        elif impeller == '4BP':
            K = 0.45
            Gc = H_T**-0.15 * L_Ls**0.2
        elif impeller == 'HE3':
            K = 0.31
            Gc = H_T**-0.15
        elif impeller == 'PROP':
            K = 0.50
            Gc = H_T**-0.15 * 1.29*P_D/(0.29 + P_D)
        elif impeller == 'anchor':
            if Re < 12:
                K = 0.
            elif Re >= 12 and Re < 100:
                K = 0.69
                a = 1/2
            elif Re >= 100:
                K = 0.32
        elif impeller == 'helical-ribbon':
            if Re < 13:
                K = 0.94
                a = 1/3
            elif Re >= 13 and Re < 210:
                K = 0.61
                a = 1/2
            else:
                K = 0.25
        else:
            impeller_error = True
    elif surface == 'bottom-head':
        if impeller == '6BD':
            K = 0.50
            Gc = H_T**-0.15 * L_Ls**0.2
        elif impeller == '4BF':
            K = 0.40
            Gc = H_T**-0.15 * L_Ls**0.2
        elif impeller == '4BP':
            K = 1.08
            Gc = H_T**-0.15 * L_Ls**0.2
        elif impeller == 'HE3':
            K = 0.90
            Gc = H_T**-0.15
        else:
            impeller_error = True
    elif surface == 'helical-coil':
        if impeller == 'PROP':
            K = 0.016
            a = 0.67
            b = 0.37
            Gc = (D_T/(1/3))**0.1 * (d_T/0.04)**0.5
        elif impeller == '6BD':
            K = 0.03
            Gc = H_T**-0.15 * L_Ls**0.2 * (D_T/(1/3))**0.1 * (d_T/0.04)**0.5
        else:
            impeller_error = True
    elif surface == 'harp-coil-0':
        if impeller == '4BF':
            K = 0.06  # the text mentions this value might be overestimated
            a = 0.65
            b = 0.3
            Gc = H_T**-0.15 * L_Ls**0.2 * (D_T/(1/3))**0.33 * (2/nb)**0.2
        else:
            impeller_error = True
    elif surface == 'harp-coil-45':
        if impeller == '6BD':
            # quite some doubts regarding this equation
            K = 0.021
            a = 0.67
            b = 0.4
            Gc = H_T**-0.15 * L_Ls**0.2 * (D_T/(1/3))**0.33 * (2/nb)**0.2
        else:
            impeller_error = True
    else:
        raise ValueError(f"Invalid heat transfer `surface`: {surface}.")

    if impeller_error:
        raise ValueError(
            f"Invalid combination of `surface`={surface} and `impeller`={impeller}.")

    return K * Re**a * Pr**b * mur**c * Gc


def Nu_cylinder_bank(v: float,
                     rho: float,
                     mu: float,
                     Pr: float,
                     Prs: float,
                     aligned: bool,
                     D: float,
                     ST: float,
                     SL: float,
                     NL: int) -> float:
    r"""Calculate the Nusselt number for cross flow over a bank of tubes.

    For flow across a bank of aligned or staggered tubes, the average Nusselt
    number $\overline{Nu}=\bar{h}D/k$ is estimated by the following
    expression:

    $$ \overline{Nu} = 
        C_2 C Re_{max}^m Pr^{0.36} \left(\frac{Pr}{Pr_s} \right)^{1/4} $$

    where $Re_{max}$ is the Reynolds number based on the maximum fluid velocity
    within the bank of tubes, and $Pr$ is the Prandtl number. Additionally,
    $C_2$, $C$ and $m$ are tabulated constants that depend on the tube bundle
    configuration. 

    **References**

    * Žukauskas, Algirdas. "Heat transfer from tubes in crossflow", Advances
      in Heat Transfer. Vol. 8. Elsevier, 1972. 93-160.
    * Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
      mass transfer", 4th edition, 1996, p. 376-381.

    Parameters
    ----------
    v : float
        Nominal velocity outside the tube bank, $v_{\infty}$ (m/s).
    rho : float
        Density (kg/m³).
    mu : float
        Viscosity (Pa·s).
    Pr : float
        Prandtl number.
    Prs : float
        Prandtl number at the surface of the tubes.
    aligned : bool
        Aligned or staggered tubes.
    D : float
        Diameter (m).
    ST : float
        Transversal pitch (m).
    SL : float
        Longitudinal pitch (m).
    NL : int
        Number of rows of tubes.

    Returns
    -------
    float
        Average Nusselt number of the tube bank.

    See also
    --------
    * [`Nu_cylinder`](Nu_cylinder.md): specific method for a single cylinder.

    Examples
    --------
    Estimate the external heat transfer coefficient for air flowing across a bank
    of staggered tubes. The tube outside diameter is 16.4 mm, the longitudinal
    pitch is 34.3 mm, the transversal pitch is 31.3 mm, and the number of rows
    is 7. The tube surface temperature is 70°C, while the upstream air stream
    has a temperature of 15°C and a velocity of 6 m/s.
    >>> from polykin.transport import Nu_cylinder_bank
    >>> v = 6.0      # m/s
    >>> ST = 31.3e-3 # m
    >>> SL = 34.3e-3 # m
    >>> D = 16.4e-3  # m
    >>> NL = 7
    >>> rho = 1.2    # kg/m³ (at 15°C, neglecting temperature increase)
    >>> mu = 1.8e-5  # Pa·s
    >>> k = 0.025    # W/m·K
    >>> Pr = 0.72
    >>> Prs = 0.71   # (at 70°C)
    >>> Nu = Nu_cylinder_bank(v, rho, mu, Pr, Prs, False, D, ST, SL, NL)
    >>> h = Nu*k/D
    >>> print(f"h={h:.1e} W/m²·K")
    h=1.4e+02 W/m²·K
    """

    # Maximum fluid velocity
    vmax = v * ST/(ST - D)
    if not aligned:
        SD = sqrt(SL**2 + (ST/2)**2)
        if SD < (ST + D)/2:
            vmax = v*ST/(2*(SD - D))

    Re_max = rho*vmax*D/mu

    check_range_warn(Re_max, 1e1, 2e6, 'Re_max')
    check_range_warn(Pr, 0.7, 500, 'Pr')

    # Nu for NL>=20
    if aligned:
        if Re_max < 1e2:
            C = 0.8
            m = 0.4
        elif Re_max >= 1e2 and Re_max < 1e3:
            C = 0.51
            m = 0.50
        elif Re_max >= 1e3 and Re_max < 2e5:
            C = 0.27
            m = 0.63
        else:
            C = 0.021
            m = 0.84
    else:
        if Re_max < 1e2:
            C = 0.9
            m = 0.4
        elif Re_max >= 1e2 and Re_max < 1e3:
            C = 0.51
            m = 0.50
        elif Re_max >= 1e3 and Re_max < 2e5:
            m = 0.60
            if ST/SL < 2:
                C = 0.35*(ST/SL)**0.2
            else:
                C = 0.40
        else:
            C = 0.022
            m = 0.84

    Nu = C * Re_max**m * Pr**0.36 * (Pr/Prs)**(1/4)

    # Correction for NL<20
    if NL < 20:
        if aligned:
            C2 = np.interp(
                NL,
                [1, 2, 3, 4, 5, 7, 10, 13, 16, 20],
                [0.70, 0.80, 0.86, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 1.00])
        else:
            C2 = np.interp(
                NL,
                [1, 2, 3, 4, 5, 7, 10, 13, 16, 20],
                [0.64, 0.76, 0.84, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99, 1.00])
    else:
        C2 = 1.

    return C2*Nu


def Nu_sphere_free(Ra: float, Pr: float) -> float:
    r"""Calculate the Nusselt number for free convection on a sphere.

    The average Nusselt number $\overline{Nu}=\bar{h}D/k$ is estimated by the
    following expression:

    $$ \overline{Nu} = 2 + \frac{0.589 Ra^{1/4}}{[1 + (0.469/Pr)^{9/16}]^{4/9}} $$

    \begin{bmatrix}
    Ra \lesssim 10^{11} \\
    Pr \ge  0.7 \\
    \end{bmatrix}

    where $Ra$ is the Rayleigh number and $Pr$ is the Prandtl number. The 
    properties are to be evaluated at the film temperature.

    **References**

    * Churchill, S.W, "Free convection around immersed bodies", in Heat Exchange
      Design Handbook, Section 2.5.7, Hemisphere Publishing, New York, 1983.
    * Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
      mass transfer", 4th edition, 1996, p. 502.

    Parameters
    ----------
    Ra : float
        Rayleigh number based on sphere diameter.
    Pr : float
        Prandtl number.

    Returns
    -------
    float
        Nusselt number.

    See also
    --------
    - [`Nu_sphere`](Nu_sphere.md): related method for forced convection.

    Examples
    --------
    Estimate the external heat transfer coefficient for a 50 mm sphere with a
    surface temperature of 330 K immersed in water with a bulk temperature of
    290 K.
    >>> from polykin.transport import Nu_sphere_free
    >>> rho = 1.0e3   # kg/m³
    >>> mu = 0.70e-3  # Pa.s
    >>> cp = 4.2e3    # J/kg/K
    >>> k = 0.63      # W/m/K
    >>> beta = 362e-6 # 1/K
    >>> D = 50e-3     # m
    >>> g = 9.81      # m/s²
    >>> Pr = cp*mu/k
    >>> Gr = g*beta*(330-290)*D**3/(mu/rho)**2
    >>> Ra = Gr*Pr
    >>> Nu = Nu_sphere_free(Ra, Pr)
    >>> h = Nu*k/D
    >>> print(f"h={h:.1e} W/m²·K")
    h=7.8e+02 W/m²·K
    """
    check_range_warn(Ra, 0, 1e11, 'Ra')
    check_range_warn(Pr, 0.7, inf, 'Pr')

    return 2 + 0.589*Ra**(1/4) / (1 + (0.469/Pr)**(9/16))**(4/9)


def Nu_plate_free(orientation: Literal['vertical',
                                       'horizontal-upper-heated',
                                       'horizontal-lower-heated'],
                  Ra: float,
                  Pr: float | None = None
                  ) -> float:
    r"""Calculate the Nusselt number for free convection on a vertical or
    horizontal plate.

    For a vertical plate of height $L$, the average Nusselt number 
    $\overline{Nu}=\bar{h}L/k$ is estimated by the following expression:

    $$  \overline{Nu} = \left(0.825 + \frac{0.387 Ra^{1/6}}
        {[1 + (0.492/Pr)^{9/16}]^{8/27}}\right)^2 $$

    where $Ra$ is the Rayleigh number based on the plate height and $Pr$ is the
    Prandtl number.

    If the plate is horizontal, the flow and heat transfer patterns depend on
    whether the surface is heated or cooled, and which direction it is facing.
    For the upper surface of a heated plate (or the lower surface of a cooled
    plate), the Nusselt number is estimated by the following expression:

    $$ \overline{Nu} =
    \begin{cases}
    0.54 Ra^{1/4} ,& 10^4 \lesssim Ra \lesssim 10^7 \\
    0.15 Ra^{1/3} ,& 10^7 \lesssim Ra \lesssim 10^{11}
    \end{cases} $$

    For the lower surface of a heated plate (or the upper surface of a cooled
    plate), the Nusselt number is estimated by the following expression:

    $$ \overline{Nu} = 0.27 Ra^{1/4} \;, 10^5 \lesssim Ra \lesssim 10^{10} $$

    where $Ra$ is the Rayleigh number based on the ratio between the plate
    surface area and perimeter.

    In all cases, the properties are to be evaluated at the film temperature.

    !!! tip

        * The correlation for vertical plates can also be applied to vertical 
        cylinders of height $L$ and diameter $D$ if $D/L \gtrsim 35/Gr_L^{1/4}$.
        * The correlation for vertical plates can also be applied to the top and
        bottom surfaces of heated and cooled _inclined_ plates, respectively, 
        if $g$ is replaced by $g \cos \theta$ in the calculaton of $Ra$. 

    **References**

    * Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
      mass transfer", 4th edition, 1996, p. 493-498.

    Parameters
    ----------
    orientation: Literal['vertical', 'horizontal-upper-heated', 'horizontal-lower-heated']
        Orientation of the plate.
    Ra : float
        Rayleigh number based on plate characteristic length. For a vertical
        plate, the characteristic length is the plate height. For a horizontal
        plate, the characteristic length is the ratio between the plate surface
        area and perimeter.
    Pr : float
        Prandtl number.

    Returns
    -------
    float
        Nusselt number.

    See also
    --------
    - [`Nu_plate`](Nu_plate.md): related method for forced convection.

    Examples
    --------
    Estimate the heat transfer coefficient between the outer surface of a
    vertical tank (D=2 m, H=3 m) with a surface temperature of 310 K and
    quiescent air at 290 K.
    >>> from polykin.transport import Nu_plate_free
    >>> L = 3.0           # m (characteristic length is tank height)
    >>> Ts, Tb = 310, 290 # K
    >>> Tf = (Ts + Tb)/2  # K (film temperature)
    >>> Pr = 0.707        # m (properties at Tf=300 K)
    >>> nu = 15.9e-6      # m²/s
    >>> k = 26.3e-3       # W/m/K
    >>> beta = 1/Tf       # 1/K
    >>> g = 9.81          # m/s²
    >>> Gr = g*beta*(Ts - Tb)*L**3/nu**2
    >>> Ra = Gr*Pr
    >>> Nu = Nu_plate_free('vertical', Ra, Pr)
    >>> h = Nu*k/L
    >>> print(f"h={h:.1e} W/m²·K")
    h=3.7e+00 W/m²·K
    """
    if orientation == 'vertical':
        if Pr:
            return (0.825 + 0.387*Ra**(1/6) / (1 + (0.492/Pr)**(9/16))**(8/27))**2
        else:
            raise ValueError("`Pr` is required for vertical plates.")
    elif orientation == 'horizontal-upper-heated':
        check_range_warn(Ra, 1e4, 1e11, 'Ra')
        if Ra < 1e7:
            return 0.54*Ra**(1/4)
        else:
            return 0.15*Ra**(1/3)
    elif orientation == 'horizontal-lower-heated':
        check_range_warn(Ra, 1e5, 1e10, 'Ra')
        return 0.27*Ra**(1/4)
    else:
        raise ValueError(f"Unknown `orientation`: {orientation}.")


def U_plane_wall(h1: float,
                 h2: float,
                 L: float,
                 k: float,
                 Rf1: float = 0.0,
                 Rf2: float = 0.0) -> float:
    r"""Calculate the overall heat transfer coefficient through a plane wall
    with convection on both sides.

    Under steady-state conditions, the overall heat transfer coefficient is
    given by the following expression:

    $$ U = \left( \frac{1}{h_1} + \frac{1}{h_2} + \frac{L}{k} + R_{f,1} + R_{f,2} \right)^{-1} $$

    where $h_i$ and $R_{f,i}$ denote, respectively, the heat transfer coefficient
    and fouling factor at surface $i$, $L$ is the wall thickness, and $k$ is the
    wall thermal conductivity.

    Parameters
    ----------
    h1 : float
        Heat transfer coefficient at surface 1 (W/(m²·K)).
    h2 : float
        Heat transfer coefficient at surface 2 (W/(m²·K)).
    L : float
        Wall thickness (m).
    k : float
        Wall thermal conductivity (W/(m·K)).
    Rf1 : float
        Fouling factor at surface 1 ((m²·K)/W).
    Rf2 : float
        Fouling factor at surface 2 ((m²·K)/W).

    Returns
    -------
    float
        Overall heat transfer coefficient (W/(m²·K)).

    See also
    --------
    - [`U_cylindrical_wall`](U_cylindrical_wall.md): related method for a
      cylindrical wall.

    Examples
    --------
    Calculate the overall heat transfer coefficient for a 10 mm-thick plane
    carbon steel wall subjected to convection on both sides, with heat transfer
    coefficients of 1000 and 2000 W/(m²·K). Neglect fouling effects.
    >>> from polykin.transport import U_plane_wall
    >>> h1 = 1e3  # W/(m²·K)
    >>> h2 = 2e3  # W/(m²·K)
    >>> k = 6e2   # W/(m·K)
    >>> L = 10e-3 # m
    >>> U = U_plane_wall(h1, h2, L, k)
    >>> print(f"U={U:.1e} W/(m²·K)")
    U=6.6e+02 W/(m²·K)
    """
    return 1/(1/h1 + 1/h2 + L/k + Rf1 + Rf2)


def U_cylindrical_wall(hi: float,
                       ho: float,
                       di: float,
                       do: float,
                       k: float,
                       Rfi: float = 0.0,
                       Rfo: float = 0.0) -> float:
    r"""Calculate the overall heat transfer coefficient through a cylindrical
    wall with convection on both sides.

    Under steady-state conditions, the overall heat transfer coefficient is
    given by the following expression:

    $$ U_o = \left( \frac{d_o}{h_i d_i} + \frac{1}{h_o} + \frac{R_{f,i}d_o}{d_i}
             + R_{f,o} + \frac{d_o}{2k}\ln(d_o/d_i) \right)^{-1} $$

    where $h$ is the heat transfer coefficient, $R_{f}$ is the fouling factor,
    $d$ is the diameter, $k$ is the wall thermal conductivity, and the subscripts
    $i$ and $o$ indicate inner and outer surfaces, respectively.

    !!! tip

        The overall heat transfer coefficient based on the inner surface, $U_i$,
        can be computed using the indentity $U_i d_i = U_o d_o$.

    Parameters
    ----------
    hi : float
        Heat transfer coefficient at inner surface (W/(m²·K)).
    ho : float
        Heat transfer coefficient at outer surface (W/(m²·K)).
    di : float
        Inner diameter (m).
    do : float
        Outer diameter (m).
    k : float
        Wall thermal conductivity (W/(m·K)).
    Rfi : float
        Fouling factor at inner surface ((m²·K)/W).
    Rfo : float
        Fouling factor at outer surface ((m²·K)/W).

    Returns
    -------
    float
        Overall heat transfer coefficient based on outer surface (W/(m²·K)).

    See also
    --------
    - [`U_plane_wall`](U_plane_wall.md): related method for a plane wall.

    Examples
    --------
    Calculate the overall heat transfer coefficient for a carbon steel tube 
    subjected to convection on both sides, with heat transfer coefficients 
    of 2000 and 1000 W/(m²·K) for the inner and outer surfaces, respectively. 
    The tube has inner and outer diameters of 40 mm and 50 mm. Neglect fouling
    effects.
    >>> from polykin.transport import U_cylindrical_wall
    >>> hi = 2e3   # W/(m²·K)
    >>> ho = 1e3   # W/(m²·K)
    >>> k = 6e2    # W/(m·K)
    >>> di = 40e-3 # m
    >>> do = 50e-3 # m
    >>> Uo = U_cylindrical_wall(hi, ho, di, do, k)
    >>> print(f"Uo={Uo:.1e} W/(m²·K)")
    Uo=6.1e+02 W/(m²·K)
    """
    return 1/(do/(hi*di) + 1/ho + Rfi*do/di + Rfo + do/(2*k)*np.log(do/di))


def Nu_combined(Nu_forced: float,
                Nu_free: float,
                assisted: bool) -> float:
    r"""Calculate the combined Nusselt number for forced and free convection.

    The combined Nusselt number is given by the following expression:

    $$ Nu = \sqrt[3]{Nu_{forced}^3 \pm Nu_{free}^3} $$

    where the sign depends on the relative motion of the forced and free flows.
    The plus sign is for assisted or transverse flow and the minus sign is for
    opposing flows.

    !!! tip

        Combined free and forced convection is important when
        $Gr/Re^2 \approx 1$.

    **References**

    * Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
      mass transfer", 4th edition, 1996, p. 515.

    Parameters
    ----------
    Nu_forced : float
        Nusselt number for forced convection.
    Nu_free : float
        Nusselt number for free convection.
    assisted : bool
        Flag to indicate the relative motion of the forced and free flows. Set
        `True` for assisted or transverse flow and `False` for opposing flows.  

    Returns
    -------
    float
        Combined Nusselt number.

    Examples
    --------
    Calculate the combined Nusselt number for a transverse flow where the free
    and forced Nusselt numbers are 10.0 and 20.0, respectively.
    >>> from polykin.transport import Nu_combined
    >>> Nu = Nu_combined(Nu_forced=20.0, Nu_free=10.0, assisted=True)
    >>> print(f"Nu={Nu:.1f}")
    Nu=20.8
    """
    if assisted:
        return cbrt(Nu_forced**3 + Nu_free**3)
    else:
        return cbrt(Nu_forced**3 - Nu_free**3)
