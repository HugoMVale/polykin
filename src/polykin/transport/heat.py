# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from typing import Literal

from numpy import sqrt

from polykin.transport.flow import fD_Haaland

__all__ = ['Nu_tube',
           'Nu_cylinder',
           'Nu_sphere',
           'Nu_drop',
           'Nu_flatplate',
           'Nu_tank'
           ]


def Nu_tube(Re: float,
            Pr: float,
            D_L: float = 0.,
            er: float = 0.
            ) -> float:
    r"""Calculate the internal Nusselt number for flow through a circular tube.

    For laminar flow, the average Nusselt number $\overline{Nu}=\bar{h}D/k$ can
    be estimated by the following expression:

    $$ \overline{Nu} = 3.66 + \frac{0.0668 (D/L) Re Pr}{1 + 0.04 [(D/L) Re Pr]^{2/3}} $$

    where $Re$ is the Reynolds number and $Pr$ is the Prandtl number. This
    correlation presumes constant surface temperature and a thermal entry length,
    thereby leading to conservative (underestimated) $\overline{Nu}$ values.

    For turbulent flow, the Nusselt number can be estimated by the following
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

    *   Gnielinski, Volker. "New equations for heat and mass transfer in
        turbulent pipe and channel flow." International chemical engineering
        16.2 (1976): 359-367.
    *   Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
        mass transfer", 4th edition, 1996, p. 444, 445.

    Parameters
    ----------
    Re : float
        Reynolds number.
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
    """
    if Re < 2.3e3:
        return 3.66 + (0.0668*D_L*Re*Pr)/(1 + 0.04*(D_L*Re*Pr)**(2/3))
    else:
        fD = fD_Haaland(Re, er)
        return (fD/8)*(Re - 1e3)*Pr/(1 + 12.7*sqrt(fD/8)*(Pr**(2/3) - 1))


def Nu_cylinder(Re: float, Pr: float) -> float:
    r"""Calculate the Nusselt number for cross flow over a circular cylinder.

    For flow normal to the axis of a circular cylinder, the average Nusselt 
    number $\overline{Nu}=\bar{h}D/k$ can be estimated by the following expression:

    $$ \overline{Nu} = 0.3 + \frac{0.62 Re^{1/2} Pr^{1/3}}
    {\left[1 + (0.4/Pr)^{2/3}\right]^{1/4}} 
    \left[1 + \left(\frac{Re}{282 \times 10^3}\right)^{5/8}\right]^{4/5} $$

    $$ \left[  Re Pr > 0.2 \right] $$

    where $Re$ is the Reynolds number and $Pr$ is the Prandtl number. The 
    properties are to be evaluated at the film temperature.

    **References**

    *   Churchill, S. W., and Bernstein, M. (May 1, 1977). "A Correlating
        Equation for Forced Convection From Gases and Liquids to a Circular
        Cylinder in Crossflow." ASME. J. Heat Transfer. May 1977; 99(2): 300.
    *   Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
        mass transfer", 4th edition, 1996, p. 370.

    Parameters
    ----------
    Re : float
        Reynolds number.
    Pr : float
        Prandtl number.

    Returns
    -------
    float
        Nusselt number.
    """
    return 0.3 + 0.62*Re**(1/2)*Pr**(1/3)/(1 + (0.4/Pr)**(2/3))**(1/4) \
        * (1 + (Re/282e3)**(5/8))**(4/5)


def Nu_sphere(Re: float, Pr: float, mur: float) -> float:
    r"""Calculate the Nusselt number for an isolated sphere.

    For flow over a sphere, the average Nusselt number $\overline{Nu}=\bar{h}D/k$
    can be estimated by the following expression:

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

    *   Whitaker, S. (1972), "Forced convection heat transfer correlations for
        flow in pipes, past flat plates, single cylinders, single spheres, and
        for flow in packed beds and tube bundles", AIChE J., 18: 361-371.
    *   Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
        mass transfer", 4th edition, 1996, p. 374.

    Parameters
    ----------
    Re : float
        Sphere Reynolds number.
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
    return 2 + (0.4*Re**(1/2) + 0.06*Re**(2/3))*Pr**0.4*(mur)**(1/4)


def Nu_drop(Re: float, Pr: float) -> float:
    r"""Calculate the Nusselt number for a single drop falling in a fluid.

    For a freely falling drop, the average Nusselt number $\overline{Nu}=\bar{h}D/k$
    can be estimated by the following expression:

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

    *   Ranz, W. and Marshall, W. (1952) "Evaporation from Drops", Chemical
        Engineering Progress, 48, 141-146.
    *   Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
        mass transfer", 4th edition, 1996, p. 374.

    Parameters
    ----------
    Re : float
        Drop Reynolds number.
    Pr : float
        Prandtl number.

    Returns
    -------
    float
        Nusselt number.

    See also
    --------
    * [`Nu_sphere`](Nu_sphere.md): generic method for spheres.
    """
    return 2 + 0.6*Re**(1/2)*Pr**(1/3)


def Nu_flatplate(Re: float, Pr: float) -> float:
    r"""Calculate the Nusselt number for parallel flow over a flat plate.

    For parallel flow over an isothermal flat plate, the average Nusselt number
    $\overline{Nu}=\bar{h}L/k$ can be estimated by the following expressions:    

    $$ \overline{Nu} =
    \begin{cases}
    0.664 Re^{1/2} Pr^{1/3} ,& Re > 5 \times 10^5 \\
    (0.037 Re^{4/5} - 871)Pr^{1/3} ,& 5 \times 10^5 < Re \lesssim 10^8
    \end{cases} $$

    $$ [0.6 < Pr < 60] $$

    where $Re$ is the Reynolds number and $Pr$ is the Prandtl number.

    **References**

    *   Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
        mass transfer", 4th edition, 1996, p. 354-356.

    Parameters
    ----------
    Re : float
        Plate Reynolds number.
    Pr : float
        Prandtl number.

    Returns
    -------
    float
        Nusselt number.
    """
    Re_c = 5e5
    if Re < Re_c:
        return 0.664*Re**(1/2)*Pr**(1/3)
    else:
        A = 0.037*Re_c**(4/5) - 0.664*Re_c**(1/2)
        return (0.037*Re**(4/5) - A)*Pr**(1/3)


def Nu_tank(
    impeller: Literal['4BF', '4BP', '6BD', 'HE3', 'PROP', 'anchor', 'helical-ribbon'],
    surface: Literal['wall', 'bottom-head', 'helical-coil', 'harp-coil-0', 'harp-coil-45'],
    Re: float,
    Pr: float,
    mur: float,
    D_T: float = 1/3,
    H_T: float = 1.,
    L_Ls: float = 1.,
    d_T: float = 0.04,
    P_D: float = 1.,
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
    impeller : Literal['4BF', '4BP', '6BD', 'HE3', 'PROP', 'anchor', 'helical-ribbon']
        Impeller type.
    surface : Literal['wall', 'bottom-head', 'helical-coil', 'harp-coil-0', 'harp-coil-45']
        Heat transfer surface type.
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
    and default dimensions.
    >>> from polykin.transport.heat import Nu_tank
    >>> T = 2.     # m
    >>> D = T/3    # m
    >>> rho = 1e3  # kg/m³
    >>> mu = 1e-3  # Pa.s
    >>> k  = 0.6   # W/m.K
    >>> Cp = 4.2e3 # J/kg.K
    >>> Re = (120/60) * D**2 * rho / mu
    >>> Pr = mu*Cp/k
    >>> Nu = Nu_tank('HE3', 'wall', Re, Pr, mur=1.)
    >>> h = Nu*k/T
    >>> print(f"h={h:.1e} W/m².K")
    h=1.6e+03 W/m².K
    """

    # Default parameters
    K = 0.
    a = 2/3
    b = 1/3
    c = 0.14
    Gc = 1.

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
            raise ValueError("Invalid `impeller`.")
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
            raise ValueError("Invalid `impeller`.")
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
            raise ValueError("Invalid `impeller`.")
    elif surface == 'harp-coil-0':
        if impeller == '4BF':
            K = 0.06  # the text mentions this value might be overestimated
            a = 0.65
            b = 0.3
            Gc = H_T**-0.15 * L_Ls**0.2 * (D_T/(1/3))**0.33 * (2/nb)**0.2
        else:
            raise ValueError("Invalid `impeller`.")
    elif surface == 'harp-coil-45':
        if impeller == '6BD':
            # quite some doubts regarding this equation
            K = 0.021
            a = 0.67
            b = 0.4
            Gc = H_T**-0.15 * L_Ls**0.2 * (D_T/(1/3))**0.33 * (2/nb)**0.2
        else:
            raise ValueError("Invalid `impeller`.")
    else:
        raise ValueError("Invalid heat transfer `surface`.")

    return K * Re**a * Pr**b * mur**c * Gc
