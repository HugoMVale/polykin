# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import sqrt

from polykin.transport.flow import fD_Haaland

__all__ = ['Nu_tube_internal',
           'Nu_cylinder',
           'Nu_sphere',
           'Nu_drop',
           'Nu_flatplate'
           ]


def Nu_tube_internal(Re: float, Pr: float, er: float = 0.) -> float:
    r"""Calculate the internal Nusselt number for turbulent flow through a
    circular tube.

    For turbulent flow in a circular tube, the Nusselt number $Nu=hD/k$ can be
    estimated by the following expression:

    $$  Nu = \frac{(f_D/8)(Re - 1000)Pr}{1 + 12.7 (f_D/8)^{1/2} (Pr^{2/3} - 1)} $$

    \begin{bmatrix}
    3000 < Re < 5 \times 10^{6} \\
    0.5 < Pr < 2000 \\
    L/D \gtrsim 10
    \end{bmatrix}

    where $f_D$ is the Darcy friction factor, $Re$ is the Reynolds number, and
    $Pr$ is the Prandtl number. The properties are to be evaluated at the mean
    fluid temperature.

    **References**

    *   Gnielinski, Volker. "New equations for heat and mass transfer in
        turbulent pipe and channel flow." International chemical engineering
        16.2 (1976): 359-367.
    *   Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
        mass transfer.", 4th edition, 1996, p. 445.

    Parameters
    ----------
    Re : float
        Reynolds number.
    Pr : float
        Prandtl number.
    er : float
        Relative pipe roughness.

    Returns
    -------
    float
        Nusselt number.
    """
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

    where $Re$ is the Reynolds number, and $Pr$ is the Prandtl number. The 
    properties are to be evaluated at the film temperature.

    **References**

    *   Churchill, S. W., and Bernstein, M. (May 1, 1977). "A Correlating
        Equation for Forced Convection From Gases and Liquids to a Circular
        Cylinder in Crossflow." ASME. J. Heat Transfer. May 1977; 99(2): 300.
    *   Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
        mass transfer.", 4th edition, 1996, p. 370.

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
    properties are to be evaluated at $T_{\infty}$, except $\mu_s$.

    **References**

    *   Whitaker, S. (1972), Forced convection heat transfer correlations for
        flow in pipes, past flat plates, single cylinders, single spheres, and
        for flow in packed beds and tube bundles. AIChE J., 18: 361-371.
    *   Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
        mass transfer.", 4th edition, 1996, p. 374.

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

    where $Re$ is the drop Reynolds number, and $Pr$ is the Prandtl number. The
    exact range for $Pr$ in this context is unspecified. All properties are to
    be evaluated at $T_{\infty}$. This correlation was developed for use in
    spray drying applications.

    **References**

    *   Ranz, W. and Marshall, W. (1952) Evaporation from Drops. Chemical
        Engineering Progress, 48, 141-146.
    *   Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
        mass transfer.", 4th edition, 1996, p. 374.

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

    where $Re$ is the Reynolds number, and $Pr$ is the Prandtl number.

    **References**

    *   Incropera, Frank P., and David P. De Witt. "Fundamentals of heat and
        mass transfer.", 4th edition, 1996, p. 354-356.

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
    Rec = 5e5
    if Re < Rec:
        return 0.664*Re**(1/2)*Pr**(1/3)
    else:
        A = 0.037*Rec**(4/5) - 0.664*Rec**(1/2)
        return (0.037*Re**(4/5) - A)*Pr**(1/3)
