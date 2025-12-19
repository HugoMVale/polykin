# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import inf, log10, pi, sqrt
from scipy.constants import g

from polykin.math import root_newton
from polykin.utils.exceptions import RootSolverError
from polykin.utils.tools import check_range_warn

__all__ = [
    "fD_Colebrook",
    "fD_Haaland",
    "DP_Hagen_Poiseuille",
    "DP_Darcy_Weisbach",
    "DP_tube",
    "DP_packed_bed",
    "Cd_sphere",
    "vt_sphere",
    "vt_Stokes",
    "DP_GL_Lockhart_Martinelli",
    "DP_GL_Mueller_Bonn",
]


def DP_Hagen_Poiseuille(
    Q: float,
    D: float,
    L: float,
    mu: float,
) -> float:
    r"""Calculate the pressure drop in a circular pipe using the
    Hagen-Poiseuille equation.

    For laminar flow through a circular pipe, the pressure drop due to friction
    is given by:

    $$ \Delta P =  \frac{128 \mu Q}{\pi D^4} L $$

    where $D$ is the pipe diameter, $L$ is the pipe length, $Q$ is the volume
    flowrate, and $\mu$ is the fluid viscosity.

    Parameters
    ----------
    Q : float
        Volume flowrate [m³/s].
    D : float
        Diameter [m].
    L : float
        Length [m].
    mu : float
        Viscosity [Pa·s].

    Returns
    -------
    float
        Pressure drop [Pa].

    Examples
    --------
    Calculate the pressure drop for a polymer solution (viscosity: 10 Pa·s)
    flowing at 1 L/s through 5 m of smooth pipe with a 50 mm internal diameter.
    >>> from polykin.flow import DP_Hagen_Poiseuille
    >>> from math import pi
    >>> Q = 1e-3  # m³/s
    >>> D = 50e-3 # m
    >>> L = 5.    # m
    >>> mu = 10.  # Pa·s
    >>> rho = 1e3 # kg/m³
    >>> v = 4*Q/(pi*D**2)
    >>> Re = rho*v*D/mu
    >>> print(f"Re = {Re:.1e}")
    Re = 2.5e+00
    >>> DP = DP_Hagen_Poiseuille(Q, D, L, mu)
    >>> print(f"DP = {DP:.1e} Pa")
    DP = 3.3e+05 Pa
    """
    return 128 * mu * Q * L / (pi * D**4)


def DP_Darcy_Weisbach(
    v: float,
    D: float,
    L: float,
    rho: float,
    fD: float,
) -> float:
    r"""Calculate the pressure drop in a circular pipe using the Darcy-Weisbach
    equation.

    For a fluid flowing through a circular pipe, the pressure drop due to
    friction is given by:

    $$ \Delta P = f_D \frac{\rho}{2} \frac{v^2}{D} L $$

    where $f_D$ is the Darcy friction factor, $v$ is the velocity, $D$ is the
    pipe diameter, $L$ is the pipe length, and $\rho$ is the fluid density.
    This equation is valid for both laminar and turbulent flow. In laminar
    flow, $f_D=64/Re$. For turbulent flow, $f_D$ can be estimated
    using either Colebrook's or Haaland's equation.

    Parameters
    ----------
    v : float
        Velocity [m/s].
    D : float
        Diameter [m].
    L : float
        Length [m].
    rho : float
        Density [kg/m³].
    fD : float
        Darcy friction factor. Should not be confused with the Fanning friction
        factor.

    Returns
    -------
    float
        Pressure drop [Pa].

    See Also
    --------
    * [`fD_Colebrook`](fD_Colebrook.md): associated method to estimate the
      friction factor.
    * [`fD_Haaland`](fD_Haaland.md): associated method to estimate the
      friction factor.
    * [`DP_Hagen_Poiseuille`](DP_Hagen_Poiseuille.md): specific method for
      laminar flow.

    Examples
    --------
    Calculate the pressure drop for water flowing at 2 m/s through 500 m of PVC
    pipe with an internal diameter of 25 mm.
    >>> from polykin.flow import DP_Darcy_Weisbach, fD_Haaland
    >>> rho = 1e3 # kg/m³
    >>> mu = 1e-3 # Pa·s
    >>> L = 5e2   # m
    >>> D = 25e-3 # m
    >>> v = 2.    # m/s
    >>> Re = rho*v*D/mu  # turbulent flow
    >>> print(f"Re = {Re:.1e}")
    Re = 5.0e+04
    >>> er = 0.0015e-3/D # from pipe table
    >>> fD = fD_Haaland(Re, er)
    >>> DP = DP_Darcy_Weisbach(v, D, L, rho, fD)
    >>> print(f"DP = {DP:.1e} Pa")
    DP = 8.3e+05 Pa
    """
    return fD * rho / 2 * v**2 / D * L


def DP_tube(
    Q: float,
    D: float,
    L: float,
    rho: float,
    mu: float,
    er: float = 0.0,
) -> float:
    r"""Calculate the pressure drop due to friction for flow through a circular
    pipe.

    This method acts as a convenience wrapper for
    [`DP_Darcy_Weisbach`](DP_Darcy_Weisbach.md). It determines the flow regime
    and estimates the Darcy friction factor using the appropriate equation. For
    laminar flow, it applies $f_D=64/Re$. For turbulent flow, it uses
    [`fD_Haaland`](fD_Haaland.md). Finally, the method calls
    [`DP_Darcy_Weisbach`](DP_Darcy_Weisbach.md) with the correct parameters.

    !!! tip

        In laminar flow, $\Delta P \propto Q/D^4$, while in turbulent flow,
        $\Delta P \propto Q^2/D^5$.

    Parameters
    ----------
    Q : float
        Volume flowrate [m³/s].
    D : float
        Diameter [m].
    L : float
        Length [m].
    rho : float
        Density [kg/m³].
    mu : float
        Viscosity [Pa·s].
    er : float
        Relative pipe roughness, $\epsilon/D$. Only required for turbulent flow.

    Returns
    -------
    float
        Pressure drop [Pa].

    Examples
    --------
    Calculate the pressure drop for water flowing at 3 m³/h through 500 m of
    PVC pipe with an internal diameter of 25 mm.
    >>> from polykin.flow import DP_tube
    >>> Q = 3.0/3600 # m³/s
    >>> rho = 1e3    # kg/m³
    >>> mu = 1e-3    # Pa·s
    >>> L = 5e2      # m
    >>> D = 25e-3    # m
    >>> er = 0.0015e-3/D # from pipe table
    >>> DP = DP_tube(Q, D, L, rho, mu, er)
    >>> print(f"DP = {DP:.1e} Pa")
    DP = 6.2e+05 Pa
    """
    v = 4 * Q / (pi * D**2)
    Re = rho * v * D / mu
    if Re < 2.3e3:
        fD = 64 / Re
    else:
        fD = fD_Haaland(Re, er)

    return DP_Darcy_Weisbach(v, D, L, rho, fD)


def DP_packed_bed(
    G: float,
    L: float,
    Dp: float,
    eps: float,
    rho: float,
    mu: float,
) -> float:
    r"""Calculate the pressure drop in a packed bed.

    In a packed bed, the pressure drop due to friction is given by:

    $$ \Delta P = \frac{G^2 (1 - \epsilon) f_p}{\rho D_p \epsilon^3} L $$

    where $G$ is the mass flux, $D_p$ is the particle diameter, $\epsilon$ is
    the bed porosity, $L$ is the packed bed length, and $\rho$ is the fluid
    density. The packing friction factor $f_p$ is estimated using the Sato and
    Tallmadge correlation:

    $$ f_p = \frac{150}{Re_p} + \frac{4.2}{Re_p^{1/6}} $$

    where $Re_p=D_p G/(\mu (1-\epsilon))$ is the packing Reynolds
    number.

    **References**

    * Walas, S. M., "Chemical Process Equipment: Selection and Design",
      Singapore: Butterworths, 1988.

    Parameters
    ----------
    G : float
        Mass flux [kg/m²·s].
    L : float
        Packed bed length [m].
    Dp : float
        Particle diameter [m].
    eps : float
        Bed porosity.
    rho : float
        Fluid density [kg/m³].
    mu : float
        Fluid viscosity [Pa·s].

    Returns
    -------
    float
        Pressure drop [Pa].

    Examples
    --------
    Calculate the pressure drop in a packed bed reactor under the conditions
    specified below.
    >>> from polykin.flow import DP_packed_bed
    >>> DP = DP_packed_bed(G=50., L=2., Dp=1e-2, eps=0.45, rho=800., mu=0.01)
    >>> print(f"DP = {DP:.1e} Pa")
    DP = 1.4e+04 Pa
    """
    Rep = Dp * G / (mu * (1 - eps))
    fp = 150 / Rep + 4.2 / Rep ** (1 / 6)
    return G**2 * (1 - eps) * fp * L / (rho * Dp * eps**3)


def fD_Colebrook(Re: float, er: float) -> float:
    r"""Calculate the Darcy friction factor using Colebrook's equation.

    For turbulent flow, i.e., $Re \gtrsim 2300$, the friction factor
    is given by the following implicit expression:

    $$  \frac{1}{\sqrt{f}}= -2 \log \left( \frac {\epsilon/D} {3.7} +
        \frac {2.51} {Re \sqrt{f}} \right) $$

    This equation is a historical landmark but has the disadvantage of
    being implicit, requiring an iterative solution.

    **References**

    * Colebrook, C F (1939). "Turbulent Flow in Pipes, with Particular Reference
      to the Transition Region Between the Smooth and Rough Pipe Laws", Journal
      of the Institution of Civil Engineers. 11 (4): 133-156.

    Parameters
    ----------
    Re : float
        Reynolds number.
    er : float
        Relative pipe roughness, $\epsilon/D$.

    Returns
    -------
    float
        Darcy friction factor.

    See Also
    --------
    * [`fD_Haaland`](fD_Haaland.md): alternative method.

    Examples
    --------
    Calculate the friction factor for water flowing at 2 m/s through a PVC pipe
    with an internal diameter of 25 mm.
    >>> from polykin.flow import fD_Colebrook
    >>> rho = 1e3 # kg/m³
    >>> mu = 1e-3 # Pa·s
    >>> D = 25e-3 # m
    >>> v = 2.    # m/s
    >>> Re = rho*v*D/mu
    >>> er = 0.0015/25 # from pipe table
    >>> fD = fD_Colebrook(Re, er)
    >>> print(f"fD = {fD:.3f}")
    fD = 0.021
    """
    check_range_warn(Re, 2.3e3, inf, "Re")

    def fnc(f):
        return 2 * log10(er / 3.7 + 2.51 / (Re * sqrt(f))) + 1 / sqrt(f)

    sol = root_newton(fnc, fD_Haaland(Re, er))

    if sol.success:
        fD = sol.x
    else:
        raise RootSolverError("fD_Colebrook: could not converge to a solution.")

    return fD


def fD_Haaland(Re: float, er: float) -> float:
    r"""Calculate the Darcy friction factor using Haaland's equation.

    For turbulent flow, i.e., $Re \gtrsim 2300$, the friction factor
    is given by the following explicit expression:

    $$ \frac{1}{\sqrt{f}}= -1.8 \log \left[\left(\frac{\epsilon/D}{3.7}\right)^{1.11}
       + \frac{6.9}{Re} \right] $$

    This equation is as accurate as Colebrook's but has the advantage of
    being explicit.

    **References**

    * Haaland, S. E. "Simple and Explicit Formulas for the Friction Factor in
      Turbulent Pipe Flow", ASME. J. Fluids Eng. March 1983; 105(1): 89-90.

    Parameters
    ----------
    Re : float
        Reynolds number.
    er : float
        Relative pipe roughness, $\epsilon/D$.

    Returns
    -------
    float
        Darcy friction factor.

    See Also
    --------
    * [`fD_Colebrook`](fD_Colebrook.md): alternative method.

    Examples
    --------
    Calculate the friction factor for water flowing at 2 m/s through a PVC pipe
    with an internal diameter of 25 mm.
    >>> from polykin.flow import fD_Haaland
    >>> rho = 1e3 # kg/m³
    >>> mu = 1e-3 # Pa·s
    >>> D = 25e-3 # m
    >>> v = 2.    # m/s
    >>> Re = rho*v*D/mu
    >>> er = 0.0015/25 # from pipe table
    >>> fD = fD_Haaland(Re, er)
    >>> print(f"fD = {fD:.3f}")
    fD = 0.021
    """
    check_range_warn(Re, 2.3e3, inf, "Re")

    return (1 / (1.8 * log10((er / 3.7) ** 1.11 + 6.9 / Re))) ** 2


def Cd_sphere(Rep: float) -> float:
    r"""Calculate the drag coefficient of an isolated sphere.

    For laminar as well as for turbulent flow, the drag coefficient is given
    by:

    $$ C_{d} = \frac{24}{Re_p} \left(1 + 0.173 Re_p^{0.657}\right)
             + \frac{0.413}{1 + 16300 Re_p^{-1.09}} $$

    where $Re_p = \rho v D_p / \mu$ is the particle Reynolds number.

    **References**

    * Turton, R., and O. Levenspiel. "A short note on the drag correlation for
      spheres", Powder technology 47.1 (1986): 83-86.

    Parameters
    ----------
    Rep : float
        Particle Reynolds number.

    Returns
    -------
    float
        Drag coefficient for an isolated sphere.

    See Also
    --------
    * [`vt_sphere`](vt_sphere.md): related method to estimate the terminal
      velocity.

    Examples
    --------
    Calculate the drag coefficient for a tennis ball traveling at 30 m/s.
    >>> from polykin.flow import Cd_sphere
    >>> Dp = 6.7e-2  # m
    >>> mu = 1.6e-5  # Pa·s
    >>> rho = 1.2    # kg/m³
    >>> v = 30.      # m/s
    >>> Rep = rho*v*Dp/mu
    >>> Cd = Cd_sphere(Rep)
    >>> print(f"Cd = {Cd:.2f}")
    Cd = 0.47
    """
    return 24 / Rep * (1 + 0.173 * Rep**0.657) + 0.413 / (1 + 16300 * Rep ** (-1.09))


def vt_Stokes(
    Dp: float,
    rhop: float,
    rho: float,
    mu: float,
) -> float:
    r"""Calculate the terminal velocity of an isolated sphere using Stokes' law.

    In laminar flow ($Re_p \lesssim 0.1$), the terminal velocity of an
    isolated particle is given by:

    $$ v_t = \frac{D_p^2 g (\rho_p - \rho)}{18 \mu} $$

    where $D_p$ is the particle diameter, $g$ is the acceleration due to gravity,
    $\rho_p$ is the particle density, $\rho$ is the fluid density, and $\mu$ is
    the fluid viscosity.

    Parameters
    ----------
    Dp : float
        Particle diameter [m].
    rhop : float
        Particle density [kg/m³].
    rho : float
        Fluid density [kg/m³].
    mu : float
        Fluid viscosity [Pa·s].

    Returns
    -------
    float
        Terminal velocity [m/s].

    See Also
    --------
    * [`vt_sphere`](vt_sphere.md): generic method for laminar and turbulent flow.

    Examples
    --------
    Calculate the terminal velocity of a 500 nm PVC particle in water.
    >>> from polykin.flow import vt_Stokes
    >>> vt = vt_Stokes(500e-9, 1.4e3, 1e3, 1e-3)
    >>> print(f"vt = {vt:.1e} m/s")
    vt = 5.4e-08 m/s
    """
    vt = Dp**2 * g * (rhop - rho) / (18 * mu)

    Rep = rho * vt * Dp / mu
    check_range_warn(Rep, 0.0, 0.1, "Re")

    return vt


def vt_sphere(
    Dp: float,
    rhop: float,
    rho: float,
    mu: float,
) -> float:
    r"""Calculate the terminal velocity of an isolated sphere in laminar or
    turbulent flow.

    In both laminar and turbulent flow, the terminal velocity of an isolated
    sphere is given by:

    $$ v_t = \sqrt{\frac{4 D_p g (\rho_p - \rho)}{3 C_d \rho}} $$

    where $C_d(Re_p)$ is the drag coefficient. This implementation uses the drag
    correlation proposed by Turton and Levenspiel.

    !!! tip

        In laminar flow, $v_t \propto D_p^2$, while in turbulent flow,
        $v_t \propto D_p^{1/2}$.

    **References**

    * Turton, R., and O. Levenspiel. "A short note on the drag correlation for
      spheres", Powder technology 47.1 (1986): 83-86.

    Parameters
    ----------
    Dp : float
        Particle diameter [m].
    rhop : float
        Particle density [kg/m³].
    rho : float
        Fluid density [kg/m³].
    mu : float
        Fluid viscosity [Pa·s].

    Returns
    -------
    float
        Terminal velocity [m/s].

    See Also
    --------
    * [`vt_Stokes`](vt_Stokes.md): specific method for laminar flow.
    * [`Cd_sphere`](Cd_sphere.md): related method to estimate the drag
      coefficient.

    Examples
    --------
    Calculate the terminal velocity of a 1 mm styrene droplet in air.
    >>> from polykin.flow import vt_sphere
    >>> vt = vt_sphere(1e-3, 910., 1.2, 1.6e-5)
    >>> print(f"vt = {vt:.1f} m/s")
    vt = 3.8 m/s
    """

    def fnc(vt):
        Rep = rho * vt * Dp / mu
        Cd = Cd_sphere(Rep)
        return vt - sqrt(4 * Dp * g * (rhop - rho) / (3 * Cd * rho))

    sol = root_newton(fnc, x0=1.0)

    if sol.success:
        vt = sol.x
    else:
        raise RootSolverError("vt_sphere: could not converge to a solution.")

    return vt


def DP_GL_Lockhart_Martinelli(
    mdotL: float,
    mdotG: float,
    D: float,
    L: float,
    rhoL: float,
    rhoG: float,
    muL: float,
    muG: float,
    er: float,
) -> float:
    r"""Calculate the pressure drop due to friction in two-phase liquid-gas
    flow through a horizontal pipe using the Lockhart-Martinelli correlation.

    The pressure drop due to friction in a two-phase flow is estimated by:

    $$ (\Delta P)_{TP} = (\Delta P)_{L} \phi^2_{L} $$

    where $(\Delta P)_{L}$ is the pressure drop if the liquid phase were alone,
    and $\phi_{L}$ is the liquid-phase multiplier. The latter is given by:

    $$ \phi_{L} = 1 + \frac{C}{X} + \frac{1}{X^2} $$

    where $X=\sqrt{(\Delta P)_L/(\Delta P)_G}$ is the Lockhart-Martinelli
    parameter and $C$ is a coefficient that varies based on the flow regimes of
    the liquid and gas phases.

    **References**

    * Walas, S. M., "Chemical Process Equipment: Selection and Design",
      Singapore: Butterworths, 1988.

    Parameters
    ----------
    mdotL : float
        Mass flow rate of liquid [kg/s].
    mdotG : float
        Mass flow rate of gas [kg/s].
    D : float
        Diameter [m].
    L : float
        Length [m].
    rhoL : float
        Density of liquid [kg/m³].
    rhoG : float
        Density of gas [kg/m³].
    muL : float
        Viscosity of liquid [Pa·s].
    muG : float
        Viscosity of gas [Pa·s].
    er : float
        Relative pipe roughness, $\epsilon/D$. Only required for turbulent flow.

    Returns
    -------
    float
        Pressure drop [Pa].

    See Also
    --------
    * [`DP_GL_Mueller_Bonn`](DP_GL_Mueller_Bonn.md): alternative method.

    Examples
    --------
    Calculate the pressure gradient due to friction in a 80 mm inner diameter
    pipe with 2 kg/s of liquid and 1 kg/s of gas. The liquid and gas have
    densities of 1000 and 1 kg/m³, respectively, and viscosities of 1e-3 and
    2e-5 Pa·s, respectively.
    >>> from polykin.flow import DP_GL_Lockhart_Martinelli
    >>> mdotL = 2.0 # kg/s
    >>> mdotG = 1.0 # kg/s
    >>> D = 80e-3   # m
    >>> L = 1.0     # m
    >>> rhoL = 1e3  # kg/m³
    >>> rhoG = 1e0  # kg/m³
    >>> muL = 1e-3  # Pa·s
    >>> muG = 2e-5  # Pa·s
    >>> er = 0.0
    >>> DP = DP_GL_Lockhart_Martinelli(mdotL, mdotG, D, L, rhoL, rhoG, muL, muG, er)
    >>> print(f"DP = {DP:.1e} Pa/m")
    DP = 8.2e+03 Pa/m
    """
    # Pressure gradient if liquid were alone
    A = (pi / 4) * D**2
    if mdotL > 0.0:
        vL = mdotL / (A * rhoL)
        ReL = rhoL * vL * D / muL
        fL = 64 / ReL if ReL < 2.3e3 else fD_Haaland(ReL, er)
        dPL = DP_Darcy_Weisbach(vL, D, 1.0, rhoL, fL)
    else:
        dPL = 0.0

    # Pressure gradient if gas were alone
    if mdotG > 0.0:
        vG = mdotG / (A * rhoG)
        ReG = rhoG * vG * D / muG
        fG = 64 / ReG if ReG < 2.3e3 else fD_Haaland(ReG, er)
        dPG = DP_Darcy_Weisbach(vG, D, 1.0, rhoG, fG)
    else:
        dPG = 0.0

    # Two-phase pressure drop
    if not dPG:
        dP = dPL
    elif not dPL:
        dP = dPG
    else:
        X = sqrt(dPL / dPG)
        if ReL > 1e3:
            C = 20.0 if ReG > 1e3 else 10.0
        else:
            C = 12.0 if ReG > 1e3 else 5.0
        YL = 1 + C / X + 1 / X**2
        dP = YL * dPL
    DP = dP * L

    return DP


def DP_GL_Mueller_Bonn(
    mdotL: float,
    mdotG: float,
    D: float,
    L: float,
    rhoL: float,
    rhoG: float,
    muL: float,
    muG: float,
) -> float:
    r"""Calculate the pressure drop due to friction in two-phase liquid-gas
    flow through a horizontal pipe using the Mueller-Bonn correlation.

    According to the authors, this correlation performs better than the
    classical Lockhart-Martinelli correlation, but the average error is still
    40%.

    **References**

    * Müller-Steinhagen, H., Heck, K. "A simple friction pressure drop
      correlation for two-phase flow in pipes." Chemical Engineering and
      Processing: Process Intensification 20.6 (1986): 297-308.

    Parameters
    ----------
    mdotL : float
        Mass flow rate of liquid [kg/s].
    mdotG : float
        Mass flow rate of gas [kg/s].
    D : float
        Diameter [m].
    L : float
        Length [m].
    rhoL : float
        Density of liquid [kg/m³].
    rhoG : float
        Density of gas [kg/m³].
    muL : float
        Viscosity of liquid [Pa·s].
    muG : float
        Viscosity of gas [Pa·s].

    Returns
    -------
    float
        Pressure drop [Pa].

    See Also
    --------
    * [`DP_GL_Lockhart_Martinelli`](DP_GL_Lockhart_Martinelli.md): alternative
      method.

    Examples
    --------
    Calculate the pressure gradient due to friction in a 80 mm inner diameter
    pipe with 2 kg/s of liquid and 1 kg/s of gas. The liquid and gas have
    densities of 1000 and 1 kg/m³, respectively, and viscosities of 1e-3 and
    2e-5 Pa·s, respectively.
    >>> from polykin.flow import DP_GL_Mueller_Bonn
    >>> mdotL = 2.0 # kg/s
    >>> mdotG = 1.0 # kg/s
    >>> D = 80e-3   # m
    >>> L = 1.0     # m
    >>> rhoL = 1e3  # kg/m³
    >>> rhoG = 1e0  # kg/m³
    >>> muL = 1e-3  # Pa·s
    >>> muG = 2e-5  # Pa·s
    >>> DP = DP_GL_Mueller_Bonn(mdotL, mdotG, D, L, rhoL, rhoG, muL, muG)
    >>> print(f"DP = {DP:.1e} Pa/m")
    DP = 1.1e+04 Pa/m
    """
    # Flow quality and mass flux
    mdot = mdotL + mdotG
    x = mdotG / mdot
    vm = mdot / ((pi / 4) * D**2)

    # Pressure gradient if total flow had liquid properties
    ReL = vm * D / muL
    fL = 64 / ReL if ReL <= 1187.0 else 0.3164 / ReL**0.25
    dPL0 = fL * vm**2 / (2 * rhoL * D)

    # Pressure gradient if total flow had gas properties
    ReG = vm * D / muG
    fG = 64 / ReG if ReG <= 1187.0 else 0.3164 / ReG**0.25
    dPG0 = fG * vm**2 / (2 * rhoG * D)

    # Two-phase pressure gradient
    dP = (dPL0 + 2 * (dPG0 - dPL0) * x) * (1 - x) ** (1 / 3) + dPG0 * x**3
    DP = dP * L

    return DP
