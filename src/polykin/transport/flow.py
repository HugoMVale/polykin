# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import log10, pi, sqrt
from scipy.constants import g

from polykin.math import root_newton

__all__ = ['fD_Colebrook',
           'fD_Haaland',
           'DP_Hagen_Poiseuille',
           'DP_Darcy_Weisbach',
           'DP_tube',
           'DP_packed_bed',
           'Cd_sphere',
           'terminal_velocity_sphere',
           'terminal_velocity_Stokes',
           ]


def DP_Hagen_Poiseuille(Q: float,
                        D: float,
                        L: float,
                        mu: float
                        ) -> float:
    r"""Calculate the pressure drop in a pipe using the Hagen-Poiseuille equation.

    In laminar flow, the pressure drop in a circular pipe is given by:

    $$ \Delta P =  \frac{128 \mu Q}{\pi D^4} L $$

    where $D$ is the pipe diameter, $L$ is the pipe length, $Q$ is the volume
    flowrate, and $\mu$ is the fluid viscosity.

    Parameters
    ----------
    Q : float
        Volume flowrate (m³/s).
    D : float
        Diameter (m).
    L : float
        Length (m).
    mu : float
        Viscosity (Pa·s).

    Returns
    -------
    float
        Pressure drop (Pa).

    Examples
    --------
    Calculate the pressure drop for a polymer solution with a viscosity of
    10 Pa.s flowing through 5 m of DN50 tube at 1 L/s. 
    >>> from polykin.transport import DP_Hagen_Poiseuille
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
    return 128*mu*Q*L/(pi*D**4)


def DP_Darcy_Weisbach(v: float,
                      D: float,
                      L: float,
                      rho: float,
                      fD: float,
                      ) -> float:
    r"""Calculate the pressure drop in a pipe using the Darcy-Weisbach
    equation.

    For a fluid flowing through a circular pipe, the pressure drop is given by:

    $$ \Delta P = f_D \frac{\rho}{2} \frac{v^2}{D} L $$

    where $f_D$ is the Darcy friction factor, $v$ is the velocity, $D$ is the
    pipe diameter, $L$ is the pipe length, and $\rho$ is the fluid density. 
    This equation is valid for both laminar and turbulent flow. In laminar
    flow, $f_D=64/Re$. For turbulent flow, $f_D$ can be estimated
    using either Colebrook's or Haaland's equation.

    Parameters
    ----------
    v : float
        Velocity (m/s).
    D : float
        Diameter (m).
    L : float
        Length (m).
    rho : float
        Density (kg/m³).
    fD : float
        Darcy friction factor. Should not be confused with the Fanning friction
        factor.

    Returns
    -------
    float
        Pressure drop (Pa).

    See also
    --------
    * [`fD_Colebrook`](fD_Colebrook.md): associated method to estimate the
      friction factor.
    * [`fD_Haaland`](fD_Haaland.md): associated method to estimate the
      friction factor.
    * [`DP_Hagen_Poiseuille`](DP_Hagen_Poiseuille.md): specific method for
      laminar flow.

    Examples
    --------
    Calculate the pressure drop for water flowing through 500 m of DN25 PVC
    pipe at 2 m/s.
    >>> from polykin.transport import DP_Darcy_Weisbach, fD_Haaland
    >>> rho = 1e3 # kg/m³
    >>> mu = 1e-3 # Pa·s
    >>> L = 5e2   # m
    >>> D = 25e-3 # m
    >>> v = 2.    # m/s
    >>> Re = rho*v*D/mu # turbulent flow
    >>> er = 0.0015e-3/D
    >>> fD = fD_Haaland(Re, er)
    >>> DP = DP_Darcy_Weisbach(v, D, L, rho, fD)
    >>> print(f"DP = {DP:.1e} Pa")
    DP = 8.3e+05 Pa
    """
    return fD * rho / 2 * v**2 / D * L


def DP_tube(Q: float,
            D: float,
            L: float,
            rho: float,
            mu: float,
            er: float = 0.0
            ) -> float:
    r"""Calculate the pressure drop in a circular pipe for both laminar and
    turbulent flow.

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
        Volume flowrate (m³/s).
    D : float
        Diameter (m).
    L : float
        Length (m).
    rho : float
        Density (kg/m³).
    mu : float
        Viscosity (Pa·s).
    er : float
        Relative pipe roughness, $\epsilon/D$. Only needed for turbulent flow.

    Returns
    -------
    float
        Pressure drop (Pa).
    """

    v = 4*Q/(pi*D**2)
    Re = rho*v*D/mu
    if Re < 2.3e3:
        fD = 64/Re
    else:
        fD = fD_Haaland(Re, er)

    return DP_Darcy_Weisbach(v, D, L, rho, fD)


def DP_packed_bed(G: float,
                  L: float,
                  Dp: float,
                  eps: float,
                  rho: float,
                  mu: float
                  ) -> float:
    r"""Calculate the pressure drop in a packed bed.

    The pressure drop in a packed bed is given by:

    $$ \Delta P = \frac{G^2 (1 - \epsilon) f_p}{\rho D_p \epsilon^3} L $$

    where $G$ is the mass flux, $D_p$ is the particle diameter, $\epsilon$ is
    the bed porosity, $L$ is the packed bed length, and $\rho$ is the fluid
    density. The packing friction factor $f_p$ is estimated using the Sato and
    Tallmadge correlation:

    $$ f_p = \frac{150}{Re_p} + \frac{4.2}{Re_p^{1/6}} $$

    where $Re_p=D_p G/(\mu (1-\epsilon))$ is the packing Reynolds
    number.

    **References**

    *   Walas, S. M. (1988). Chemical Process Equipment: Selection and Design.
        Singapore: Butterworths.

    Parameters
    ----------
    G : float
        Mass flux (kg/m²·s).
    L : float
        Packed bed length (m).
    Dp : float
        Particle diameter (m).
    eps : float
        Bed porosity.
    rho : float
        Fluid density (kg/m³).
    mu : float
        Fluid viscosity (Pa·s).

    Returns
    -------
    float
        Pressure drop (Pa).

    Examples
    --------
    Calculate the pressure drop in a packed bed reactor.
    >>> from polykin.transport import DP_packed_bed
    >>> DP = DP_packed_bed(G=50., L=2., Dp=1e-2, eps=0.45, rho=800., mu=0.01)
    >>> print(f"DP = {DP:.1e} Pa")
    DP = 1.4e+04 Pa
    """
    Rep = Dp*G/(mu*(1 - eps))
    fp = 150/Rep + 4.2/Rep**(1/6)
    return G**2*(1 - eps)*fp*L/(rho*Dp*eps**3)


def fD_Colebrook(Re: float, er: float) -> float:
    r"""Calculate the Darcy friction factor using Colebrook's equation.

    For turbulent flow, i.e., $Re \gtrsim 2300$, the friction factor
    is given by the following implicit expression:

    $$  \frac{1}{\sqrt{f}}= -2 \log \left( \frac {\epsilon/D} {3.7} +
        \frac {2.51} {Re \sqrt{f}} \right) $$

    This equation is a historical landmark but has the disadvantage of
    being implicit, requiring an iterative solution.

    **References**

    *   Colebrook, C F (1939). "Turbulent Flow in Pipes, with Particular Reference
        to the Transition Region Between the Smooth and Rough Pipe Laws". Journal
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

    See also
    --------
    * [`fD_Haaland`](fD_Haaland.md): alternative method.

    Examples
    --------
    Calculate the friction factor for water flowing through a DN25 PVC pipe at
    2 m/s.
    >>> from polykin.transport import fD_Colebrook
    >>> Re = 1e3*2*25e-3/1e-3
    >>> er = 0.0015/25
    >>> fD = fD_Colebrook(Re, er)
    >>> print(f"fD = {fD:.3f}")
    fD = 0.021
    """

    def fnc(f):
        return 2*log10(er/3.7 + 2.51/(Re*sqrt(f))) + 1/sqrt(f)

    sol = root_newton(fnc, fD_Haaland(Re, er), 1e-5)

    return sol.x


def fD_Haaland(Re: float, er: float) -> float:
    r"""Calculate the Darcy friction factor using Haaland's equation.

    For turbulent flow, i.e., $Re \gtrsim 2300$, the friction factor
    is given by the following implicit expression:

    $$ \frac{1}{\sqrt{f}}= -1.8 \log \left[\left(\frac{\epsilon/D}{3.7}\right)^{1.11} 
       + \frac{6.9}{Re} \right] $$

    This equation is as accurate as Colebrook's but has the advantage of
    being explicit.

    **References**

    *   Haaland, S. E. (March 1, 1983). "Simple and Explicit Formulas for the
        Friction Factor in Turbulent Pipe Flow." ASME. J. Fluids Eng. March 1983;
        105(1): 89-90.

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

    See also
    --------
    * [`fD_Colebrook`](fD_Colebrook.md): alternative method.

    Examples
    --------
    Calculate the friction factor for water flowing through a DN25 PVC pipe at
    2 m/s.
    >>> from polykin.transport import fD_Haaland
    >>> Re = 1e3*2*25e-3/1e-3
    >>> er = 0.0015/25
    >>> fD = fD_Haaland(Re, er)
    >>> print(f"fD = {fD:.3f}")
    fD = 0.021
    """
    return (1/(1.8*log10((er/3.7)**1.11 + 6.9/Re)))**2


def Cd_sphere(Re: float) -> float:
    r"""Calculate the drag coefficient of an isolated sphere.

    For laminar as well as for turbulent flow, the drag coefficient is given
    by the following expression:

    $$ C_{d} = \frac{24}{Re} \left(1 + 0.173 Re^{0.657}\right) 
             + \frac{0.413}{1 + 16300 Re^{-1.09}} $$

    where $Re$ is the particle Reynolds number.

    **References**

    *   Turton, R., and O. Levenspiel. "A short note on the drag correlation
        for spheres." Powder technology 47.1 (1986): 83-86.

    Parameters
    ----------
    Re : float
        Particle Reynolds number.

    Returns
    -------
    float
        Drag coefficient for an isolated sphere. 

    See also
    --------
    * [`terminal_velocity_sphere`](terminal_velocity_sphere.md): related method
        to estimate the terminal velocity.

    Examples
    --------
    Calculate the drag coefficient for a tennis ball traveling at 30 m/s.
    >>> from polykin.transport import Cd_sphere
    >>> D = 6.7e-2  # m
    >>> mu = 1.6e-5 # Pa·s
    >>> rho = 1.2   # kg/m³
    >>> v = 30.     # m/s
    >>> Re = rho*v*D/mu
    >>> Cd = Cd_sphere(Re)
    >>> print(f"Cd = {Cd:.3f}")
    Cd = 0.468
    """
    return 24/Re*(1 + 0.173*Re**0.657) + 0.413/(1 + 16300*Re**(-1.09))


def terminal_velocity_Stokes(D: float,
                             rhop: float,
                             rho: float,
                             mu: float
                             ) -> float:
    r"""Calculate the terminal velocity of an isolated sphere using Stokes' law.

    In laminar flow ($Re \lesssim 0.1$), the terminal velocity of an
    isolated particle is given by:

    $$ v_t = \frac{D^2 g (\rho_p - \rho)}{18 \mu} $$

    Parameters
    ----------
    D : float
        Particle diameter (m).
    rhop : float
        Particle density (kg/m³).
    rho : float
        Fluid density (kg/m³).
    mu : float
        Fluid viscosity (Pa·s).

    Returns
    -------
    float
        Terminal velocity (m/s).

    See also
    --------
    *   [`terminal_velocity_sphere`](terminal_velocity_sphere.md): generic
        method for laminar and turbulent flow.

    Examples
    --------
    Calculate the terminal velocity of a 500 nm PVC particle in water.
    >>> from polykin.transport import terminal_velocity_Stokes
    >>> vt = terminal_velocity_Stokes(500e-9, 1.4e3, 1e3, 1e-3)
    >>> print(f"vt = {vt:.1e} m/s")
    vt = 5.4e-08 m/s
    """
    return D**2*g*(rhop - rho)/(18*mu)


def terminal_velocity_sphere(D: float,
                             rhop: float,
                             rho: float,
                             mu: float
                             ) -> float:
    r"""Calculate the terminal velocity of an isolated sphere in laminar or
    turbulent flow.

    In both laminar and turbulent flow, the terminal velocity of an isolated
    sphere can be estimated by:

    $$ v_t = \sqrt{\frac{4 D g (\rho_p - \rho)}{3 C_d \rho}} $$

    where $C_d$ is the drag coefficient. This implementation uses an empirical
    drag correlation proposed by Turton and Levenspiel.

    !!! tip

        In laminar flow, $v_t \propto D^2$, while in turbulent flow, 
        $v_t \propto D^{1/2}$.

    **References**

    *   Turton, R., and O. Levenspiel. "A short note on the drag correlation
        for spheres." Powder technology 47.1 (1986): 83-86.

    Parameters
    ----------
    D : float
        Particle diameter (m).
    rhop : float
        Particle density (kg/m³).
    rho : float
        Fluid density (kg/m³).
    mu : float
        Fluid viscosity (Pa·s).

    Returns
    -------
    float
        Terminal velocity (m/s).

    See also
    --------
    *   [`terminal_velocity_Stokes`](terminal_velocity_Stokes.md): specific
        method for laminar flow.
    *   [`Cd_sphere`](Cd_sphere.md): related method to estimate the drag
        coefficient.

    Examples
    --------
    Calculate the terminal velocity of a 1 mm styrene droplet in air.
    >>> from polykin.transport import terminal_velocity_sphere
    >>> vt = terminal_velocity_sphere(1e-3, 910., 1.2, 1.6e-5)
    >>> print(f"vt = {vt:.1f} m/s")
    vt = 3.8 m/s
    """

    def fnc(vt):
        Re = rho*vt*D/mu
        Cd = Cd_sphere(Re)
        return vt - sqrt(4*D*g*(rhop - rho)/(3*Cd*rho))

    sol = root_newton(fnc, terminal_velocity_Stokes(D, rhop, rho, mu))

    return sol.x
