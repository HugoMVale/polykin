# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import log10, sqrt

from polykin.math import root_newton

__all__ = ['fD_Colebrook',
           'fD_Haaland',
           'pressure_drop_pipe']


def pressure_drop_pipe(D: float,
                       v: float,
                       rho: float,
                       fD: float,
                       L: float = 1.0
                       ) -> float:
    r"""Pressure drop in a pipe according to the Darcy-Weisbach equation.

    $$ \Delta P = f_D \frac{\rho}{2} \frac{v^2}{D} L $$

    Parameters
    ----------
    D : float
        Diameter (m).
    v : float
        Velocity (m/s)
    rho : float
        Density (kg/m³).
    fD : float
        Darcy friction factor. Should not be confused with the Fanning friction
        factor.
    L : float, optional
        Length, by default 1

    Returns
    -------
    float
        Pressure drop (Pa).

    See also
    --------
    * [`fD_Colebrook`](fD_Colebrook.md): method to estimate the friction factor.
    * [`fD_Haaland`](fD_Haaland.md): method to estimate the friction factor.
    """
    return fD * rho / 2 * v**2 / D * L


def fD_Colebrook(Re: float, er: float) -> float:
    r"""Darcy friction factor equation by Colebrook.

    $$  \frac{1}{\sqrt{f}}= -2 \log \left( \frac {\epsilon/D} {3.7} +
        \frac {2.51} {\mathrm{Re} \sqrt{f}} \right) $$

    Parameters
    ----------
    Re : float
        Reynolds number.
    er : float
        Relative roughness, $\epsilon/D$.

    **References**

    *   Colebrook, C F (1939). "Turbulent Flow in Pipes, with Particular Reference
        to the Transition Region Between the Smooth and Rough Pipe Laws". Journal
        of the Institution of Civil Engineers. 11 (4): 133-156.

    Returns
    -------
    float
        Darcy friction factor.

    See also
    --------
    * [`fD_Haaland`](fD_Haaland.md): alternative method.
    """

    def func(f):
        return 2*log10(er/3.7 + 2.51/(Re*sqrt(f))) + 1/sqrt(f)

    sol = root_newton(func, fD_Haaland(Re, er), 1e-5)

    return sol.x


def fD_Haaland(Re: float, er: float) -> float:
    r"""Darcy friction factor equation by Haaland.

    $$ \frac{1}{\sqrt{f}}= -1.8 \log \left[\left(\frac{\epsilon/D}{3.7}\right)^{1.11} 
       + \frac{6.9}{\mathrm{Re}} \right] $$

    Parameters
    ----------
    Re : float
        Reynolds number.
    er : float
        Relative roughness, $\epsilon/D$.

    **References**

    *   Haaland, S. E. (March 1, 1983). "Simple and Explicit Formulas for the
        Friction Factor in Turbulent Pipe Flow." ASME. J. Fluids Eng. March 1983;
        105(1): 89-90.

    Returns
    -------
    float
        Darcy friction factor.

    See also
    --------
    * [`fD_Colebrook`](fD_Colebrook.md): alternative method.
    """
    return (1/(1.8*log10((er/3.7)**1.11 + 6.9/Re)))**2
