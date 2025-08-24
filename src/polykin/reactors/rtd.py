# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from math import erf, gamma

from numpy import exp, inf, pi, sqrt
from scipy import integrate

__all__ = ['E_cstr',
           'F_cstr',
           'E_tanks_series',
           'F_tanks_series',
           'E_laminar_flow',
           'F_laminar_flow',
           'E_dispersion_model',
           'F_dispersion_model',
           'Pe_tube',
           ]


def E_cstr(t: float, tavg: float) -> float:
    r"""Differential residence time distribution for a single CSTR.

    $$ E(t) = \frac{1}{\bar{t}} e^{-t/\bar{t}} $$

    **References**

    * Levenspiel, O. "Chemical reaction engineering", 3rd ed., John Wiley &
      Sons, 1999, p. 322.

    Parameters
    ----------
    t : float
        Residence time.
    tavg : float
        Average residence time, $\bar{t}$.

    Returns
    -------
    float
        Differential residence time distribution.

    See also
    --------
    * [`F_cstr`](F_cstr.md): related method to determine the cumulative
      distribution.
    """
    return (1/tavg)*exp(-t/tavg)


def F_cstr(t: float, tavg: float) -> float:
    r"""Cumulative residence time distribution for a single CSTR.

    $$ F(t) = 1 - e^{-t/\bar{t}} $$

    **References**

    * Levenspiel, O. "Chemical reaction engineering", 3rd ed., John Wiley &
      Sons, 1999, p. 327.

    Parameters
    ----------
    t : float
        Residence time.
    tavg : float
        Average residence time, $\bar{t}$.

    Returns
    -------
    float
        Cumulative residence time distribution.

    See also
    --------
    * [`E_cstr`](E_cstr.md): related method to determine the differential
      distribution.
    """
    return 1 - exp(-t/tavg)


def E_tanks_series(t: float, tavg: float, N: int) -> float:
    r"""Differential residence time distribution for a series of equal CSTRs.

    $$ E(t) =  \frac{1}{\bar{t}} \left(\frac{t}{\bar{t}} \right)^{N-1}
               \frac{N^N}{(N-1)!} e^{-N t / \bar{t}} $$

    **References**

    * Levenspiel, O. "Chemical reaction engineering", 3rd ed., John Wiley &
      Sons, 1999, p. 322.

    Parameters
    ----------
    t : float
        Residence time.
    tavg : float
        Total average residence time, $\bar{t}$.
    N : int
        Number of tanks in series.

    Returns
    -------
    float
        Differential residence time distribution.

    See also
    --------
    * [`F_tanks_series`](F_tanks_series.md): related method to determine the
      cumulative distribution.
    """
    q = t/tavg
    if q == inf:
        return 0
    else:
        return q**(N-1) * (N**N / gamma(N)) * exp(-N*q) / tavg


def F_tanks_series(t: float, tavg: float, N: int) -> float:
    r"""Cumulative residence time distribution for a series of equal CSTRs.

    $$ F(t) = 1 - e^{-N t / \bar{t}} \;
        \sum_{i=0}^{N-1} \frac{(N t / \bar{t})^i}{i!}  $$

    **References**

    * Levenspiel, O. "Chemical reaction engineering", 3rd ed., John Wiley &
      Sons, 1999, p. 327.

    Parameters
    ----------
    t : float
        Residence time.
    tavg : float
        Total average residence time, $\bar{t}$.
    N : int
        Number of tanks in series.

    Returns
    -------
    float
        Cumulative residence time distribution.

    See also
    --------
    * [`E_tanks_series`](E_tanks_series.md): related method to determine the
      differential distribution.
    """
    q = t/tavg
    if q == 0:
        return 0
    elif q == inf:
        return 1
    else:
        S = sum((N*q)**i / gamma(i+1) for i in range(1, N))
        return 1 - exp(-N*q) * (1 + S)


def E_laminar_flow(t: float, tavg: float) -> float:
    r"""Differential residence time distribution for a tubular reactor with
    laminar flow.

    $$ E(t) =
    \begin{cases}
        0 ,                               & t < \bar{t}/2 \\
        \frac{1}{2}\frac{\bar{t}^2}{t^3}, & \text{else}
    \end{cases} $$

    **References**

    * Levenspiel, O. "Chemical reaction engineering", 3rd ed., John Wiley &
      Sons, 1999, p. 343.

    Parameters
    ----------
    t : float
        Residence time.
    tavg : float
        Average residence time, $\bar{t}$.

    Returns
    -------
    float
        Differential residence time distribution.

    See also
    --------
    * [`F_laminar_flow`](F_laminar_flow.md): related method to determine the
      cumulative distribution.
    """
    q = t/tavg
    if q < 1/2:
        return 0
    else:
        return 0.5 * tavg**2 / t**3


def F_laminar_flow(t: float, tavg: float) -> float:
    r"""Cumulative residence time distribution for a tubular reactor with
    laminar flow.

    $$ F(t) =
    \begin{cases}
        0 , & t \le \bar{t}/2 \\
        1 - \frac{1}{4} (t/\bar{t})^{-2} , & \text{else}
    \end{cases} $$

    **References**

    * Levenspiel, O. "Chemical reaction engineering", 3rd ed., John Wiley &
      Sons, 1999, p. 344.

    Parameters
    ----------
    t : float
        Residence time.
    tavg : float
        Average residence time, $\bar{t}$.

    Returns
    -------
    float
        Cumulative residence time distribution.

    See also
    --------
    * [`E_laminar_flow`](E_laminar_flow.md): related method to determine the
      differential distribution.
    """
    q = t/tavg
    if q <= 1/2:
        return 0
    else:
        return 1 - 1/(4*q**2)


def E_dispersion_model(t: float, tavg: float, Pe: float) -> float:
    r"""Differential residence time distribution for the dispersed plug flow
    model (also known as dispersion model).

    Only approximate analytical solutions are available for this model. For
    small deviations from plug flow, i.e. when $Pe > 10^2$, the distribution is
    approximated by:

    $$ E(t) = \frac{1}{\bar{t}} \sqrt{\frac{Pe}{4\pi}}
              \exp\left[-\frac{Pe}{4}(1-\theta)^2\right]  $$

    Otherwise, when $Pe \le 10^2$, the distribution is approximated by the
    so-called open-open solution:

    $$ E(t) = \frac{1}{\bar{t}} \sqrt{\frac{Pe}{4\pi\theta}}
              \exp\left[-\frac{Pe}{4}\frac{(1-\theta)^2}{\theta}\right]  $$

    where $\theta = t/\bar{t}$, and $Pe = (v L)/D$. 

    **References**

    * Levenspiel, O. "Chemical reaction engineering", 3rd ed., John Wiley &
      Sons, 1999, p. 296-301.

    Parameters
    ----------
    t : float
        Residence time.
    tavg : float
        Average residence time, $\bar{t}$.
    Pe : float
        Péclet number based on reactor length, $(v L)/D$.

    Returns
    -------
    float
        Differential residence time distribution.

    See also
    --------
    * [`F_dispersion_model`](F_dispersion_model.md): related method to determine
      the cumulative distribution.
    * [`Pe_tube`](Pe_tube.md): method to estimate the Péclet number.
    """
    q = t/tavg
    if q == 0 or q == inf:
        return 0
    else:
        if Pe > 1e2:
            return sqrt(Pe/pi)/(2*tavg) * exp(-Pe/4*(1 - q)**2)
        else:
            return sqrt(Pe/(pi*q))/(2*tavg) * exp(-Pe/(4*q)*(1 - q)**2)


def F_dispersion_model(t: float, tavg: float, Pe: float) -> float:
    r"""Cumulative residence time distribution for the dispersed plug flow
    model (also known as dispersion model).

    Only approximate analytical solutions are available for this model. For
    small deviations from plug flow, i.e. when $Pe > 10^2$, the distribution is
    approximated by:

    $$ F(t) = \frac{1}{2}\left[ 1 +
        \mathrm{erf} \left( \frac{\sqrt{Pe}}{2}(\theta-1)\right) \right]  $$

    Otherwise, when $Pe \le 10^2$, the distribution is approximated by the
    numerical integral of the so-called open-open solution:

    $$ F(t) = \frac{1}{\bar{t}} \sqrt{\frac{Pe}{4\pi}}
        \int_0^{\theta} \frac{1}{\sqrt{\theta'}}
        \exp\left[-\frac{Pe}{4}\frac{(1-\theta')^2}{\theta'}\right] d\theta' $$

    where $\theta = t/\bar{t}$, and $Pe = (v L)/D$.

    Parameters
    ----------
    t : float
        Residence time.
    tavg : float
        Average residence time, $\bar{t}$.
    Pe : float
        Péclet number based on reactor length, $(v L)/D$.

    Returns
    -------
    float
        Cumulative residence time distribution.

    See also
    --------
    * [`E_dispersion_model`](E_dispersion_model.md): related method to determine
      the differential distribution.
    * [`Pe_tube`](Pe_tube.md): method to estimate the Péclet number.
    """
    q = t/tavg
    if q == 0:
        return 0
    elif q == inf:
        return 1
    else:
        if Pe > 1e2:
            return 0.5*(1 + erf(sqrt(Pe)/2*(q - 1)))
        else:
            result, _ = integrate.quad(
                E_dispersion_model,
                a=0.,
                b=t,
                args=(tavg, Pe),
                epsabs=1e-5)
            return result


def Pe_tube(Re: float, Sc: float | None = None) -> float:
    r"""Calculate the Péclet number for flow through a circular tube.

    For laminar flow, the tube Péclet number $Pe=(v d_t)/D$ is estimated by the
    following expression:

    $$ Pe = \left( \frac{1}{Re Sc} + \frac{Re Sc}{192} \right)^{-1} $$

    where $Re$ is the Reynolds number and $Sc$ is the Schmidt number.

    For turbulent flow, the tube Péclet number is estimated by the following
    expression:

    $$ Pe = \left( \frac{3\times10^{7}}{Re^{2.1}} + \frac{1.35}{Re^{0.125}} \right)^{-1} $$  

    !!! note

        The equation gives the Péclet number based on tube diameter. To obtain
        the Péclet number based on tube length, the Péclet number must be
        multiplied by the length-to-diameter ratio.

    **References**

    * Levenspiel, O. "Chemical reaction engineering", 3rd ed., John Wiley &
      Sons, 1999, p. 310.

    Parameters
    ----------
    Re : float
        Reynolds number based on tube diameter.
    Sc : float | None
        Schmidt number. Required only for laminar flow.

    Returns
    -------
    float
        Péclet number based on tube diameter.
    """
    if Re < 2300:
        if Sc is None:
            raise ValueError("`Sc` must be specified for laminar flow.")
        return 1/(1/(Re*Sc) + (Re*Sc)/192)
    else:
        # to be checked / improved
        return 1/(3e7/Re**2.1 + 1.35/Re**0.125)
