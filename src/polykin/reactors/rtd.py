# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from math import gamma

from numpy import exp, pi, sqrt

__all__ = ['E_cstr',
           'F_cstr',
           'E_tanks_series',
           'F_tanks_series',
           'E_laminar_flow',
           'F_laminar_flow',
           'E_dispersion_model',
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
    """
    q = t/tavg
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
    """
    q = t/tavg
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
    small deviations from plug flow, i.e. when $Pe > 10^2$, the residence time
    distribution is approximated by:

    $$ E(t) = \frac{1}{\bar{t}} \sqrt{\frac{Pe}{4\pi}} 
              \exp\left[-\frac{Pe}{4}(1-\theta)^2\right]  $$

    Otherwise, the residence time distribution is approximated by the so-called
    open-open solution:

    $$ E(t) = \frac{1}{\bar{t}} \sqrt{\frac{Pe}{4\pi\theta}} 
              \exp\left[-\frac{Pe}{4}\frac{(1-\theta)^2}{\theta}\right]  $$

    where $\theta = t/\bar{t}$.

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
        Peclet number, $D/(v L)$.

    Returns
    -------
    float
        Differential residence time distribution.
    """
    q = t/tavg
    if Pe > 1e2:
        return sqrt(Pe/pi)/(2*tavg) * exp(-Pe/4*(1 - q)**2)
    else:
        if q == 0:
            return 0
        else:
            return sqrt(Pe/(pi*q))/(2*tavg) * exp(-Pe/(4*q)*(1 - q)**2)
