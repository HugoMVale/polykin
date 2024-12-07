# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from math import gamma

from numpy import exp

__all__ = ['cstr_differential',
           'cstr_cumulative',
           'tanks_series_differential',
           'tanks_series_cumulative',
           ]


def cstr_differential(t: float, tavg: float) -> float:
    r"""Differential residence time distribution for a single CSTR.

    $$ E(t) = \frac{1}{\bar{t}} e^{-t/\bar{t}} $$

    **References**

    * Levenspiel, O. "Chemical reaction engineering", John wiley & sons, 1998, 
      p. 322.

    Parameters
    ----------
    t : float
        Time.
    tavg : float
        Average residence time, $\bar{t}$.

    Returns
    -------
    float
        Differential residence time distribution.
    """
    return (1/tavg)*exp(-t/tavg)


def cstr_cumulative(t: float, tavg: float) -> float:
    r"""Cumulative residence time distribution for a single CSTR.

    $$ F(t) = 1 - e^{-t/\bar{t}} $$

    **References**

    * Levenspiel, O. "Chemical reaction engineering", John wiley & sons, 1998, 
      p. 327.

    Parameters
    ----------
    t : float
        Time.
    tavg : float
        Average residence time, $\bar{t}$.

    Returns
    -------
    float
        Cumulative residence time distribution.
    """
    return 1 - exp(-t/tavg)


def tanks_series_differential(t: float, tavg: float, N: int) -> float:
    r"""Differential residence time distribution for a series of equal CSTRs.

    $$ E(t) =  \frac{1}{\bar{t}} \left(\frac{t}{\bar{t}} \right)^{N-1} 
               \frac{N^N}{(N-1)!} e^{-N t / \bar{t}} $$

    **References**

    * Levenspiel, O. "Chemical reaction engineering", John wiley & sons, 1998, 
      p. 322.

    Parameters
    ----------
    t : float
        Time.
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


def tanks_series_cumulative(t: float, tavg: float, N: int) -> float:
    r"""Cumulative residence time distribution for a series of equal CSTRs.

    $$ F(t) = 1 - e^{-N t / \bar{t}} \; 
        \sum_{i=0}^{N-1} \frac{(N t / \bar{t})^i}{i!}  $$

    **References**

    * Levenspiel, O. "Chemical reaction engineering", John wiley & sons, 1998, 
      p. 327.

    Parameters
    ----------
    t : float
        Time.
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
