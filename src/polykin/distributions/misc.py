# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from math import comb, factorial
from typing import Callable

import numpy as np
from numpy.polynomial.laguerre import Laguerre

from polykin.utils.types import FloatArray, FloatArrayLike, FloatOrArrayLike

__all__ = [
    'convolve_moments',
    'convolve_moments_self',
    'convert_polymer_standards',
    'reconstruct_Laguerre',
]


def convolve_moments(q0: float,
                     q1: float,
                     q2: float,
                     r0: float,
                     r1: float,
                     r2: float
                     ) -> tuple[float, float, float]:
    r"""Compute the first three moments of the convolution of two distributions.

    If $P = Q * R$ is the convolution of $Q$ and $R$, defined as:

    $$ P_n = \sum_{i=0}^{n} Q_{n-i}R_{i} $$

    then the first three moments of $P$ are related to the moments of $Q$ and
    $R$ by:

    \begin{aligned}
    p_0 &= q_0 r_0 \\
    p_1 &= q_1 r_0 + q_0 r_1 \\
    p_2 &= q_2 r_0 + 2 q_1 r_1 + q_0 r_2
    \end{aligned}

    where $p_i$, $q_i$ and $r_i$ denote the $i$-th moments of $P$, $Q$ and $R$,
    respectively.    

    Parameters
    ----------
    q0 : float
        0-th moment of $Q$.
    q1 : float
        1-st moment of $Q$.
    q2 : float
        2-nd moment of $Q$.
    r0 : float
        0-th moment of $R$.
    r1 : float
        1-st moment of $R$.
    r2 : float
        2-nd moment of $R$.

    Returns
    -------
    tuple[float, float, float]
        0-th, 1-st and 2-nd moments of $P=Q*R$.

    Examples
    --------
    >>> from polykin.distributions import convolve_moments
    >>> convolve_moments(1., 1e2, 2e4, 1., 50., 5e4)
    (1.0, 150.0, 80000.0)
    """
    p0 = q0*r0
    p1 = q1*r0 + q0*r1
    p2 = q2*r0 + 2*q1*r1 + q0*r2
    return p0, p1, p2


def convolve_moments_self(q0: float,
                          q1: float,
                          q2: float,
                          order: int = 1
                          ) -> tuple[float, float, float]:
    r"""Compute the first three moments of the k-th order convolution of a
    distribution with itself.

    If $P^k$ is the $k$-th order convolution of $Q$ with itself, defined as:

    \begin{aligned}
    P^1_n &= Q*Q = \sum_{i=0}^{n} Q_{n-i} Q_{i} \\
    P^2_n &= (Q*Q)*Q = \sum_{i=0}^{n} Q_{n-i} P^1_{i} \\
    P^3_n &= ((Q*Q)*Q)*Q = \sum_{i=0}^{n} Q_{n-i} P^2_{i} \\
    ...
    \end{aligned}

    then the first three moments of $P^k$ are related to the moments of $Q$ by:

    \begin{aligned}
    p_0 &= q_0^{k+1}  \\
    p_1 &= (k+1) q_0^k q_1 \\
    p_2 &= (k+1) q_0^{k-1} (k q_1^2 +q_0 q_2)
    \end{aligned}

    where $p_i$ and $q_i$ denote the $i$-th moments of $P^k$ and $Q$,
    respectively.    

    Parameters
    ----------
    q0 : float
        0-th moment of $Q$.
    q1 : float
        1-st moment of $Q$.
    q2 : float
        2-nd moment of $Q$.

    Returns
    -------
    tuple[float, float, float]
        0-th, 1-st and 2-nd moments of $P^k=(Q*Q)*...$.

    Examples
    --------
    >>> from polykin.distributions import convolve_moments_self
    >>> convolve_moments_self(1., 1e2, 2e4, 2)
    (1.0, 300.0, 120000.0)
    """
    p0 = q0**(order+1)
    p1 = (order+1) * q0**order * q1
    p2 = (order+1) * q0**(order-1) * (order*q1**2 + q0*q2)
    return p0, p1, p2


def convert_polymer_standards(
    M1: FloatOrArrayLike,
    K1: float,
    K2: float,
    a1: float,
    a2: float
) -> FloatArray:
    r"""Convert a molar mass from a given polymer standard to another using the 
    respective Mark-Houwink parameters.

    The conversion from a polymer standard 1 to a polymer standard 2 is given
    by:

    $$ M_2 = \left(\frac{K_1}{K_2}\right)^{\frac{1}{1 + a_2}} 
            M_1^{\frac{1 + a_1}{1 + a_2}} $$

    where $M_i$ is the molar mass in standard $i$, and $K_i$ and $a_i$ are the
    Mark-Houwink parameters for standard $i$.

    !!! tip

        This tranformation is linear in terms of the logarithm of the molar
        mass, i.e., $d \ln M_2/d \ln M_1 = \frac{1 + a_1}{1 + a_2}$. This means
        that a GPC distribution can be converted from one standard to another
        by applying this transformation to the x-axis. If you need to convert a
        number or weight distribution, then the y-axis must also be converted
        by a suitable approach. 

    Parameters
    ----------
    M1 : FloatOrArrayLike
        Molar mass in standard 1.
    K1 : float
        Mark-Houwink coefficient for standard 1.
    K2 : float
        Mark-Houwink coefficient for standard 2.
    a1 : float
        Mark-Houwink exponent for standard 1.
    a2 : float
        Mark-Houwink exponent for standard 2.

    Returns
    -------
    FloatArray
        Molar mass in standard 2.

    Examples
    --------
    A sample of PMMA was mesured to have a molar mass of 100 kg/mol in PS
    equivalent weight. What is the sample molar mass in actual PMMA weight?

    >>> from polykin.distributions import convert_polymer_standards
    >>> a1 = 0.77      # PS in THF
    >>> K1 = 6.82e-3   # PS in THF
    >>> a2 = 0.69      # PMMA in THF
    >>> K2 = 1.28e-2   # PMMA in THF
    >>> M2 = convert_polymer_standards(100, K1, K2, a1, a2) 
    >>> print(f"{M2:.2f} kg/mol")
    85.68 kg/mol
    """
    M1 = np.asarray(M1, dtype=np.float64)
    return (K1/K2)**(1/(1 + a2)) * M1**((1 + a1)/(1 + a2))


def reconstruct_Laguerre(
    moments: FloatArrayLike,
) -> Callable[[FloatArrayLike], FloatArray]:
    r"""Reconstruct a differential number distribution from its first `k` 
    moments using a Laguerre-series approximation.

    According to Bamford and Tompa, a number distribution can be expressed as
    an (infinite) expansion in Laguerre polynomials:

    $$ P(n) = \frac{e^{-\rho}}{(DP_n)^2} 
              \sum_{m=0}^{\infty} \gamma_m L_m(\rho) $$

    with coefficients:

    $$ \gamma_m = \sum_{i=0}^{m} \binom{m}{i} (-1)^{i} 
                  \frac{\lambda_i}{i!(DP_n)^{i-1}} $$ 

    where $L_m$ is the Laguerre polynomial of degree $m$, $\lambda_i$ is the
    $i$-th moment of the distribution, $DP_n = \lambda_1/ \lambda_0$ is the
    number-average chain length, and $\rho = n/DP_n$.

    In principle, an infinite number of moments is required, but in certain
    well-behaved cases a modest (finite) number is sufficient.

    !!! note

        This method is mainly of historical interest. Its success depends
        strongly on the shape of the underlying distribution. It works well
        when the distribution is close to Flory-like, but may perform poorly
        for more complex shapes.

    **References**

    *   C.H. Bamford and H. Tompa, "The calculation of molecular weight
        distributions from kinetic schemes", Trans. Faraday Soc., 50, 1097 (1954).

    Parameters
    ----------
    moments : FloatArrayLike
        First `k` raw moments of the number distribution (`λ₀`, `λ₁`, ..., `λ_k`).

    Returns
    -------
    Callable[[FloatArrayLike], FloatArray]
        A function `pdf(n)` that evaluates the reconstructed number distribution
        $P(n)$ at the supplied chain lengths `n`.
    """

    moments = np.asarray(moments, dtype=float)
    DPn = moments[1]/moments[0]
    k = len(moments)

    # Series coefficients
    γ = np.empty(k, dtype=float)
    for m in range(k):
        γ[m] = sum(comb(m, i) * (-1)**i * moments[i] / DPn ** (i-1)
                   / factorial(i) for i in range(m+1))

    # Laguerre series γ0*L0 + γ1*L1 + ... + γm*Lm
    L = Laguerre(γ)

    # Bild pdf
    def pdf(n: FloatArrayLike) -> FloatArray:
        n = np.asarray(n, dtype=float)
        ρ = n/DPn
        return np.exp(-ρ)/DPn**2 * L(ρ)

    return pdf
