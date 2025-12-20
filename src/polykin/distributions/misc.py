# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from collections.abc import Callable
from math import comb, factorial

import numpy as np
from numpy.polynomial.laguerre import Laguerre

from polykin.utils.types import FloatArray, FloatArrayLike, FloatVector, FloatVectorLike

__all__ = [
    "convolve_moments",
    "convolve_moments_self",
    "convert_polymer_standards",
    "reconstruct_Laguerre",
]


def convolve_moments(
    q: FloatVectorLike,
    r: FloatVectorLike,
) -> FloatVector:
    r"""Compute the moments of the convolution of two distributions.

    If $P = Q * R$ is the convolution of distributions $Q$ and $R$, defined as:

    $$ P_n = \sum_{m=0}^{n} Q_{n-m}R_{m} $$

    and $p_i$, $q_i$ and $r_i$ denote the $i$-th moments of $P$, $Q$ and $R$,
    respectively,

    \begin{aligned}
        p_i & = \sum_{n=0}^{\infty} n^i P_n \\
        q_i & = \sum_{n=0}^{\infty} n^i Q_n \\
        r_i & = \sum_{n=0}^{\infty} n^i R_n
    \end{aligned}

    then the moments of $P$ are related to the moments of $Q$ and $R$ by:

    $$ p_i = \sum_{j=0}^{i} \binom{i}{j} q_j r_{i-j} $$

    Parameters
    ----------
    q : FloatVectorLike (N)
        Moments of $Q$, denoted $(q_0, q_1, \ldots)$.
    r : FloatVectorLike (N)
        Moments of $R$, denoted $(r_0, r_1, \ldots)$.

    Returns
    -------
    FloatVector (N)
        Moments of $P=Q*R$, denoted $(p_0, p_1, \ldots)$.

    Examples
    --------
    >>> from polykin.distributions import convolve_moments
    >>> convolve_moments([1.0, 1e2, 2e4], [1.0, 50.0, 5e4])
    array([1.0e+00, 1.5e+02, 8.0e+04])
    """
    if len(q) != len(r):
        raise ValueError("`q` and `r` must have the same length.")

    p = np.zeros(len(q))
    for i in range(len(q)):
        for j in range(i + 1):
            p[i] += comb(i, j) * q[j] * r[i - j]

    return p


def convolve_moments_self(
    q: FloatVectorLike,
    order: int,
) -> FloatVector:
    r"""Compute the moments of the k-th order convolution of a distribution
    with itself.

    If $P^{(k)}$ is the $k$-th order convolution of $Q$ with itself, defined as:

    \begin{aligned}
    P^{(1)}_n &= Q*Q = \sum_{m=0}^{n} Q_{n-m} Q_{m} \\
    P^{(2)}_n &= (Q*Q)*Q = \sum_{m=0}^{n} Q_{n-m} P^{(1)}_{m} \\
    P^{(3)}_n &= ((Q*Q)*Q)*Q = \sum_{m=0}^{n} Q_{n-m} P^{(2)}_{m} \\
    ...
    \end{aligned}

    then the moments of $P^{(k)}$ are related to the moments of $Q$ by:

    \begin{aligned}
    p_0 &= q_0^{k+1}  \\
    p_1 &= (k+1) q_0^k q_1 \\
    p_2 &= (k+1) q_0^{k-1} (k q_1^2 +q_0 q_2) \\
    \ldots
    \end{aligned}

    where $p_i$ and $q_i$ denote the $i$-th moments of $P^{(k)}$ and $Q$,
    respectively.

    Parameters
    ----------
    q : FloatVectorLike (N)
        Moments of $Q$, denoted $(q_0, q_1, \ldots)$.
    order : int
        Order of the convolution.

    Returns
    -------
    FloatVector (N)
        Moments of $P^{(k)}=(Q*Q)*...$, denoted $(p_0, p_1, \ldots)$.

    Examples
    --------
    >>> from polykin.distributions import convolve_moments_self
    >>> convolve_moments_self([1e0, 1e2, 2e4], 2)
    array([1.0e+00, 3.0e+02, 1.2e+05])
    """
    q = np.asarray(q, dtype=np.float64)

    if order == 0:
        return q.copy()

    N = len(q)
    if N <= 3:
        # Use closed-form expressions for the first three moments
        p = np.empty(N)
        if N > 0:
            p[0] = q[0] ** (order + 1)
        if N > 1:
            p[1] = (order + 1) * q[0] ** order * q[1]
        if N > 2:
            p[2] = (order + 1) * q[0] ** (order - 1) * (order * q[1] ** 2 + q[0] * q[2])
    else:
        p = q.copy()
        for _ in range(order):
            p = convolve_moments(q, p)

    return p


def convert_polymer_standards(
    M1: float | FloatArrayLike,
    K1: float,
    K2: float,
    a1: float,
    a2: float,
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
    M1 : float | FloatArrayLike
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
    A sample of PMMA was mesured to have a molar mass of 1e5 g/mol in PS
    equivalent weight. What is the sample molar mass in actual PMMA weight?

    >>> from polykin.distributions import convert_polymer_standards
    >>> a1 = 0.77      # PS in THF
    >>> K1 = 6.82e-3   # PS in THF
    >>> a2 = 0.69      # PMMA in THF
    >>> K2 = 1.28e-2   # PMMA in THF
    >>> M2 = convert_polymer_standards(1e5, K1, K2, a1, a2)
    >>> print(f"{M2:.2e} g/mol")
    1.19e+05 g/mol
    """
    M1 = np.asarray(M1, dtype=np.float64)
    return (K1 / K2) ** (1 / (1 + a2)) * M1 ** ((1 + a1) / (1 + a2))


def reconstruct_Laguerre(
    moments: FloatVectorLike,
) -> Callable[[FloatArrayLike], FloatArray]:
    r"""Reconstruct a differential number distribution from a finite set of
    moments using a Laguerre-series approximation.

    According to Bamford and Tompa, a number distribution $P(n)$ can be expressed
    as an (infinite) expansion in Laguerre polynomials:

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
    moments : FloatVectorLike
        Moments of $P$, denoted $(\lambda_0, \lambda_1, \ldots)$.

    Returns
    -------
    Callable[[FloatArrayLike], FloatArray]
        A function `pdf(n)` that evaluates the reconstructed number distribution
        $P(n)$ at the supplied chain lengths `n`.
    """
    moments = np.asarray(moments, dtype=np.float64)
    DPn = moments[1] / moments[0]
    k = len(moments)

    # Series coefficients
    γ = np.empty(k, dtype=np.float64)
    for m in range(k):
        γ[m] = sum(
            comb(m, i) * (-1) ** i * moments[i] / DPn ** (i - 1) / factorial(i)
            for i in range(m + 1)
        )

    # Laguerre series γ0*L0 + γ1*L1 + ... + γm*Lm
    L = Laguerre(γ)

    # Bild pdf
    def pdf(n: FloatArrayLike) -> FloatArray:
        n = np.asarray(n, dtype=np.float64)
        ρ = n / DPn
        return np.exp(-ρ) / DPn**2 * L(ρ)

    return pdf
