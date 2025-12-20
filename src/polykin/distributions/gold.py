from math import factorial

import numpy as np
import scipy.special as sp
from numpy import exp, sqrt
from numpy import log as ln

from polykin.distributions.analyticaldistributions import poisson
from polykin.math import root_brent
from polykin.utils.math import eps
from polykin.utils.types import FloatArray, IntArrayLike


def WeibullNycanderGold_pdf(
    k: int | IntArrayLike,
    v: float,
    r: float,
) -> float | FloatArray:
    r"""Weibull, Nycander and Gold's analytical chain-length distribution for
    living polymerization with different initiation and propagation rate
    coefficients.

    For a living polymerization with only initiation and propagation (i.e.,
    constant number of chains), the number fraction of chains of length
    $k$ can be computed in two steps. First, the number fraction of unreacted
    initiator molecules, $p_0=p(0)$, is found by solving the equation:

    $$ v + r \ln{p_0} + (r - 1)(1 - p_0) = 0 $$

    where $v$ denotes the number-average degree of polymerization of all chains,
    including unreacted initiator molecules, and $r=k_p/k_i$ is the ratio of the
    propagation and initiation rate coefficients. Then, the number fraction
    of chains with $k \ge 1$ monomer units can be evaluated by one of two
    expressions, depending on the value of $r$. For $r > 1$:

    $$ p(k) = \frac{p_0}{r} \left(\frac{r}{r-1} \right)^k P(k,a) $$

    where $a = (1-r) \ln{p_0}$ and $P(k,a) = \gamma(k,a)/\Gamma(k)$ denotes the
    regularized lower incomplete gamma function. For $r < 1$:

    $$ p(k) = \frac{p_0}{r} \left(\frac{r}{1-r} \right)^k (-1)^k
              \left[1 - e^{-a} \sum_{j=0}^{k-1} \frac{a^j}{j!} \right] $$

    This branched analytical solution has an obvious singularity at $r=1$; in
    that case, the solution reduces to the well-known Poisson distribution:

    $$ p(k) = \frac{v^k}{k!}e^{-v} $$

    valid for $k \ge 0$.

    !!! note

        * The solution is numerically unstable in certain domains, namely for
        $r \to 1^{-}$ and $r \to 0$, and also for $k \gg v$. This is an intrinsic
        feature of the analytical solution.
        * For $r < 1$, the calculation is not vectorized. Therefore, performance
        will be worse in this regime.
        * For $|r-1| < 10^{-3}$, the algorithm automatically switches to the
        Poisson distribution. Some numerical discontinuity at this boundary is
        to be expected.

    **References**

    * Weibull, B.; Nycander, E.. "The Distribution of Compounds Formed in the
    Reaction between Ethylene Oxide and Water, Ethanol, Ethylene Glycol, or
    Ethylene Glycol Monoethyl Ether." Acta Chemica Scandinavica 8 (1954):
    847-858.
    * Gold, L. "Statistics of polymer molecular size distribution for an
    invariant number of propagating chains." The Journal of Chemical Physics
    28.1 (1958): 91-99.

    Parameters
    ----------
    k : int | IntArrayLike
        Chain length (>=0).
    v : float
        Number-average degree of polymerization considering chains with zero
        length.
    r : float
        Ratio of propagation and initiation rate coefficients.

    Returns
    -------
    float | FloatArray
        Number probability density.

    Examples
    --------
    Compute the fraction of chains with lengths 0 to 2 for a system with $r=5$
    and $v=1$.

    >>> from polykin.distributions import WeibullNycanderGold_pdf
    >>> WeibullNycanderGold_pdf([0, 1, 2], 1.0, 5.0)
    array([0.58958989, 0.1295864 , 0.11493254])
    """
    r = float(r)
    scalar_input = np.isscalar(k)
    k = np.atleast_1d(k).astype(int)

    # Special case where it reduces to Poisson(kmin=0)
    if abs(r - 1.0) < 1e-3:
        return poisson(k, v, True)

    # Find p0
    def find_p0(p0):
        return v + r * ln(p0) + (r - 1.0) * (1.0 - p0)

    if find_p0(eps) < 0.0:
        sol = root_brent(find_p0, eps, 1.0, tolx=1e-12, tolf=1e-12)
        p0 = sol.x
    else:
        # Limiting analytical solution for p0->0
        p0 = exp((1.0 - v - r) / r)

    # Special case k=0
    result = np.zeros_like(k, dtype=np.float64)
    result[k == 0] = p0

    # Weibull-Nycander-Gold solution p(k>=1)
    mask = k > 0
    kp = k[mask]
    if len(kp) > 0:
        a = (1.0 - r) * ln(p0)
        if r > 1.0:
            result[mask] = exp(ln(p0 / r) + kp * ln(r / (r - 1.0))) * sp.gammainc(kp, a)
        else:
            result[mask] = _gold(kp, v, r, p0, a)

    # Return scalar if input was scalar
    if scalar_input:
        result = result[0]

    return result


@np.vectorize()
def _gold(
    k: int,
    v: float,
    r: float,
    p0: float,
    a: float,
) -> float:
    """Compute Weibull-Nycander-Gold's distribution for r<1.

    Parameters
    ----------
    k : int
        Chain length (>=1).
    v : float
        Number-average degree of polymerization considering chains with zero
        length.
    r : float
        Ratio of propagation and initiation rate coefficients.
    p0 : float
        Fraction of unreacted initiator molecules.
    a : float
        Precomputed value of (1-r)*ln(p0).

    Returns
    -------
    float
        Number probability density.
    """
    A = exp(ln(p0 / r) + k * ln(r / (1.0 - r))) * (-1) ** k
    B = exp(-a)

    # Try normal sum
    s = 1.0
    S = s
    for j in range(1, k):
        s *= a / j
        S += s

    if abs(B * S - 1) > sqrt(eps) and k < v:
        return A * (1.0 - B * S)

    # Tail sum
    s = a**k / factorial(k)
    S = s
    for j in range(k + 1, k + 101):
        s *= a / j
        S += s
        if abs(s) < eps * 100:
            break
    return A * B * S
