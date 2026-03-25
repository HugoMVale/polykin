# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

from numpy import pi, sqrt, tanh

from polykin.constants import NA

__all__ = [
    "kentry_collision",
    "kentry_diffusion",
    "kentry_diffusion_reversible",
]


def kentry_collision(
    r: float,
    C: float,
) -> float:
    r"""Radical entry coefficient assuming irreversible collision-controlled entry.

    The entry coefficient is given by:

    $$ k_e = 4 \pi r^2 C N_A $$

    where $r$ is the particle radius, $C$ is the collision coefficient, and $N_A$ is
    Avogadro's number.

    **References**

    *   Gardon, J. L. Emulsion Polymerization. I. Recalculation and Extension of the
        Smith-Ewart Theory. J. Polym. Sci., Part A-1: Polym. Chem. 1968, 6, 623-641.

    Parameters
    ----------
    r : float
        Particle radius [m].
    C : float
        Collision coefficient [m/s].

    Returns
    -------
    float
        Radical entry coefficient [m³/(mol·s)].

    Examples
    --------
    Evaluate the radical entry coefficient for a particle radius of 100 nm and a collision
    coefficient of 1e-2 m/s.
    >>> from polykin.kinetics import kentry_collision
    >>> ke = kentry_collision(r=100e-9, C=1e-2)
    >>> print(f"ke = {ke:.2e} m³/(mol·s)")
    ke = 7.57e+08 m³/(mol·s)
    """
    return 4 * pi * (r**2) * C * NA


def kentry_diffusion(
    r: float,
    Dw: float,
    f: float = 1.0,
) -> float:
    r"""Radical entry coefficient assuming irreversible diffusion-controlled entry.

    The entry coefficient is given by:

    $$ k_e = 4 \pi r D_w N_A f $$

    where $r$ is the particle radius, $D_w$ is the diffusion coefficient of the radical in
    the aqueous phase, $N_A$ is Avogadro's number, and $f$ is an efficiency factor.

    Parameters
    ----------
    r : float
        Particle radius [m].
    Dw : float
        Diffusion coefficient of the radical in the aqueous phase [m²/s].
    f : float
        Efficiency factor.

    Returns
    -------
    float
        Radical entry coefficient [m³/(mol·s)].

    See Also
    --------
    * [`kentry_diffusion_reversible`](kentry_diffusion_reversible.md):
      Related method assuming reversible diffusion-controlled entry.

    Examples
    --------
    Evaluate the radical entry coefficient for a particle radius of 100 nm and a diffusion
    coefficient of 1e-9 m²/s.
    >>> from polykin.kinetics import kentry_diffusion
    >>> ke = kentry_diffusion(r=100e-9, Dw=1e-9)
    >>> print(f"ke = {ke:.2e} m³/(mol·s)")
    ke = 7.57e+08 m³/(mol·s)
    """
    return f * 4 * pi * r * Dw * NA


def kentry_diffusion_reversible(
    r: float,
    Dw: float,
    Dp: float,
    q: float,
    k: float,
) -> float:
    r"""Radical entry coefficient assuming reversible diffusion-controlled entry.

    The entry coefficient is given by:

    $$ k_e = 4 \pi r D_w N_A
       \frac{q D_p (X \coth{X} - 1)}{D_w + q D_p (X \coth{X} - 1)} $$

    where

    $$ X = r \sqrt{\frac{k}{D_p}} $$

    Here, $r$ is the particle radius, $D_w$ and $D_p$ are the diffusion coefficients of
    the radical in the aqueous and particle phases, respectively, $q$ is the partition
    coefficient (particle/aqueous), and $N_A$ is Avogadro's number.

    The parameter $k$ is an effective first-order rate coefficient describing radical
    disappearance within the particle:

    $$ k = k_p^p [M^p] + \frac{n k_t^p}{N_A v_p} $$

    where $k_p^p$ and $k_t^p$ are the propagation and termination rate coefficients in
    the particle, $[M^p]$ is the monomer concentration in the particle, $n$ is the number
    of radicals per particle, and $v_p$ is the particle volume.

    Parameters
    ----------
    r : float
        Particle radius [m].
    Dw : float
        Diffusion coefficient in the aqueous phase [m²/s].
    Dp : float
        Diffusion coefficient in the particle phase [m²/s].
    q : float
        Partition coefficient of the radical (particle/aqueous).
    k : float
        First-order rate coefficient for radical disappearance inside the particle [s⁻¹].

    Returns
    -------
    float
        Radical entry coefficient [m³/(mol·s)].

    See Also
    --------
    * [`kentry_diffusion`](kentry_diffusion.md):
      Related method assuming irreversible diffusion-controlled entry.

    Examples
    --------
    Evaluate the radical entry coefficient for a particle of radius 100 nm, with diffusion
    coefficients of 1e-9 m²/s in the aqueous phase and 1e-8 m²/s in the particle phase, a
    partition coefficient of 100, and a first-order rate coefficient of 1e5 s⁻¹.
    >>> from polykin.kinetics import kentry_diffusion_reversible
    >>> ke = kentry_diffusion_reversible(r=100e-9, Dw=1e-9, Dp=1e-8, q=100, k=1e5)
    >>> print(f"ke = {ke:.2e} m³/(mol·s)")
    ke = 7.35e+08 m³/(mol·s)
    """
    X = r * sqrt(k / Dp)
    Y = X / tanh(X) - 1.0
    f = (q * Dp * Y) / (Dw + q * Dp * Y)
    return kentry_diffusion(r, Dw, f)
