# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from numpy import log as ln

__all__ = [
    "U_plane_wall",
    "U_cylindrical_wall",
]


def U_plane_wall(
    h1: float,
    h2: float,
    L: float,
    k: float,
    Rf1: float = 0.0,
    Rf2: float = 0.0,
) -> float:
    r"""Calculate the overall heat transfer coefficient through a plane wall
    with convection on both sides.

    Under steady-state conditions, the overall heat transfer coefficient is
    given by the following expression:

    $$ U = \left( \frac{1}{h_1} + \frac{1}{h_2} + \frac{L}{k} + R_{f,1} + R_{f,2} \right)^{-1} $$

    where $h_i$ and $R_{f,i}$ denote, respectively, the heat transfer coefficient
    and fouling factor at surface $i$, $L$ is the wall thickness, and $k$ is the
    wall thermal conductivity.

    Parameters
    ----------
    h1 : float
        Heat transfer coefficient at surface 1 [W/(m²·K)].
    h2 : float
        Heat transfer coefficient at surface 2 [W/(m²·K)].
    L : float
        Wall thickness [m].
    k : float
        Wall thermal conductivity [W/(m·K)].
    Rf1 : float
        Fouling factor at surface 1 [(m²·K)/W].
    Rf2 : float
        Fouling factor at surface 2 [(m²·K)/W].

    Returns
    -------
    float
        Overall heat transfer coefficient [W/(m²·K)].

    See Also
    --------
    - [`U_cylindrical_wall`](U_cylindrical_wall.md): related method for a
      cylindrical wall.

    Examples
    --------
    Calculate the overall heat transfer coefficient for a 10 mm-thick plane
    carbon steel wall subjected to convection on both sides, with heat transfer
    coefficients of 1000 and 2000 W/(m²·K). Neglect fouling effects.
    >>> from polykin.hmt import U_plane_wall
    >>> h1 = 1e3  # W/(m²·K)
    >>> h2 = 2e3  # W/(m²·K)
    >>> k = 6e2   # W/(m·K)
    >>> L = 10e-3 # m
    >>> U = U_plane_wall(h1, h2, L, k)
    >>> print(f"U={U:.1e} W/(m²·K)")
    U=6.6e+02 W/(m²·K)
    """
    return 1 / (1 / h1 + 1 / h2 + L / k + Rf1 + Rf2)


def U_cylindrical_wall(
    hi: float,
    ho: float,
    di: float,
    do: float,
    k: float,
    Rfi: float = 0.0,
    Rfo: float = 0.0,
) -> float:
    r"""Calculate the overall heat transfer coefficient through a cylindrical
    wall with convection on both sides.

    Under steady-state conditions, the overall heat transfer coefficient is
    given by the following expression:

    $$ U_o = \left( \frac{d_o}{h_i d_i} + \frac{1}{h_o} + \frac{R_{f,i}d_o}{d_i}
             + R_{f,o} + \frac{d_o}{2k}\ln(d_o/d_i) \right)^{-1} $$

    where $h$ is the heat transfer coefficient, $R_{f}$ is the fouling factor,
    $d$ is the diameter, $k$ is the wall thermal conductivity, and the subscripts
    $i$ and $o$ indicate inner and outer surfaces, respectively.

    !!! tip

        The overall heat transfer coefficient based on the inner surface, $U_i$,
        can be computed using the indentity $U_i d_i = U_o d_o$.

    Parameters
    ----------
    hi : float
        Heat transfer coefficient at inner surface [W/(m²·K)].
    ho : float
        Heat transfer coefficient at outer surface [W/(m²·K)].
    di : float
        Inner diameter [m].
    do : float
        Outer diameter [m].
    k : float
        Wall thermal conductivity [W/(m·K)].
    Rfi : float
        Fouling factor at inner surface [(m²·K)/W].
    Rfo : float
        Fouling factor at outer surface [(m²·K)/W].

    Returns
    -------
    float
        Overall heat transfer coefficient based on outer surface [W/(m²·K)].

    See Also
    --------
    - [`U_plane_wall`](U_plane_wall.md): related method for a plane wall.

    Examples
    --------
    Calculate the overall heat transfer coefficient for a carbon steel tube
    subjected to convection on both sides, with heat transfer coefficients
    of 2000 and 1000 W/(m²·K) for the inner and outer surfaces, respectively.
    The tube has inner and outer diameters of 40 mm and 50 mm. Neglect fouling
    effects.
    >>> from polykin.hmt import U_cylindrical_wall
    >>> hi = 2e3   # W/(m²·K)
    >>> ho = 1e3   # W/(m²·K)
    >>> k = 6e2    # W/(m·K)
    >>> di = 40e-3 # m
    >>> do = 50e-3 # m
    >>> Uo = U_cylindrical_wall(hi, ho, di, do, k)
    >>> print(f"Uo={Uo:.1e} W/(m²·K)")
    Uo=6.1e+02 W/(m²·K)
    """
    return 1 / (
        do / (hi * di) + 1 / ho + Rfi * do / di + Rfo + do / (2 * k) * ln(do / di)
    )
