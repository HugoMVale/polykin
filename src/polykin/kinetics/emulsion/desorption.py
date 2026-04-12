# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

__all__ = [
    "kdesorption_Asua",
    "K0_Nomura",
]


def kdesorption_Asua(
    kfmp: float,
    kpp: float,
    Mp: float,
    K0: float,
    beta: float,
) -> float:
    r"""Radical desorption coefficient according to Asua's model.

    The desorption coefficient is given by:

    $$ k_{des} = k^p_{fm}[M^p] \left( \frac{K_0}{\beta K_0 + k^p_p [M^p]} \right) $$

    where $k^p_{fm}$ is the transfer-to-monomer rate coefficient in the particle phase,
    $k^p_p$ is the propagation rate coefficient in the particle phase, $[M^p]$ is the
    monomer concentration in the particle phase, $K_0$ is the monomeric radical desorption
    rate coefficient, and $\beta$ is the probability that the desorbed monomeric radical
    reacts in the aqueous phase by propagation or termination .

    !!! note

        Nomura and Harada's model can be seen as a particular case of Asua's model where
        $\beta=\bar{n}$.


    **References**

    *   Asua, J.M., Sudol, E.D. and El-Aasser, M.S. (1989), Radical desorption in emulsion
        polymerization. J. Polym. Sci. A Polym. Chem., 27: 3903-3913.
    *   Nomura, M. and Harada, M. (1981), Rate coefficient for radical desorption in
        emulsion polymerization. J. Appl. Polym. Sci., 26: 17-26.

    Parameters
    ----------
    kfmp : float
        Transfer-to-monomer rate coefficient in the particle [L/(mol·s)].
    kpp : float
        Propagation rate coefficient in the particle phase [L/(mol·s)].
    Mp : float
        Monomer concentration in the particle phase [mol/L].
    K0 : float
        Monomeric radical desorption coefficient [s⁻¹].
    beta : float
        Probability that the desorbed monomeric radical reacts in the aqueous phase by
        propagation or termination.

    Returns
    -------
    float
        Radical desorption rate coefficient [s⁻¹].

    See Also
    --------
    * [`K0_Nomura`](K0_Nomura.md): Associated method for $K_0$.

    Examples
    --------
    Evaluate the radical desorption coefficient for a system with a transfer-to-monomer
    rate coefficient of 1e-3 L/(mol·s), a propagation rate coefficient of 1e2 L/(mol·s),
    a monomer concentration in the particle phase of 10 mol/L, a monomeric radical
    desorption coefficient of 1e3 s⁻¹, and a probability of 20% that the desorbed
    monomeric radical reacts in the aqueous phase.

    >>> from polykin.kinetics import kdesorption_Asua
    >>> kdes = kdesorption_Asua(kfmp=1e-3, kpp=1e2, Mp=10, K0=1e3, beta=0.20)
    >>> print(f"kdes = {kdes:.2e} s⁻¹")
    kdes = 8.33e-03 s⁻¹
    """
    return (kfmp * Mp * K0) / (beta * K0 + kpp * Mp)


def K0_Nomura(
    Dw: float,
    Dp: float,
    q: float,
    dp: float,
    *,
    b: float = 10.0,
) -> float:
    r"""Monomeric radical desorption coefficient according to Nomura's two-film
    mass-transfer model.

    The monomeric radical desorption coefficient is given by:

    $$ K_0 = \frac{6}{d_p} \left[
             \left( \frac{b D_p}{d_p} \right)^{-1} +
             \left( \frac{2 D_w}{q d_p} \right)^{-1} \right]^{-1} $$

    where $D_w$ is the diffusivity of the monomeric radical in the aqueous phase, $D_p$ is
    the diffusivity of the monomeric radical in the particle phase, $q$ is the partition
    coefficient of the monomeric radical between the particle and aqueous phases, $d_p$
    is the particle diameter, and $b$ is a parameter that characterizes the particle-side
    mass transfer coefficient.

    **References**

    *   Nomura, M. and Harada, M. (1981), Rate coefficient for radical desorption in
        emulsion polymerization. J. Appl. Polym. Sci., 26: 17-26.

    Parameters
    ----------
    Dw : float
        Diffusion coefficient of the monomeric radical in the aqueous phase [m²/s].
    Dp : float
        Diffusion coefficient of the monomeric radical in the particle phase [m²/s].
    q : float
        Partition coefficient of the monomeric radical (particle/aqueous).
    dp : float
        Particle diameter [m].
    b : float
        Proportionality constant for the particle-side mass transfer coefficient.
        Literature values vary substantially: Ugelstad and Hansen (1976) use b=2,
        Nomura and Harada (1981) use b=1/3, Asua et al. (1989) use b=1, and Hernandez and
        Tauer (2008) used b=10. The default value is b=10, corresponding to the linear
        driving force approximation for intraparticle diffusion. This choice is adopted
        here as the most physically consistent representation of particle-side mass
        transfer.

    Returns
    -------
    float
        Monomeric radical desorption coefficient [s⁻¹].

    See Also
    --------
    * [`kdesorption_Asua`](kdesorption_Asua.md): Related desorption coefficient method.

    Examples
    --------
    Evaluate the monomeric radical desorption coefficient for a system with a diffusivity
    of 1e-9 m²/s in the aqueous phase, a diffusivity of 1e-10 m²/s in the particle phase,
    a partition coefficient of 30, and a particle diameter of 200 nm.
    >>> from polykin.kinetics import K0_Nomura
    >>> K0 = K0_Nomura(Dw=1e-9, Dp=1e-10, q=30, dp=200e-9)
    >>> print(f"K0 = {K0:.2e} s⁻¹")
    K0 = 9.38e+03 s⁻¹
    """
    return (12 * b * Dw * Dp) / (dp**2 * (2 * Dw + b * q * Dp))
