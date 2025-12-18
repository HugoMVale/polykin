# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from polykin.utils.types import FloatArray

__all__ = [
    "mu_PowerLaw",
    "mu_Cross",
    "mu_Cross_modified",
    "mu_Carreau_Yasuda",
    "aT_WLF",
]


def mu_PowerLaw(
    gdot: float | FloatArray,
    K: float,
    n: float,
) -> float | FloatArray:
    r"""Calculate the viscosity of a fluid using the Power-Law model.

    The viscosity $\mu$ at a given shear rate $\dot{\gamma}$ is calculated
    using the following equation:

    $$ \mu = K \dot{\gamma}^{n-1} $$

    where $K$ is the consistency and $n$ is the flow index.

    Parameters
    ----------
    gdot : float | FloatArray
        Shear rate [s⁻¹].
    K : float
        Consistency [Pa·sⁿ].
    n : float
        Flow index.

    Returns
    -------
    float | FloatArray
        Viscosity at the given shear rate [Pa·s].

    Examples
    --------
    Determine the viscosity of a fluid with a consistency of 10 Pa·sⁿ and a
    flow index of 0.2, at a shear rate of 1e3 s⁻¹.
    >>> from polykin.flow import mu_PowerLaw
    >>> gdot = 1e3  # s⁻¹
    >>> K = 10.0    # Pa·sⁿ
    >>> n = 0.2
    >>> mu = mu_PowerLaw(gdot, K, n)
    >>> print(f"mu={mu:.2e} Pa.s")
    mu=3.98e-02 Pa.s
    """
    return K * gdot ** (n - 1)


def mu_Cross(
    gdot: float | FloatArray,
    mu0: float,
    lmbda: float,
    n: float,
) -> float | FloatArray:
    r"""Calculate the viscosity of a fluid using the Cross model.

    The viscosity $\mu$ at a given shear rate $\dot{\gamma}$ is calculated
    using the following equation:

    $$ \mu = \frac{\mu_0}{1 + (\lambda \dot{\gamma})^{1-n}} $$

    where $\mu_0$ is the zero-shear viscosity, $\lambda$ is the relaxation time,
    and $n$ is the power-law index.

    Parameters
    ----------
    gdot : float | FloatArray
        Shear rate [s⁻¹].
    mu0 : float
        Zero-shear viscosity [Pa·s].
    lmbda : float
        Relaxation constant [s].
    n : float
        Power-law index.

    Returns
    -------
    float | FloatArray
        Viscosity at the given shear rate [Pa·s].

    See Also
    --------
    * [`mu_Cross_modified`](mu_Cross_modified.md): modified version of the Cross
      model.

    Examples
    --------
    Determine the viscosity of a fluid with a zero-shear viscosity of 1.0 Pa·s,
    a relaxation time of 1 second, and a power-law index of 0.2, at a shear rate
    of 20 s⁻¹.
    >>> from polykin.flow import mu_Cross
    >>> gdot = 20.0   # s⁻¹
    >>> mu0 = 1.0     # Pa·s
    >>> lmbda = 1.0   # s
    >>> n = 0.2
    >>> mu = mu_Cross(gdot, mu0, lmbda, n)
    >>> print(f"mu={mu:.2e} Pa.s")
    mu=8.34e-02 Pa.s
    """
    return mu0 / (1 + (lmbda * gdot) ** (1 - n))


def mu_Cross_modified(
    gdot: float | FloatArray,
    mu0: float,
    C: float,
    n: float,
) -> float | FloatArray:
    r"""Calculate the viscosity of a fluid using the modified Cross model.

    The viscosity $\mu$ at a given shear rate $\dot{\gamma}$ is calculated
    using the following equation:

    $$ \mu = \frac{\mu_0}{1 + (C \mu_0 \dot{\gamma})^{1-n}} $$

    where $\mu_0$ is the zero-shear viscosity, $C$ is the relaxation constant,
    and $n$ is the power-law index.

    Parameters
    ----------
    gdot : float | FloatArray
        Shear rate [s⁻¹].
    mu0 : float
        Zero-shear viscosity [Pa·s].
    C : float
        Relaxation constant [Pa⁻¹].
    n : float
        Power-law index.

    Returns
    -------
    float | FloatArray
        Viscosity at the given shear rate [Pa·s].

    See Also
    --------
    * [`mu_Cross`](mu_Cross.md): unmodified version of the Cross model.

    Examples
    --------
    Determine the viscosity of a fluid with a zero-shear viscosity of 1e6 Pa·s,
    a relaxation constant of 2e-5 Pas⁻¹, and a power-law index of 0.2, at a shear
    rate of 1.0 s⁻¹.
    >>> from polykin.flow import mu_Cross_modified
    >>> gdot = 1.0  # s⁻¹
    >>> mu0 = 1e6   # Pa·s
    >>> C = 2e-5    # Pa⁻¹
    >>> n = 0.2
    >>> mu = mu_Cross_modified(gdot, mu0, C, n)
    >>> print(f"mu={mu:.2e} Pa.s")
    mu=8.34e+04 Pa.s
    """
    return mu0 / (1 + (C * mu0 * gdot) ** (1 - n))


def mu_Carreau_Yasuda(
    gdot: float | FloatArray,
    mu0: float,
    muinf: float,
    lmbda: float,
    n: float,
    a: float = 2.0,
) -> float | FloatArray:
    r"""Calculate the viscosity of a fluid using the Carreau-Yasuda model.

    The viscosity $\mu$ at a given shear rate $\dot{\gamma}$ is calculated
    using the following equation:

    $$ \mu = \mu_{\infty} + \frac{\mu_0 - \mu_{\infty}}
                {\left[1 + (\lambda \dot{\gamma})^a\right]^{\frac{1-n}{a}}} $$

    where $\mu_0$ is the zero-shear viscosity, $\mu_{\infty}$ is the infinite-shear
    viscosity, $\lambda$ is the relaxation time, $n$ is the power-law index,
    and $a$ is the Yasuda transition parameter.

    Parameters
    ----------
    gdot : float | FloatArray
        Shear rate [s⁻¹].
    mu0 : float
        Zero-shear viscosity [Pa·s].
    muinf : float
        Infinite-shear viscosity [Pa·s].
    lmbda : float
        Relaxation constant [s].
    n : float
        Power-law index.
    a : float
        Yasuda transition parameter (often assumed equal to 2).

    Returns
    -------
    mu : float | FloatArray
        Viscosity at the given shear rate [Pa·s].

    Examples
    --------
    Determine the viscosity of a fluid with a zero-shear viscosity of 1.0 Pa·s,
    an infinite-shear viscosity of 0.001 Pa·s, a relaxation time of 1 second,
    a power-law index of 0.2, and a Yasuda parameter of 2.0, at a shear rate of
    20 s⁻¹.
    >>> from polykin.flow import mu_Carreau_Yasuda
    >>> gdot = 20.0   # s⁻¹
    >>> mu0 = 1.0     # Pa·s
    >>> muinf = 1e-3  # Pa·s
    >>> lmbda = 1.0   # s
    >>> n = 0.2
    >>> a = 2.0
    >>> mu = mu_Carreau_Yasuda(gdot, mu0, muinf, lmbda, n, a)
    >>> print(f"mu={mu:.2e} Pa.s")
    mu=9.18e-02 Pa.s
    """
    return muinf + (mu0 - muinf) * (1 + (lmbda * gdot) ** a) ** ((n - 1) / a)


def aT_WLF(
    T: float,
    T0: float,
    C1: float | None = None,
    C2: float | None = None,
) -> float:
    r"""Calculate the temperature shift factor using the Williams-Landel-Ferry
    equation.

    The temperature shift factor $a_T$ at a given temperature $T$ is calculated
    using the following equation:

    $$ a_T = 10^{\frac{-C_1 (T - T_0)}{C_2 + (T - T_0)}} $$

    where $T_0$ is the reference temperature, and $C_1$ and $C_2$ are the WLF
    constants.

    If $C_1$ and $C_2$ are not provided, the universal values ($C_1 = 17.44$
    and $C_2 = 51.60$ K) are used, in which case $T_0$ is expected to be the
    glass-transition temperature $T_g$.

    The application of this equation is tipically limited to the range
    $T_g \leq T \leq T_g + 100$ K.

    **References**

    * Stephen L. Rosen, "Fundamental principles of polymeric materials", Wiley,
      2nd edition, 1993, p. 339.

    Parameters
    ----------
    T : float
        Temperature [K].
    T0 : float
        Reference temperature [K], usually taken to be the glass transition
        temperature.
    C1 : float | None
        WLF constant. If None, the universal value is used.
    C2 : float | None
        WLF constant [K]. If None, the universal value is used.

    Returns
    -------
    float
        Temperature shift factor.

    Examples
    --------
    Determine the temperature shift factor at 120 °C for polystyrene, given
    that its glass transition temperature is 100 °C.
    >>> from polykin.flow import aT_WLF
    >>> T = 120 + 273.15  # K
    >>> T0 = 100 + 273.15 # K
    >>> aT = aT_WLF(T, T0)
    >>> print(f"aT={aT:.2e}")
    aT=1.34e-05
    """
    if C1 is None and C2 is None:
        C1 = 17.44
        C2 = 51.60
    elif C1 is not None and C2 is not None:
        pass
    else:
        raise ValueError("Both C1 and C2 must be provided, or neither.")

    return 10 ** (-C1 * (T - T0) / (C2 + T - T0))
