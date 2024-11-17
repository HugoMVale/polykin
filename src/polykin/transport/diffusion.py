# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import cos, exp, pi, sin, sqrt
from scipy.special import erfc

from polykin.math.special import ierfc
from polykin.utils.math import eps

__all__ = ['profile_semiinf',
           'profile_sheet',
           'profile_sphere',
           'uptake_sheet',
           'uptake_sphere']


def profile_semiinf(t: float,
                    x: float,
                    D: float
                    ) -> float:
    r"""Concentration profile for transient diffusion in semi-infinite medium. 

    For a semi-infinite medium, where the concentration is initially $C_0$
    everywhere, and the surface concentration is maintained at $C_s$, the
    normalized concentration is:

    $$ \frac{C - C_0}{C_s - C_0}=
        \mathrm{erfc} \left( \frac{x}{2\sqrt{Dt}} \right) $$

    where $x$ is the distance from the surface, $t$ is the time, and $D$ is
    the diffusion coefficient.

    **References**

    * J. Crank, "The mathematics of diffusion", Oxford University Press, 1975,
      p. 32.

    Parameters
    ----------
    t : float
        Time (s).
    x : float
        Distance from surface (m).
    D : float
        Diffusion coefficient (m²/s).

    Returns
    -------
    float
        Fractional accomplished concentration change.

    Examples
    --------
    Determine the fractional concentration change after 100 s in a thick polymer
    film (diffusivity: 1e-10 m²/s) at a depth of 0.1 mm below the surface.
    >>> from polykin.transport.diffusion import profile_semiinf
    >>> profile_semiinf(t=1e2, x=0.1e-3, D=1e-10)
    0.4795001221869535
    """
    return erfc(x/(2*sqrt(D*t)))


def profile_sheet(t: float,
                  x: float,
                  a: float,
                  D: float
                  ) -> float:
    r"""Concentration profile for transient diffusion in a plane sheet. 

    For a plane sheet of thickness $2a$, with diffusion from _both_ faces, where
    the concentration is initially $C_0$ everywhere, and the concentration at
    both surfaces is maintained at $C_s$, the normalized concentration is:

    $$ \frac{C - C_0}{C_s - C_0} = 
        1-\frac{4}{\pi}\sum_{n=0}^{\infty}\frac{(-1)^n}{2n+1}
        \exp\left[-\frac{D \pi^2 t}{4 a^2} (2n+1)^2 \right] 
        \cos\left[ \frac{\pi x}{2a} (2n+1) \right] $$

    where $x$ is the distance from the center of the sheet, $t$ is the time,
    and $D$ is the diffusion coefficient.

    !!! tip

        This equation is also applicable to a plane sheet of thickness $a$ if
        one of the faces is sealed.

    **References**

    * J. Crank, "The mathematics of diffusion", Oxford University Press, 1975,
      p. 47.

    Parameters
    ----------
    t : float
        Time (s).
    x : float
        Distance from the center or sealed face (m).
    a : float
        Half thickness of sheet (m).
    D : float
        Diffusion coefficient (m²/s).

    Returns
    -------
    float
        Fractional accomplished concentration change.

    See also
    --------
    * [`uptake_sheet`](uptake_sheet.md): related method to determine the mass
      uptake.

    Examples
    --------
    Determine the fractional concentration change after 100 s in a 0.2 mm-thick
    polymer film (diffusivity: 1e-10 m²/s) at its maximum depth.
    >>> from polykin.transport.diffusion import profile_sheet
    >>> profile_sheet(t=1e2, x=0., a=0.2e-3, D=1e-10)
    0.3145542331096479
    """
    N = 4  # Number of terms in series expansion (sufficient for convergence)

    A = 2*sqrt(D*t)/a
    if A < 1.:
        # Solution for small times
        S = sum((-1 if n % 2 else 1) * (erfc(((2*n + 1) - x/a)/A) + erfc(((2*n + 1) + x/a)/A))
                for n in range(0, N))
        result = S
    else:
        # Solution for normal times
        B = -D*pi**2*t/(4*a**2)
        C = pi*x/(2*a)
        S = sum((-1 if n % 2 else 1) / (2*n + 1) * exp(B*(2*n + 1)**2) * cos(C*(2*n + 1))
                for n in range(0, N))
        result = 1 - (4/pi)*S

    return result


def profile_sphere(t: float,
                   r: float,
                   a: float,
                   D: float
                   ) -> float:
    r"""Concentration profile for transient diffusion in a sphere. 

    For a sphere of radius $a$, where the concentration is initially $C_0$
    everywhere, and the surface concentration is maintained at $C_s$, the
    normalized concentration is:

    $$ \frac{C - C_0}{C_s - C_0} =
        1 + \frac{2 a}{\pi r} \sum_{n=1}^\infty \frac{(-1)^n}{n} 
        \exp \left(-\frac{D n^2 \pi^2 t}{a^2} \right)
        \sin \left(\frac{n \pi r}{a} \right) $$

    where $r$ is the radial distance from the center of the sphere, $t$ is the
    time, and $D$ is the diffusion coefficient.

    **References**

    * J. Crank, "The mathematics of diffusion", Oxford University Press, 1975,
      p. 91.

    Parameters
    ----------
    t : float
        Time (s).
    r : float
        Radial distance from center of sphere (m).
    a : float
        Radius of sphere (m).
    D : float
        Diffusion coefficient (m²/s).

    Returns
    -------
    float
        Normalized concentration.

    See also
    --------
    * [`uptake_sphere`](uptake_sphere.md): related method to determine the mass
      uptake.

    Examples
    --------
    Determine the fractional concentration change after 100 s at the center of
    a polymer sphere with a radius of 0.2 mm and a diffusivity of 1e-10 m²/s.
    >>> from polykin.transport.diffusion import profile_sphere
    >>> profile_sphere(t=100., r=0., a=0.2e-3, D=1e-10)
    0.8304935009764247
    """
    N = 4  # Number of terms in series expansion (sufficient for convergence)

    A = 2*sqrt(D*t)/a
    B = -D*pi**2*t/a**2
    C = r/a
    if abs(C) < eps:
        # Solution for particular case r->0
        S = sum((-1 if n % 2 else 1) * exp(B*n**2) for n in range(1, N))
        result = 1 + 2*S
    else:
        if A < 1.:
            # Solution for small times
            S = sum(erfc(((2*n + 1) - C)/A) - erfc(((2*n + 1) + C)/A)
                    for n in range(0, N))
            result = S/C
        else:
            # Solution for normal times
            S = sum((-1 if n % 2 else 1) / n * exp(B*n**2) * sin(C*pi*n)
                    for n in range(1, N))
            result = 1 + (2/(pi*C))*S

    return result


def uptake_sheet(t: float,
                 a: float,
                 D: float
                 ) -> float:
    r"""Fractional mass uptake for transient diffusion in a plane sheet. 

    For a plane sheet of thickness $2a$, with diffusion from _both_ faces, where
    the concentration is initially $C_0$ everywhere, and the concentration at
    both surfaces is maintained at $C_s$, the fractional mass uptake is:

    $$ \frac{M_t}{M_{\infty}} = 
        1 - \frac{8}{\pi^2} \sum_{n=0}^{\infty}\frac{1}{(2n+1)^2}
        \exp\left[\frac{-D(2n+1)^2 \pi^2t}{4 a^2}\right] $$

    where $t$ is the time, and $D$ is the diffusion coefficient.

    !!! tip

        This equation is also applicable to a plane sheet of thickness $a$ if
        one of the faces is sealed.

    **References**

    * J. Crank, "The mathematics of diffusion", Oxford University Press, 1975,
      p. 48.

    Parameters
    ----------
    t : float
        Time (s).
    a : float
        Half thickness of sheet (m).
    D : float
        Diffusion coefficient (m²/s).

    Returns
    -------
    float
        Fractional mass uptake.

    See also
    --------
    * [`profile_sheet`](profile_sheet.md): related method to determine the
      concentration profile.

    Examples
    --------
    Determine the fractional mass uptake after 100 seconds for a polymer solution
    film with a thickness of 0.2 mm and a diffusion coefficient of 1e-10 m²/s.
    >>> from polykin.transport.diffusion import uptake_sheet
    >>> uptake_sheet(t=1e2, a=0.2e-3, D=1e-10)
    0.562233541762138
    """
    N = 4  # Number of terms in series expansion (optimal value)

    A = sqrt(D*t)/a
    if A == 0.:
        result = 0.
    elif A < 0.5:
        # Solution for small times
        S = sum((-1 if n % 2 else 1) * ierfc(n/A) for n in range(1, N))
        result = 2*A * (1/sqrt(pi) + 2*S)
    else:
        # Solution for normal times
        B = -D*pi**2*t/(4*a**2)
        S = sum(1/(2*n + 1)**2 * exp(B*(2*n + 1)**2) for n in range(0, N-1))
        result = 1 - (8/pi**2)*S

    return result


def uptake_sphere(t: float,
                  a: float,
                  D: float
                  ) -> float:
    r"""Fractional mass uptake for transient diffusion in a sphere. 

    For a sphere of radius $a$, where the concentration is initially $C_0$
    everywhere, and the surface concentration is maintained at $C_s$, the
    fractional mass uptake is:

    $$ \frac{M_t}{M_{\infty}} = 
        1 - \frac{6}{\pi^2} \sum_{n=1}^{\infty}\frac{1}{n^2}
        \exp \left( \frac{-D n^2 \pi^2 t}{a^2} \right) $$

    where $t$ is the time, and $D$ is the diffusion coefficient.

    **References**

    * J. Crank, "The mathematics of diffusion", Oxford University Press, 1975,
      p. 91.

    Parameters
    ----------
    t : float
        Time (s).
    a : float
        Radius of sphere (m).
    D : float
        Diffusion coefficient (m²/s).

    Returns
    -------
    float
        Fractional mass uptake.

    See also
    --------
    * [`profile_sphere`](profile_sphere.md): related method to determine the
      concentration profile.

    Examples
    --------
    Determine the fractional mass uptake after 100 seconds for a polymer sphere
    with a radius of 0.2 mm and a diffusion coefficient of 1e-10 m²/s.
    >>> from polykin.transport.diffusion import uptake_sphere
    >>> uptake_sphere(t=1e2, a=0.2e-3, D=1e-10)
    0.9484368978658284
    """
    N = 4  # Number of terms in series expansion (optimal value)

    A = sqrt(D*t)/a
    if A == 0.:
        result = 0.
    elif A < 0.5:
        # Solution for small times
        S = sum(ierfc(n/A) for n in range(1, N))
        result = 6*A * (1/sqrt(pi) + 2*S) - 3*A**2
    else:
        # Solution for normal times
        B = -D*pi**2*t/a**2
        S = sum(1/n**2 * exp(B*n**2) for n in range(1, N))
        result = 1 - (6/pi**2)*S

    return result
