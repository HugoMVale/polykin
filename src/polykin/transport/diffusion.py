# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from numpy import cos, exp, pi, sin, sqrt
from scipy.special import erfc

from polykin.math.special import ierfc, roots_xcotx, roots_xtanx
from polykin.utils.math import eps

__all__ = ['profile_constc_semiinf',
           'profile_constc_sheet',
           'profile_constc_sphere',
           'uptake_constc_sheet',
           'uptake_constc_sphere',
           'uptake_convection_sheet',
           'uptake_convection_sphere',
           'diffusivity_composite',
           ]


def profile_constc_semiinf(t: float,
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
    >>> from polykin.transport import profile_constc_semiinf
    >>> profile_constc_semiinf(t=1e2, x=0.1e-3, D=1e-10)
    0.4795001221869535
    """
    return erfc(x/(2*sqrt(D*t)))


def profile_constc_sheet(Fo: float, xstar: float) -> float:
    r"""Concentration profile for transient diffusion in a plane sheet subjected
    to a constant surface concentration boundary condition.

    For a plane sheet of thickness $2a$, with diffusion from _both_ faces, where
    the concentration is initially $C_0$ everywhere, and the concentration at
    both surfaces is maintained at $C_s$, the normalized concentration is:

    $$ \frac{C - C_0}{C_s - C_0} = 
        1-\frac{4}{\pi}\sum_{n=0}^{\infty}\frac{(-1)^n}{2n+1}
        \exp\left[-\frac{\pi^2 Fo}{4} (2n+1)^2 \right] 
        \cos\left[ \frac{\pi x^*}{2} (2n+1) \right] $$

    where $Fo = D t/a^2$ is the Fourier number, and $x^*=x/a$ is the normalized
    distance from the center of the sheet.

    !!! tip

        This equation is also applicable to a plane sheet of thickness $a$ if
        one of the faces is sealed.

    **References**

    * J. Crank, "The mathematics of diffusion", Oxford University Press, 1975,
      p. 47.

    Parameters
    ----------
    Fo : float
        Fourier number, $D t/a^2$.
    xstar : float
        Normalized distance from the center or sealed face, $r/a$.

    Returns
    -------
    float
        Fractional accomplished concentration change.

    See also
    --------
    * [`uptake_constc_sheet`](uptake_constc_sheet.md): related method to
      determine the mass uptake.

    Examples
    --------
    Determine the fractional concentration change after 100 s in a 0.2 mm-thick
    polymer film (diffusivity: 1e-10 m²/s) at its maximum depth.
    >>> from polykin.transport import profile_constc_sheet
    >>> t = 1e2    # s
    >>> a = 0.2e-3 # m
    >>> D = 1e-10  # m²/s
    >>> x = 0.     # m
    >>> Fo = D*t/a**2
    >>> xstar = x/a
    >>> profile_constc_sheet(Fo, xstar)
    0.3145542331096478
    """
    N = 4  # Number of terms in series expansion (sufficient for convergence)

    if Fo == 0:
        result = 0
    elif Fo < 1/4:
        # Solution for small times
        A = 2*sqrt(Fo)
        S = sum((-1 if n % 2 else 1) * (erfc(((2*n + 1) - xstar)/A) + erfc(((2*n + 1) + xstar)/A))
                for n in range(0, N))
        result = S
    else:
        # Solution for normal times
        B = -(pi**2 / 4)*Fo
        C = (pi/2)*xstar
        S = sum((-1 if n % 2 else 1) / (2*n + 1) * exp(B*(2*n + 1)**2) * cos(C*(2*n + 1))
                for n in range(0, N))
        result = 1 - (4/pi)*S

    return result


def profile_constc_sphere(Fo: float, rstar: float) -> float:
    r"""Concentration profile for transient diffusion in a sphere subjected
    to a constant surface concentration boundary condition. 

    For a sphere of radius $a$, where the concentration is initially $C_0$
    everywhere, and the surface concentration is maintained at $C_s$, the
    normalized concentration is:

    $$ \frac{C - C_0}{C_s - C_0} =
        1 + \frac{2}{\pi r^*} \sum_{n=1}^\infty \frac{(-1)^n}{n} 
        \exp(- n^2 \pi^2 Fo) \sin (n \pi r^*) $$

    where $Fo = D t/a^2$ is the Fourier number, and $r^*=r/a$ is the normalized
    radial distance from the center of the sphere.

    **References**

    * J. Crank, "The mathematics of diffusion", Oxford University Press, 1975,
      p. 91.

    Parameters
    ----------
    Fo : float
        Fourier number, $D t/a^2$.
    rstar : float
        Normalized radial distance from the center of the sphere, $r/a$.

    Returns
    -------
    float
        Normalized concentration.

    See also
    --------
    * [`uptake_constc_sphere`](uptake_constc_sphere.md): related method to
      determine the mass uptake.

    Examples
    --------
    Determine the fractional concentration change after 100 s at the center of
    a polymer sphere with a radius of 0.2 mm and a diffusivity of 1e-10 m²/s.
    >>> from polykin.transport import profile_constc_sphere
    >>> t = 1e2    # s
    >>> a = 0.2e-3 # m
    >>> r = 0      # m
    >>> D = 1e-10  # m²/s
    >>> Fo = D*t/a**2
    >>> rstar = r/a
    >>> profile_constc_sphere(Fo, rstar)
    0.8304935009764246
    """
    N = 4  # Number of terms in series expansion (sufficient for convergence)

    if abs(rstar) < eps:
        # Solution for particular case r->0
        B = -(pi**2)*Fo
        S = sum((-1 if n % 2 else 1) * exp(B*n**2) for n in range(1, N))
        result = 1 + 2*S
    else:
        if Fo == 0:
            result = 0
        elif Fo < 1/4:
            # Solution for small times
            A = 2*sqrt(Fo)
            S = sum(erfc(((2*n + 1) - rstar)/A) - erfc(((2*n + 1) + rstar)/A)
                    for n in range(0, N))
            result = S/rstar
        else:
            # Solution for normal times
            B = -(pi**2)*Fo
            S = sum((-1 if n % 2 else 1) / n * exp(B*n**2) * sin(rstar*pi*n)
                    for n in range(1, N))
            result = 1 + (2/(pi*rstar))*S

    return result


def uptake_constc_sheet(Fo: float) -> float:
    r"""Fractional mass uptake for transient diffusion in a plane sheet
    subjected to a constant surface concentration boundary condition. 

    For a plane sheet of thickness $2a$, with diffusion from _both_ faces, where
    the concentration is initially $C_0$ everywhere, and the concentration at
    both surfaces is maintained at $C_s$, the fractional mass uptake is:

    $$ \frac{\bar{C}-C_0}{C_s -C_0} = 
        1 - \frac{8}{\pi^2} \sum_{n=0}^{\infty}\frac{1}{(2n+1)^2}
        \exp\left[-\frac{(2n+1)^2 \pi^2}{4}Fo\right] $$

    where $Fo = D t/a^2$ is the Fourier number.

    !!! tip

        This equation is also applicable to a plane sheet of thickness $a$ if
        one of the faces is sealed.

    **References**

    * J. Crank, "The mathematics of diffusion", Oxford University Press, 1975,
      p. 48.

    Parameters
    ----------
    Fo : float
        Fourier number, $D t/a^2$.

    Returns
    -------
    float
        Fractional mass uptake.

    See also
    --------
    * [`profile_constc_sheet`](profile_constc_sheet.md): related method to
      determine the concentration profile.
    * [`uptake_convection_sheet`](uptake_convection_sheet.md): related method
      for surface convection boundary condition.

    Examples
    --------
    Determine the fractional mass uptake after 500 seconds for a polymer solution
    film with a thickness of 0.2 mm and a diffusion coefficient of 1e-11 m²/s.
    >>> from polykin.transport import uptake_constc_sheet
    >>> t = 5e2   # s
    >>> a = 2e-4  # m
    >>> D = 1e-11 # m²/s
    >>> Fo = D*t/a**2
    >>> uptake_constc_sheet(Fo)
    0.39892798988456807
    """
    N = 4  # Number of terms in series expansion (optimal value)

    if Fo == 0:
        result = 0
    elif Fo < 1/4:
        # Solution for small times
        A = sqrt(Fo)
        S = sum((-1 if n % 2 else 1) * ierfc(n/A) for n in range(1, N))
        result = 2*A * (1/sqrt(pi) + 2*S)
    else:
        # Solution for normal times
        B = -(Fo*pi**2)/4
        S = sum(1/(2*n + 1)**2 * exp(B*(2*n + 1)**2) for n in range(0, N-1))
        result = 1 - (8/pi**2)*S

    return result


def uptake_constc_sphere(Fo: float) -> float:
    r"""Fractional mass uptake for transient diffusion in a sphere subjected
    to a constant surface concentration boundary condition. 

    For a sphere of radius $a$, where the concentration is initially $C_0$
    everywhere, and the surface concentration is maintained at $C_s$, the
    fractional mass uptake is:

    $$ \frac{\bar{C}-C_0}{C_s -C_0} =
    1 - \frac{6}{\pi^2}\sum_{n=1}^{\infty}\frac{1}{n^2} \exp (-n^2 \pi^2 Fo) $$

    where $Fo = D t/a^2$ is the Fourier number.

    **References**

    * J. Crank, "The mathematics of diffusion", Oxford University Press, 1975,
      p. 91.

    Parameters
    ----------
    Fo : float
        Fourier number, $D t/a^2$.

    Returns
    -------
    float
        Fractional mass uptake.

    See also
    --------
    * [`profile_constc_sphere`](profile_constc_sphere.md): related method to
      determine the concentration profile.
    * [`uptake_convection_sphere`](uptake_convection_sphere.md): related method
      for surface convection boundary condition.

    Examples
    --------
    Determine the fractional mass uptake after 100 seconds for a polymer sphere
    with a radius of 0.1 mm and a diffusion coefficient of 1e-11 m²/s.
    >>> from polykin.transport import uptake_constc_sphere
    >>> t = 1e2   # s
    >>> a = 1e-4  # m
    >>> D = 1e-11 # m²/s
    >>> Fo = D*t/a**2
    >>> uptake_constc_sphere(Fo)
    0.7704787380259631
    """
    N = 4  # Number of terms in series expansion (optimal value)

    if Fo == 0:
        result = 0
    elif Fo < 1/4:
        # Solution for small times
        A = sqrt(Fo)
        S = sum(ierfc(n/A) for n in range(1, N))
        result = 6*A * (1/sqrt(pi) + 2*S) - 3*Fo
    else:
        # Solution for normal times
        B = -Fo*pi**2
        S = sum(1/n**2 * exp(B*n**2) for n in range(1, N))
        result = 1 - (6/pi**2)*S

    return result


def uptake_convection_sheet(Fo: float, Bi: float) -> float:
    r"""Fractional mass uptake for transient diffusion in a plane sheet
    subjected to a surface convection boundary condition. 

    For a plane sheet of thickness $2a$, with diffusion from _both_ faces,
    where the concentration is initially $C_0$ everywhere, and the flux at the
    surface is:

    $$ -D \left.\frac{\partial C}{\partial x} \right|_{x=a}=k(C(a,t)-C_{\infty}) $$

    the fractional mass uptake is:

    $$ \frac{\bar{C}-C_0}{C_{\infty} -C_0} =
        1 - 2 Bi^2 \sum_{n=1}^{\infty}\frac{1}{\beta_n^2[\beta_n^2+Bi(Bi+1)]}
        \exp (-\beta_n^2 Fo) $$

    where $Fo = D t/a^2$ is the Fourier number, $Bi = k a/D$ is the Biot
    number, and $\beta_n$ are the positive roots of the transcendental equation 
    $\beta \tan(\beta) = Bi$.

    **References**

    * J. Crank, "The mathematics of diffusion", Oxford University Press, 1975,
      p. 60.

    Parameters
    ----------
    Fo : float
        Fourier number, $D t/a^2$.
    Bi : float
        Biot number, $k a/D$.

    Returns
    -------
    float
        Fractional mass uptake.

    See also
    --------
    * [`uptake_constc_sheet`](uptake_constc_sheet.md): related method for
      constant surface concentration boundary condition.

    Examples
    --------
    Determine the fractional mass uptake after 500 seconds for a polymer solution
    film with a thickness of 0.2 mm, a diffusion coefficient of 1e-11 m²/s, and
    an external mass transfer coefficient of 1e-6 m/s.
    >>> from polykin.transport import uptake_convection_sheet
    >>> t = 5e2   # s
    >>> a = 2e-4  # m
    >>> D = 1e-11 # m²/s
    >>> k = 1e-6  # m/s
    >>> Fo = D*t/a**2
    >>> Bi = k*a/D
    >>> uptake_convection_sheet(Fo, Bi)
    0.3528861780625614
    """
    if Fo and Bi:
        N = 4  # Number of terms in series expansion (optimal value)
        x = roots_xtanx(Bi, N)
        b2 = x**2
        S = sum(exp(-b2[n]*Fo)/(b2[n]*(b2[n] + Bi*(Bi + 1)))
                for n in range(0, N))
        return 1 - (2*Bi**2)*S
    else:
        return 0


def uptake_convection_sphere(Fo: float, Bi: float) -> float:
    r"""Fractional mass uptake for transient diffusion in a sphere subjected
    to a surface convection boundary condition. 

    For a sphere of radius $a$, where the concentration is initially $C_0$
    everywhere, and the flux at the surface is:

    $$ -D \left.\frac{\partial C}{\partial r} \right|_{r=a}=k(C(a,t)-C_{\infty}) $$

    the fractional mass uptake is:

    $$ \frac{\bar{C}-C_0}{C_{\infty} -C_0} = 
        1 - 6 Bi^2 \sum_{n=1}^{\infty}\frac{1}{\beta_n^2[\beta_n^2+Bi(Bi-1)]}
        \exp (-\beta_n^2 Fo) $$

    where $Fo = D t/a^2$ is the Fourier number, $Bi = k a/D$ is the Biot
    number, and $\beta_n$ are the positive roots of the transcendental equation 
    $1 - \beta \cot(\beta) = Bi$.

    **References**

    * J. Crank, "The mathematics of diffusion", Oxford University Press, 1975,
      p. 96.

    Parameters
    ----------
    Fo : float
        Fourier number, $D t/a^2$.
    Bi : float
        Biot number, $k a/D$.

    Returns
    -------
    float
        Fractional mass uptake.

    See also
    --------
    * [`uptake_constc_sphere`](uptake_constc_sphere.md): related method for
      constant surface concentration boundary condition.

    Examples
    --------
    Determine the fractional mass uptake after 100 seconds for a polymer sphere
    with a radius of 0.1 mm, a diffusion coefficient of 1e-11 m²/s, and an
    external mass transfer coefficient of 1e-6 m/s.
    >>> from polykin.transport import uptake_convection_sphere
    >>> t = 1e2   # s
    >>> a = 1e-4  # m
    >>> D = 1e-11 # m²/s
    >>> k = 1e-6  # m/s
    >>> Fo = D*t/a**2
    >>> Bi = k*a/D
    >>> uptake_convection_sphere(Fo, Bi)
    0.6539883120664335
    """
    if Fo and Bi:
        N = 4  # Number of terms in series expansion (optimal value)
        x = roots_xcotx(Bi, N)
        b2 = x**2
        S = sum(exp(-b2[n]*Fo)/(b2[n]*(b2[n] + Bi*(Bi - 1)))
                for n in range(0, N))
        return 1 - (6*Bi**2)*S
    else:
        return 0


def diffusivity_composite(Dd: float,
                          Dc: float,
                          fd: float,
                          sphericity: float = 1) -> float:
    r"""Calculate the effective diffusivity of a composite medium containing a 
    dispersed particle phase.

    The effective diffusivity $D$ is calculated using a generalization of
    Maxwell's analytical solution for spherical particles:

    $$ \frac{D - D_c}{D + x D_c} = \phi_d \frac{D_d - D_c}{D_d + x D_c} $$

    with $x = 3/s - 1$. Here, $D_d$ is the diffusivity of the dispersed phase, 
    $D_c$ is the diffusivity of the continuous phase, $\phi_d$ is the volume
    fraction of the dispersed phase, and $s$ is the sphericity of the dispersed
    particles.

    **References**

    * J. Crank, "The mathematics of diffusion", Oxford University Press, 1975,
      p. 271.

    Parameters
    ----------
    Dd : float
        Diffusity of the dispersed phase.
    Dc : float
        Diffusity of the continuous phase.
    fd : float
        Volume fraction of the dispersed phase.
    sphericity : float
        Sphericity of the particles. Ratio of the surface area of a sphere of 
        volume equal to that of the particle, to the surface area of the
        particle.

    Returns
    -------
    float
        Effective diffusivity of the composite medium.

    Examples
    --------
    Determine the effective diffusivity of a composite medium containing 5 vol%
    of spherical particles with a diffusivity of 1e-10 m²/s. The diffusivity
    of the continuous phase is 1e-11 m²/s.
    >>> from polykin.transport import diffusivity_composite
    >>> diffusivity_composite(Dd=1e-10, Dc=1e-11, fd=0.05)
    1.1168831168831167e-11
    """
    x = 3/sphericity - 1
    Y = (Dd - Dc)/(Dd + x*Dc)
    return Dc*(1 + fd*Y*x)/(1 - fd*Y)
