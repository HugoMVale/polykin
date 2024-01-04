# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatOrArray

__all__ = ['inst_copolymer_binary',
           'average_kp_binary']


def inst_copolymer_binary(f1: FloatOrArray,
                          r1: FloatOrArray,
                          r2: FloatOrArray
                          ) -> FloatOrArray:
    r"""Calculate the instantaneous copolymer composition using the
    [Mayo-Lewis](https://en.wikipedia.org/wiki/Mayo%E2%80%93Lewis_equation)
     equation.

    For a binary system, the instantaneous copolymer composition $F_i$ is
    related to the comonomer composition $f_i$ by:

    $$ F_1=\frac{r_1 f_1^2 + f_1 f_2}{r_1 f_1^2 + 2 f_1 f_2 + r_2 f_2^2} $$

    where $r_i$ are the reactivity ratios. Although the equation is written
    using terminal model notation, it is equally applicable in the frame of the
    penultimate model if $r_i \rightarrow \bar{r}_i$.

    References
    ----------
    *   [Mayo & Lewis (1944)](https://doi.org/10.1021/ja01237a052)

    Parameters
    ----------
    f1 : FloatOrArray
        Molar fraction of M1.
    r1 : FloatOrArray
        Reactivity ratio of M1, $r_1$ or $\bar{r}_1$.
    r2 : FloatOrArray
        Reactivity ratio of M2, $r_2$ or $\bar{r}_2$.

    Returns
    -------
    FloatOrArray
        Instantaneous copolymer composition, $F_1$.

    !!! note annotate "See also"

        * [`inst_copolymer_ternary`](../multicomponent/inst_copolymer_ternary.md):
          method for terpolymer systems.
        * [`inst_copolymer_multicomponent`](../multicomponent/inst_copolymer_multicomponent.md):
          method for multicomponent systems.
    """
    f2 = 1 - f1
    return (r1*f1**2 + f1*f2)/(r1*f1**2 + 2*f1*f2 + r2*f2**2)


def average_kp_binary(f1: FloatOrArray,
                      r1: FloatOrArray,
                      r2: FloatOrArray,
                      k11: FloatOrArray,
                      k22: FloatOrArray
                      ) -> FloatOrArray:
    r"""Calculate the average propagation rate coefficient.

    For a binary system, the instantaneous average propagation rate
    coefficient is related to the instantaneous comonomer composition by:

    $$ \bar{k}_p = \frac{r_1 f_1^2 + r_2 f_2^2 + 2f_1 f_2}
        {(r_1 f_1/k_{11}) + (r_2 f_2/k_{22})} $$

    where $f_i$ is the instantaneous comonomer composition of monomer $i$,
    $r_i$ are the reactivity ratios, and $k_{ii}$ are the homo-propagation
    rate coefficients. Although the equation is written using terminal
    model notation, it is equally applicable in the frame of the
    penultimate model if $r_i \rightarrow \bar{r}_i$ and
    $k_{ii} \rightarrow \bar{k}_{ii}$.

    Parameters
    ----------
    f1 : FloatOrArray
        Molar fraction of M1.
    r1 : float
        Reactivity ratio of M1, $r_1$ or $\bar{r}_1$.
    r2 : float
        Reactivity ratio of M2, $r_2$ or $\bar{r}_2$.
    k11 : float
        Propagation rate coefficient of M1, $k_{11}$ or $\bar{k}_{11}$.
        Unit = L/(mol·s)
    k22 : float
        Propagation rate coefficient of M2, $k_{22}$ or $\bar{k}_{22}$.
        Unit = L/(mol·s)

    Returns
    -------
    FloatOrArray
        Average propagation rate coefficient. Unit = L/(mol·s)
    """
    f2 = 1 - f1
    return (r1*f1**2 + r2*f2**2 + 2*f1*f2)/((r1*f1/k11) + (r2*f2/k22))
