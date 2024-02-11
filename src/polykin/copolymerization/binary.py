# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Union

import numpy as np

from polykin.utils.types import FloatArray, FloatArrayLike

__all__ = ['inst_copolymer_binary',
           'kp_average_binary']


def inst_copolymer_binary(f1: Union[float, FloatArrayLike],
                          r1: Union[float, FloatArray],
                          r2: Union[float, FloatArray]
                          ) -> Union[float, FloatArray]:
    r"""Calculate the instantaneous copolymer composition using the
    [Mayo-Lewis](https://en.wikipedia.org/wiki/Mayo%E2%80%93Lewis_equation)
     equation.

    For a binary system, the instantaneous copolymer composition $F_i$ is
    related to the comonomer composition $f_i$ by:

    $$ F_1=\frac{r_1 f_1^2 + f_1 f_2}{r_1 f_1^2 + 2 f_1 f_2 + r_2 f_2^2} $$

    where $r_i$ are the reactivity ratios. Although the equation is written
    using terminal model notation, it is equally applicable in the frame of the
    penultimate model if $r_i \rightarrow \bar{r}_i$.

    **References**

    *   [Mayo & Lewis (1944)](https://doi.org/10.1021/ja01237a052)

    Parameters
    ----------
    f1 : float | FloatArrayLike
        Molar fraction of M1.
    r1 : float | FloatArray
        Reactivity ratio of M1, $r_1$ or $\bar{r}_1$.
    r2 : float | FloatArray
        Reactivity ratio of M2, $r_2$ or $\bar{r}_2$.

    Returns
    -------
    float | FloatArray
        Instantaneous copolymer composition, $F_1$.

    !!! note annotate "See also"

        * [`inst_copolymer_ternary`](inst_copolymer_ternary.md):
          specific method for terpolymer systems.
        * [`inst_copolymer_multi`](inst_copolymer_multi.md):
          generic method for multicomponent systems.

    Examples
    --------
    >>> from polykin.copolymerization import inst_copolymer_binary

    An example with f1 as scalar.
    >>> F1 = inst_copolymer_binary(f1=0.5, r1=0.16, r2=0.70)
    >>> print(f"F1 = {F1:.3f}")
    F1 = 0.406

    An example with f1 as list.
    >>> F1 = inst_copolymer_binary(f1=[0.2, 0.6, 0.8], r1=0.16, r2=0.70)
    >>> F1
    array([0.21487603, 0.45812808, 0.58259325])

    """
    f1 = np.asarray(f1)
    f2 = 1 - f1
    return (r1*f1**2 + f1*f2)/(r1*f1**2 + 2*f1*f2 + r2*f2**2)


def kp_average_binary(f1: Union[float, FloatArrayLike],
                      r1: Union[float, FloatArray],
                      r2: Union[float, FloatArray],
                      k11: Union[float, FloatArray],
                      k22: Union[float, FloatArray]
                      ) -> Union[float, FloatArray]:
    r"""Calculate the average propagation rate coefficient in a
    copolymerization.

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
    f1 : float | FloatArrayLike
        Molar fraction of M1.
    r1 : float | FloatArray
        Reactivity ratio of M1, $r_1$ or $\bar{r}_1$.
    r2 : float | FloatArray
        Reactivity ratio of M2, $r_2$ or $\bar{r}_2$.
    k11 : float | FloatArray
        Propagation rate coefficient of M1, $k_{11}$ or $\bar{k}_{11}$.
        Unit = L/(mol·s)
    k22 : float | FloatArray
        Propagation rate coefficient of M2, $k_{22}$ or $\bar{k}_{22}$.
        Unit = L/(mol·s)

    Returns
    -------
    float | FloatArray
        Average propagation rate coefficient. Unit = L/(mol·s)

    Examples
    --------
    >>> from polykin.copolymerization import kp_average_binary

    An example with f1 as scalar.
    >>> kp = kp_average_binary(f1=0.5, r1=0.16, r2=0.70, k11=100., k22=1000.)
    >>> print(f"{kp:.0f} L/(mol·s)")
    622 L/(mol·s)

    An example with f1 as list.
    >>> f1 = [0.2, 0.6, 0.8]
    >>> kp = kp_average_binary(f1=f1, r1=0.16, r2=0.70, k11=100., k22=1000.)
    >>> kp
    array([880.        , 523.87096774, 317.18309859])
    """
    f1 = np.asarray(f1)
    f2 = 1 - f1
    return (r1*f1**2 + r2*f2**2 + 2*f1*f2)/((r1*f1/k11) + (r2*f2/k22))
