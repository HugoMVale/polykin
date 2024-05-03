# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Literal, Union

import numpy as np
from numba import jit
from scipy.integrate import solve_ivp

from polykin.utils.exceptions import ODESolverError
from polykin.utils.math import eps
from polykin.utils.types import (FloatArray, FloatArrayLike, FloatMatrix,
                                 FloatVectorLike)

__all__ = ['inst_copolymer_binary',
           'kp_average_binary',
           'monomer_drift_binary']


@jit
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

    See also
    --------
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


def monomer_drift_binary(f10: Union[float, FloatVectorLike],
                         x: FloatVectorLike,
                         r1: float,
                         r2: float,
                         atol: float = 1e-4,
                         rtol: float = 1e-4,
                         method: Literal['LSODA', 'RK45'] = 'LSODA'
                         ) -> FloatMatrix:
    r"""Compute the monomer composition drift for a binary system.

    In a closed binary system, the drift in monomer composition is given by
    the solution of the following differential equation:

    $$ \frac{\textup{d} f_1}{\textup{d}x} = \frac{f_1 - F_1}{1 - x} $$

    with initial condition $f_1(0)=f_{1,0}$, where $f_1$ and $F_1$ are,
    respectively, the instantaneous comonomer and copolymer composition of
    M1, and $x$ is the total molar monomer conversion.

    Parameters
    ----------
    f10 : float | FloatVectorLike (N)
        Initial molar fraction of M1, $f_{1,0}=f_1(0)$.
    x : FloatVectorLike (M)
        Total monomer conversion values where the drift is to be evaluated.
    r1 : float | FloatArray
        Reactivity ratio of M1.
    r2 : float | FloatArray
        Reactivity ratio of M2.
    atol : float
        Absolute tolerance of ODE solver.
    rtol : float
        Relative tolerance of ODE solver.
    method : Literal['LSODA', 'RK45']
        ODE solver.

    Returns
    -------
    FloatMatrix (M, N)
        Monomer fraction of M1 at a given conversion, $f_1(x)$.

    See also
    --------
    * [`monomer_drift_multi`](monomer_drift_multi.md):
      generic method for multicomponent systems.

    Examples
    --------
    >>> from polykin.copolymerization import monomer_drift_binary

    An example with f10 as scalar.
    >>> f1 = monomer_drift_binary(f10=0.5, x=[0.1, 0.5, 0.9], r1=0.16, r2=0.70)
    >>> f1
    array([0.51026241, 0.57810678, 0.87768138])

    An example with f10 as list.
    >>> f1 = monomer_drift_binary(f10=[0.2, 0.8], x=[0.1, 0.5, 0.9],
    ...                           r1=0.16, r2=0.70)
    >>> f1
    array([[0.19841009, 0.18898084, 0.15854395],
           [0.82315475, 0.94379024, 0.99996457]])
    """

    if isinstance(f10, (int, float)):
        f10 = [f10]

    sol = solve_ivp(df1dx,
                    (0., max(x)),
                    f10,
                    args=(r1, r2),
                    t_eval=x,
                    method=method,
                    vectorized=True,
                    atol=atol,
                    rtol=rtol)

    if sol.success:
        result = sol.y
        result = np.maximum(0., result)
        if result.shape[0] == 1:
            result = result[0]
    else:
        raise ODESolverError(sol.message)

    return result


@jit
def df1dx(x: float, f1: FloatArray, r1: float, r2: float) -> FloatArray:
    """Skeist equation for a binary system.

    Parameters
    ----------
    x : float
        Total monomer conversion.
    f1 : FloatArray
        Molar fraction of M1.
    r1 : float
        Reactivity ratio of M1.
    r2 : float
        Reactivity ratio of M2.

    Returns
    -------
    FloatVector
        df1/dx.
    """
    return (f1 - inst_copolymer_binary(f1, r1, r2))/(1 - x + eps)

# %% Jacobian for monomer_drift_binary
# Tried, but does not really accelerate computation
# @jit
# def jac(x: float, f1: FloatVector, r1, r2) -> FloatVector:
#     f2 = 1 - f1
#     dF1df1 = (r1*f1**2 + r2*f2*(1 + (-1 + 2*r1)*f1)) / \
#         (r2*f2**2 + f1*(2 + (-2 + r1)*f1))**2
#     return (1 - dF1df1)/(1 - x + eps)
