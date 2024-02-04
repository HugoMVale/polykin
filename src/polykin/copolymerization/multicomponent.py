# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from numpy import exp
from scipy.integrate import solve_ivp
from typing import Optional

from polykin.utils.types import (FloatMatrix, FloatOrArray, FloatSquareMatrix,
                                 FloatVector, FloatVectorLike)
from polykin.utils.exceptions import ShapeError, ODESolverError
from polykin.utils.math import eps

__all__ = ['inst_copolymer_ternary',
           'inst_copolymer_multi',
           'monomer_drift_multi',
           'convert_Qe_to_r',
           'transitions_multi']

# %% Multicomponent versions of the Mayo-Lewis equation
#
# By definition, $r_{ii}=1$, but the matrix is _not_ symmetrical. For the
# particular case of a binary system, the matrix is given by:
#
#    $$ r = \begin{bmatrix}
#            1   & r1 \\
#            r_2 & 1
#           \end{bmatrix} $$
#


def inst_copolymer_ternary(f1: FloatOrArray,
                           f2: FloatOrArray,
                           r12: float,
                           r21: float,
                           r13: float,
                           r31: float,
                           r23: float,
                           r32: float,
                           ) -> tuple[FloatOrArray, FloatOrArray, FloatOrArray]:
    r"""Calculate the instantaneous copolymer composition for a ternary system.

    For a ternary system, the instantaneous copolymer composition $F_i$ is
    related to the monomer composition $f_i$ by:

    \begin{aligned}
        a &= \frac{f_1}{r_{21} r_{31}} + \frac{f_2}{r_{21} r_{32}} + \frac{f_3}{r_{31} r_{23}} \\
        b &= f_1 + \frac{f_2}{r_{12}} + \frac{f_3}{r_{13}} \\
        c &= \frac{f_1}{r_{12} r_{31}} + \frac{f_2}{r_{12} r_{32}} + \frac{f_3}{r_{13} r_{32}} \\
        d &= f_2 + \frac{f_1}{r_{21}} + \frac{f_3}{r_{23}} \\
        e &= \frac{f_1}{r_{13} r_{21}} + \frac{f_2}{r_{23} r_{12}} + \frac{f_3}{r_{13} r_{23}} \\
        g &= f_3 + \frac{f_1}{r_{31}} + \frac{f_2}{r_{32}} \\
        F_1 &= \frac{a b f_1}{a b f_1 + c d f_2 + e g f_3} \\
        F_2 &= \frac{c d f_2}{a b f_1 + c d f_2 + e g f_3} \\
        F_3 &= \frac{e g f_3}{a b f_1 + c d f_2 + e g f_3}
    \end{aligned}

    where $r_{ij}=k_{ii}/k_{ij}$ are the multicomponent reactivity ratios.

    **References**

    *   Kazemi, N., Duever, T.A. and Penlidis, A. (2014), Demystifying the
    estimation of reactivity ratios for terpolymerization systems. AIChE J.,
    60: 1752-1766.

    Parameters
    ----------
    f1 : FloatOrArray
        Molar fraction of M1.
    f2 : FloatOrArray
        Molar fraction of M2.
    r12 : float
        Reactivity ratio.
    r21 : float
        Reactivity ratio.
    r13 : float
        Reactivity ratio.
    r31 : float
        Reactivity ratio.
    r23 : float
        Reactivity ratio.
    r32 : float
        Reactivity ratio.

    Returns
    -------
    tuple[FloatOrArray, FloatOrArray, FloatOrArray]:
        Instantaneous terpolymer composition, $(F_1, F_2, F_3)$.

    !!! note annotate "See also"

        * [`inst_copolymer_binary`](../binary/inst_copolymer_binary.md):
          specific method for binary systems.
        * [`inst_copolymer_multi`](../multicomponent/inst_copolymer_multi.md):
          generic method for multicomponent systems.

    Examples
    --------
    >>> from polykin.copolymerization import inst_copolymer_ternary
    >>> 
    >>> F1, F2, F3 = inst_copolymer_ternary(f1=0.5, f2=0.3, r12=0.2, r21=2.3,
    ...                                     r13=3.0, r31=0.9, r23=0.4, r32=1.5) 
    >>> 
    >>> print(f"F1 = {F1:.2f}; F2 = {F2:.2f}; F3 = {F3:.2f}")
    F1 = 0.32; F2 = 0.41; F3 = 0.27

    """
    f3 = 1. - (f1 + f2)

    a = f1/(r21*r31) + f2/(r21*r32) + f3/(r31*r23)
    b = f1 + f2/r12 + f3/r13
    c = f1/(r12*r31) + f2/(r12*r32) + f3/(r13*r32)
    d = f2 + f1/r21 + f3/r23
    e = f1/(r13*r21) + f2/(r23*r12) + f3/(r13*r23)
    g = f3 + f1/r31 + f2/r32

    denominator = f1*a*b + f2*c*d + f3*e*g

    F1 = f1*a*b/denominator
    F2 = f2*c*d/denominator
    F3 = 1. - (F1 + F2)

    return (F1, F2, F3)


def inst_copolymer_multi(f: FloatVector,
                         r: FloatSquareMatrix
                         ) -> FloatVector:
    """Calculate the instantaneous copolymer composition for a system with an
    arbitrary number of monomers.

    This algorithm relies on general linear algebra procedure applicable to
    systems with any number of monomers.

    Parameters
    ----------
    f : FloatVector
        Vector (N) of instantaneous monomer composition.
    r : FloatSquareMatrix
        Reactivity ratio matrix (NxN), $r_{ij}=k_{ii}/k_{ij}$.

    Returns
    -------
    FloatVector
        Vector (N) of instantaneous copolymer composition.

    !!! note annotate "See also"

        * [`inst_copolymer_binary`](../binary/inst_copolymer_binary.md):
          specific method for binary systems.
        * [`inst_copolymer_ternary`](../multicomponent/inst_copolymer_ternary.md):
          specific method for terpolymer systems.

    **References**

    *   Jung, W. Mathematical modeling of free-radical six-component bulk
        and solution polymerization. MS thesis. University of Waterloo, 2008.

    Examples
    --------
    >>> from polykin.copolymerization import inst_copolymer_multi
    >>> import numpy as np

    Define reactivity ratio matrix.
    >>> r = np.ones((3, 3))
    >>> r[0, 1] = 0.2
    >>> r[1, 0] = 2.3
    >>> r[0, 2] = 3.0
    >>> r[2, 0] = 0.9
    >>> r[1, 2] = 0.4
    >>> r[2, 1] = 1.5

    Evaluate instantaneous copolymer composition at f1=0.5, f2=0.3, f3=0.2.
    >>> inst_copolymer_multi(np.array([0.5, 0.3, 0.2]), r)
    array([0.32138111, 0.41041608, 0.26820282])
    """
    # Compute radical probabilities, p(i)
    N = len(f)
    A = np.empty((N - 1, N - 1))
    for m in range(N - 1):
        for n in range(N - 1):
            if m == n:
                a = -f[m] / r[N - 1, m]
                for j in range(N):
                    if j != m:
                        a -= f[j] / r[m, j]
            else:
                a = f[m]*(1./r[n, m] - 1./r[N - 1, m])
            A[m, n] = a

    b = -f[:-1] / r[-1, :-1]
    p = np.linalg.solve(A, b)
    p = np.append(p, 1. - np.sum(p, dtype=np.float64))

    # Compute copolymer compositions, F(i)
    F = p*np.sum(f/r, axis=1)/np.sum(p[:, np.newaxis]*f/r)

    return F

# %% Multicomponent Skeist equation


def monomer_drift_multi(f0: FloatVectorLike,
                        r: FloatSquareMatrix,
                        x: FloatVectorLike,
                        rtol: float = 1e-4
                        ) -> FloatMatrix:
    r"""Compute the monomer composition drift for a system with an arbitrary
    number of monomers.

    In a closed system, the drift in monomer composition is given by
    the solution of the following system of differential equations:

    $$ \frac{\textup{d} f_i}{\textup{d}x} = \frac{f_i - F_i}{1 - x} $$

    with initial condition $f_i(0)=f_{i,0}$, where $f_i$ and $F_i$ are,
    respectively, the instantaneous comonomer and copolymer composition of
    monomer $i$, and $x$ is the total molar monomer conversion.

    Parameters
    ----------
    f0 : FloatVectorLike
        Vector (N) of initial instantaneous comonomer composition.
    r : FloatSquareMatrix
        Reactivity ratio matrix (NxN).
    x : FloatVectorLike
        Vector (M) of total monomer conversion values where the drift is to be
        evaluated.
    rtol : float
        Relative tolerance of ODE solver.

    Returns
    -------
    FloatMatrix
        Matrix (MxN) of monomer fraction of monomer $i$ at the specified
        total monomer conversion(s), $f_i(x)$.

    Examples
    --------
    >>> from polykin.copolymerization import monomer_drift_multi
    >>> import numpy as np
    >>>
    >>> r = np.ones((3, 3))
    >>> r[0, 1] = 0.2
    >>> r[1, 0] = 2.3
    >>> r[0, 2] = 3.0
    >>> r[2, 0] = 0.9
    >>> r[1, 2] = 0.4
    >>> r[2, 1] = 1.5
    >>>
    >>> f = monomer_drift_multi(f0=np.array([0.5, 0.3, 0.2]),
    ...                         r=r, x=np.array([0.1, 0.5, 0.9, 0.99]))
    >>> f
    array([[5.19230749e-01, 2.87888151e-01, 1.92881100e-01],
           [6.38387269e-01, 2.04571229e-01, 1.57041502e-01],
           [8.31725111e-01, 5.64177982e-03, 1.62633109e-01],
           [5.00285275e-01, 1.93845681e-07, 4.99714531e-01]])

    """

    N = len(f0)
    if r.shape != (N, N):
        raise ShapeError("Shape mismatch between `f0` and `r`.")

    def dfdx(x: float, f: FloatVector) -> FloatVector:
        F = inst_copolymer_multi(np.append(f, 1 - f.sum()), r)
        return (f - F[:-1]) / (1. - x + eps)

    sol = solve_ivp(dfdx,
                    t_span=(0., x[-1]),
                    y0=f0[:-1],
                    t_eval=x,
                    rtol=rtol,
                    method='LSODA')

    if sol.success:
        f = np.empty((len(x), N))
        f[:, :-1] = np.transpose(sol.y)
        f[:, -1] = 1 - np.sum(f[:, :-1], axis=1)
    else:
        raise ODESolverError(sol.message)

    return f

# %% Multicomponent transition probabilities


def transitions_multi(f: FloatVectorLike,
                      r: FloatSquareMatrix
                      ) -> FloatSquareMatrix:
    r"""Calculate the instantaneous transition probabilities.

    For a multicomponent system, the transition probabilities are given
    by:

    $$ P_{ij} = \frac{r_{ij}^{-1} f_j}{\sum_k r_{ik}^{-1} f_k} $$

    where $f_i$ is the molar fraction of monomer $i$ and $r_{ij}=k_{ii}/k_{ij}$
    is the multicomponent reactivity ratio matrix.

    **References**

    *   NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 178.

    Parameters
    ----------
    f : FloatVectorLike
        Vector (N) of instantaneous monomer composition.
    r : FloatSquareMatrix
        Reactivity ratio matrix(NxN), $r_{ij}=k_{ii}/k_{ij}$.

    Returns
    -------
    FloatSquareMatrix
        Matrix (NxN) of transition probabilities.

    Examples
    --------
    >>> from polykin.copolymerization import transitions_multi
    >>> import numpy as np
    >>>
    >>> r = np.ones((3, 3))
    >>> r[0, 1] = 0.2
    >>> r[1, 0] = 2.3
    >>> r[0, 2] = 3.0
    >>> r[2, 0] = 0.9
    >>> r[1, 2] = 0.4
    >>> r[2, 1] = 1.5
    >>>
    >>> P = transitions_multi([0.5, 0.3, 0.2], r)
    >>> P
    array([[0.24193548, 0.72580645, 0.03225806],
           [0.21367521, 0.29487179, 0.49145299],
           [0.58139535, 0.20930233, 0.20930233]])

    """
    # N = len(f)
    # P = np.empty((N, N))
    # for i in range(N):
    #     for j in range(N):
    #         P[i, j] = f[j]/r[i, j] / np.sum(f/r[i, :])
    f = np.asarray(f)
    return (f/r) / np.sum(f/r, axis=1)[:, np.newaxis]

# %% Multicomponent sequence


# def sequence_multi(f: FloatVectorLike,
#                    r: FloatSquareMatrix,
#                    k: Optional[IntOrArrayLike] = None,
#                    ) -> dict[str, FloatOrArray]:
#     r"""Calculate the instantaneous sequence length probability or the
#     number-average sequence length.

#     For a binary system, the probability of finding $k$ consecutive units
#     of monomer $i$ in a chain is:

#     $$ S_{i,k} = (1 - P_{ii})P_{ii}^{k-1} $$

#     and the corresponding number-average sequence length is:

#     $$ \bar{S}_i = \sum_k k S_{i,k} = \frac{1}{1 - P_{ii}} $$

#     where $P_{ii}$ is the transition probability $i \rightarrow i$, which
#     is a function of the monomer composition and the reactivity ratios.

#     **References**

#     * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
#     process modeling, Wiley, 1996, p. 177.

#     Parameters
#     ----------
#     k : int | None
#         Sequence length, i.e., number of consecutive units in a chain.
#         If `None`, the number-average sequence length will be computed.
#     f1 : FloatOrArrayLike
#         Molar fraction of M1.

#     Returns
#     -------
#     dict[str, FloatOrArray]
#         If `k is None`, the number-average sequence lengths,
#         {'1': $\bar{S}_1$, '2': $\bar{S}_2$}. Otherwise, the
#         sequence probabilities, {'1': $S_{1,k}$, '2': $S_{2,k}$}.
#     """

#     P11, _, _, P22 = self.transitions(f1).values()

#     if k is None:
#         result = {str(i + 1): 1/(1 - P + eps)
#                   for i, P in enumerate([P11, P22])}
#     else:
#         if isinstance(k, (list, tuple)):
#             k = np.array(k, dtype=np.int32)
#         result = {str(i + 1): (1. - P)*P**(k - 1)
#                   for i, P in enumerate([P11, P22])}

#     return result

# %% Multicomponent Q-e


def convert_Qe_to_r(Qe_values: list[tuple[float, float]]
                    ) -> FloatMatrix:
    r"""Convert Q-e values to reactivity ratios.

    According to the Q-e scheme proposed by Alfrey and Price, the reactivity
    ratios of the terminal model can be estimated using the relationship:

    $$ r_{ij} = \frac{Q_i}{Q_j}\exp{\left(-e_i(e_i -e_j)\right)} $$

    where $Q_i$ and $e_i$ are monomer-specific constants, and
    $r_{ij}=k_{ii}/k_{ij}$ is the multicomponent reactivity ratio matrix.

    **References**

    *   T Alfrey, CC Price. J. Polym. Sci., 1947, 2: 101-106.

    Parameters
    ----------
    Qe_values : list[tuple[float, float]]
        List (N) of Q-e values.

    Returns
    -------
    FloatSquareMatrix
        Reactivity ratio matrix (NxN).

    Examples
    --------
    Estimate the reactivity ratio matrix for styrene (1), methyl
    methacrylate (2), and vinyl acetate(3) using Q-e values from the
    literature.

    >>> from polykin.copolymerization import convert_Qe_to_r
    >>> 
    >>> Qe1 = (1.0, -0.80)    # Sty
    >>> Qe2 = (0.78, 0.40)    # MMA
    >>> Qe3 = (0.026, -0.88)  # VAc
    >>> 
    >>> convert_Qe_to_r([Qe1, Qe2, Qe3])
    array([[1.00000000e+00, 4.90888315e-01, 4.10035538e+01],
           [4.82651046e-01, 1.00000000e+00, 1.79788736e+01],
           [2.42325444e-02, 1.08066091e-02, 1.00000000e+00]])

    """
    Q = [x[0] for x in Qe_values]
    e = [x[1] for x in Qe_values]
    N = len(Q)
    r = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            r[i, j] = Q[i]/Q[j]*exp(-e[i]*(e[i] - e[j]))
    return r
