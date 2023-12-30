# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import numpy as np
from numpy import exp
from scipy.integrate import solve_ivp

from polykin.types import (FloatArray, FloatMatrix, FloatSquareMatrix,
                           FloatVector, FloatVectorLike)

# %% Multicomponent Mayo-Lewis equation


def inst_copolymer_composition(f: FloatVector,
                               r: FloatSquareMatrix
                               ) -> FloatVector:
    """Compute the instantaneous copolymer composition for an arbitrary
    monomer mixture.

    Parameters
    ----------
    f : FloatVector
        Vector(N-1) of instantaneous monomer composition.
    r : FloatSquareMatrix
        Reactivity ratio matrix (NxN).

    Returns
    -------
    FloatVector
        Vector(N) of instantaneous copolymer composition.
    """

    # Compute radical probabilities, p(i)
    # The radical propabilites can be found from the solution of a system of
    # linear equations, A.p = b (W. Jung, 2008).
    N = len(f) + 1
    flong = np.concatenate((f, [1. - np.sum(f, dtype=np.float64)]))
    A = np.empty((N - 1, N - 1))
    for m in range(N - 1):
        for n in range(N - 1):
            if m == n:
                a = -flong[m] / r[N - 1, m]
                for j in range(N):
                    if j != m:
                        a -= flong[j] / r[m, j]
            else:
                a = flong[m] * (1 / r[n, m] - 1 / r[N - 1, m])
            A[m, n] = a

    b = -flong[:-1] / r[-1, :-1]
    p = np.linalg.solve(A, b)
    p = np.append(p, 1 - np.sum(p, dtype=np.float64))

    # Compute copolymer compositions, F(i)
    denominator = np.sum(p[:, np.newaxis] * flong / r)  # scalar
    numerator = np.sum(flong / r, axis=1)               # vector
    F = p * numerator / denominator

    return F

# %% Multicomponent Skeist equation


def composition_drift(f0: FloatVectorLike,
                      r: FloatSquareMatrix,
                      xteval: FloatVectorLike
                      ) -> tuple[FloatMatrix, FloatVector]:
    """Compute the monomer composition drift for an arbitrary monomer mixture.

    Parameters
    ----------
    f0 : FloatVectorLike
        Vector (N) of initial instantaneous comonomer composition.
    r : FloatSquareMatrix
        Reactivity ratio matrix (NxN).
    xteval : FloatVectorLike
        Vector (M) of total monomer conversion values where the drift is to be
        evaluated.

    Returns
    -------
    tuple[FloatMatrix, FloatVector]
        Tuple of monomer conversion (~xteval) and corresponding comonomer
        composition (MxN).
    """

    N = len(f0)
    if r.shape != (N, N):
        raise ValueError("Shape mismatch between `f0` and `r`.")

    def ode(xt_: float, f_: np.ndarray) -> np.ndarray:
        F = inst_copolymer_composition(f_, r)
        return (f_ - F[:-1]) / (1 - xt_)

    sol = solve_ivp(ode,
                    t_span=(0., xteval[-1]),
                    y0=f0[:-1],
                    t_eval=xteval,
                    rtol=1e-4,
                    method='LSODA')

    if sol.success:
        xt = sol.t
        f = np.empty((len(xteval), N))
        f[:, :-1] = np.transpose(sol.y)
        f[:, -1] = 1 - np.sum(f[:, :-1], axis=1)
    else:
        f = np.array([np.nan])
        xt = np.array([np.nan])
        print(sol.message)

    return (xt, f)

# %% Multicomponent transition probabilities


def transitions_multicomponent(f: FloatArray,
                               r: FloatSquareMatrix
                               ) -> FloatArray:
    r"""Calculate the instantaneous transition probabilities.

    For a multicomponent system, the self-transition probabilities are given
    by:

    $$ P_{ii} &= \frac{f_i}{\sum_j \frac{f_j}{r_{ij}}} $$

    where $f_i$ is the molar fraction of monomer $i$ and $r_{ij}=k_{ii}/k_{ij}$
    is the multicomponent reactivity ratio matrix. By definition, $r_{ii}=1$,
    but the matrix is _not_ symmetrical. For the particular case of a binary
    system, the matrix is given by:

    $$ r = \begin{bmatrix}
            1   & r1 \\
            r_2 & 1
           \end{bmatrix} $$

    Reference:

    * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
    process modeling, Wiley, 1996, p. 178.

    Parameters
    ----------
    f : FloatArray
        Array(..., N) of instantaneous monomer composition.
    r : FloatSquareMatrix
        Reactivity ratio matrix (NxN).

    Returns
    -------
    FloatArray
        Array of transition probabilities$.
    """

    # fN = 1. - np.sum(f, axis=-1)
    # f = np.concatenate((f, fN))

    # P = np.empty_like(f)
    # for i in range(f.shape[-1]):
    #     P[:, i] = f[:, i]/np.sum(f/r, axis=-1)

    denominator = np.sum(f / r, axis=-1)
    P = f / denominator[:, np.newaxis]

    return P


# %% Multicomponent Q-e

# @dataclass(frozen=True)
# class QePair:
#     Q: float
#     e: float


def convert_Qe_to_r(Qe_values: list[tuple[float, float]]
                    ) -> FloatMatrix:
    r"""Convert Q-e values to reactivity ratios.

    According to the Q-e scheme proposed by Alfrey and Price, the reactivity
    ratios of the terminal model can be estimated using the relationship:

    $$ r_{ij} = \frac{Q_i}{Q_j}\exp{\left(-e_i(e_i -e_j)\right)} $$

    where $Q_i$ and $e_i$ are monomer-specific constants, and
    $r_{ij}=k_{ii}/k_{ij}$ is the multicomponent reactivity ratio matrix.

    Reference:

    * T Alfrey, CC Price. J. Polym. Sci., 1947, 2: 101-106.

    Parameters
    ----------
    Qe_values : list[tuple[float, float]]
        List (N) of Q-e values.

    Returns
    -------
    FloatSquareMatrix
        Reactivity ratio matrix (NxN).
    """
    Q = [x[0] for x in Qe_values]
    e = [x[1] for x in Qe_values]
    N = len(Q)
    r = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            r[i, j] = Q[i]/Q[j]*exp(-e[i]*(e[i] - e[j]))
    return r
