# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

import itertools
from typing import Optional, Union

import numpy as np
from numpy import exp
from scipy.integrate import solve_ivp

from polykin.utils.exceptions import ODESolverError, ShapeError
from polykin.utils.math import eps
from polykin.utils.types import (FloatArray, FloatArrayLike, FloatMatrix,
                                 FloatSquareMatrix, FloatVector,
                                 FloatVectorLike, IntArrayLike)

__all__ = ['convert_Qe_to_r',
           'inst_copolymer_ternary',
           'inst_copolymer_multi',
           'radical_fractions_ternary',
           'radical_fractions_multi',
           'monomer_drift_multi',
           'sequence_multi',
           'transitions_multi',
           'tuples_multi']


def inst_copolymer_ternary(f1: Union[float, FloatArrayLike],
                           f2: Union[float, FloatArrayLike],
                           r12: float,
                           r21: float,
                           r13: float,
                           r31: float,
                           r23: float,
                           r32: float,
                           ) -> tuple[Union[float, FloatArray],
                                      Union[float, FloatArray],
                                      Union[float, FloatArray]]:
    r"""Calculate the instantaneous copolymer composition for a ternary system.

    In a ternary system, the instantaneous copolymer composition $F_i$ is
    related to the monomer composition $f_i$ by:

    \begin{aligned}
        a &= \frac{f_1}{r_{21} r_{31}} + \frac{f_2}{r_{21} r_{32}} +
             \frac{f_3}{r_{31} r_{23}} \\
        b &= f_1 + \frac{f_2}{r_{12}} + \frac{f_3}{r_{13}} \\
        c &= \frac{f_1}{r_{12} r_{31}} + \frac{f_2}{r_{12} r_{32}} +
             \frac{f_3}{r_{13} r_{32}} \\
        d &= f_2 + \frac{f_1}{r_{21}} + \frac{f_3}{r_{23}} \\
        e &= \frac{f_1}{r_{13} r_{21}} + \frac{f_2}{r_{23} r_{12}} +
             \frac{f_3}{r_{13} r_{23}} \\
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
    f1 : float | FloatArrayLike
        Molar fraction of M1.
    f2 : float | FloatArrayLike
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
    tuple[float | FloatArray, ...]
        Instantaneous terpolymer composition, $(F_1, F_2, F_3)$.

    See also
    --------
    * [`inst_copolymer_binary`](inst_copolymer_binary.md):
      specific method for binary systems.
    * [`inst_copolymer_multi`](inst_copolymer_multi.md):
      generic method for multicomponent systems.

    Examples
    --------
    >>> from polykin.copolymerization import inst_copolymer_ternary
    >>> F1, F2, F3 = inst_copolymer_ternary(f1=0.5, f2=0.3, r12=0.2, r21=2.3,
    ...                                     r13=3.0, r31=0.9, r23=0.4, r32=1.5)
    >>> print(f"F1 = {F1:.2f}; F2 = {F2:.2f}; F3 = {F3:.2f}")
    F1 = 0.32; F2 = 0.41; F3 = 0.27
    """

    f1 = np.asarray(f1)
    f2 = np.asarray(f2)

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


def inst_copolymer_multi(f: Optional[FloatVectorLike],
                         r: Optional[FloatSquareMatrix],
                         P: Optional[FloatSquareMatrix] = None
                         ) -> FloatVector:
    r"""Calculate the instantaneous copolymer composition for a multicomponent
    system.

    In a multicomponent system, the instantaneous copolymer composition $F_i$
    can be determined by solving the following set of linear algebraic
    equations:

    $$ \begin{bmatrix}
    P_{11}-1  & P_{21}   & \cdots  & P_{N1} \\
    P_{12}    & P_{22}-1 & \cdots  & P_{N2} \\
    \vdots    & \vdots   & \vdots  & \vdots \\
    1         & 1        & \cdots  & 1
    \end{bmatrix}
    \begin{bmatrix}
    F_1    \\
    F_2    \\
    \vdots \\
    F_N
    \end{bmatrix} =
    \begin{bmatrix}
    0      \\
    0      \\
    \vdots \\
    1
    \end{bmatrix} $$

    where $P_{ij}$ are the transition probabilitites, which can be computed
    from the instantaneous monomer composition and the reactivity matrix.

    **References**

    *   H. K. Frensdorff, R. Pariser; Copolymerization as a Markov Chain.
        J. Chem. Phys. 1 November 1963; 39 (9): 2303-2309.

    Parameters
    ----------
    f : FloatVectorLike (N) | None
        Vector of instantaneous monomer compositions, $f_i$.
    r : FloatSquareMatrix (N, N) | None
        Matrix of reactivity ratios, $r_{ij}=k_{ii}/k_{ij}$.
    P : FloatSquareMatrix (N, N) | None
        Matrix of transition probabilities, $P_{ij}$. If `None`, it will
        be computed internally. When calculating other quantities (e.g.,
        sequence lengths, tuples) that also depend on $P$, it is more efficient
        to precompute $P$ once and use it in all cases.

    Returns
    -------
    FloatVector (N)
        Vector of instantaneous copolymer compositions, $F_i$.

    See also
    --------
    * [`inst_copolymer_binary`](inst_copolymer_binary.md):
      specific method for binary systems.
    * [`inst_copolymer_ternary`](inst_copolymer_ternary.md):
      specific method for terpolymer systems.
    * [`monomer_drift_multi`](monomer_drift_multi.md):
      monomer composition drift.
    * [`transitions_multi`](transitions_multi.md):
      instantaneous transition probabilities.

    Examples
    --------
    >>> from polykin.copolymerization import inst_copolymer_multi
    >>> import numpy as np

    Define the reactivity ratio matrix.
    >>> r = np.ones((3, 3))
    >>> r[0, 1] = 0.2
    >>> r[1, 0] = 2.3
    >>> r[0, 2] = 3.0
    >>> r[2, 0] = 0.9
    >>> r[1, 2] = 0.4
    >>> r[2, 1] = 1.5

    Evaluate the instantaneous copolymer composition at f1=0.5, f2=0.3, f3=0.2.
    >>> f = [0.5, 0.3, 0.2]
    >>> F = inst_copolymer_multi(f, r)
    >>> F
    array([0.32138111, 0.41041608, 0.26820282])
    """

    if P is not None and (f is None and r is None):
        pass
    elif P is None and (f is not None and r is not None):
        P = transitions_multi(f, r)
    else:
        raise ValueError("Invalid combination of inputs.")

    N = P.shape[0]
    A = P.T - np.eye(N)
    A[-1, :] = 1.
    b = np.zeros(N)
    b[-1] = 1.
    F = np.linalg.solve(A, b)

    return F


def radical_fractions_ternary(f1: Union[float, FloatArrayLike],
                              f2: Union[float, FloatArrayLike],
                              k12: float,
                              k21: float,
                              k13: float,
                              k31: float,
                              k23: float,
                              k32: float,
                              ) -> tuple[Union[float, FloatArray],
                                         Union[float, FloatArray],
                                         Union[float, FloatArray]]:
    r"""Calculate the radical fractions for a ternary system.

    In a ternary system, the radical fractions $p_i$ are related to the
    monomer composition $f_i$ by:

    \begin{aligned}
    a &= k_{21}k_{31}f_1^2+k_{21}k_{32}f_1f_2+k_{23}k_{31}f_1f_3 \\
    b &= k_{12}k_{31}f_1f_2+k_{12}k_{32}f_2^2+k_{13}k_{32}f_2f_3 \\
    c &= k_{12}k_{23}f_2f_3+k_{13}k_{21}f_1f_3+k_{13}k_{23}f_3^2 \\
    p_1 &= \frac{a}{a+b+c} \\
    p_2 &= \frac{b}{a+b+c} \\
    p_3 &= \frac{c}{a+b+c}
    \end{aligned}

    where $k_{ij}$ are the cross-propagation rate coefficients. Note that the
    homo-propagation rate coefficients $k_{ii}$ do not appear in the equations.
    For this reason, radical fractions cannot be evaluated from reactivity
    ratios alone.

    **References**

    *   Hamielec, A.E., MacGregor, J.F. and Penlidis, A. (1987), Multicomponent
    free-radical polymerization in batch, semi- batch and continuous reactors.
    Makromolekulare Chemie. Macromolecular Symposia, 10-11: 521-570.

    Parameters
    ----------
    f1 : float | FloatArrayLike
        Molar fraction of M1.
    f2 : float | FloatArrayLike
        Molar fraction of M2.
    k12 : float
        Propagation rate coefficient.
    k21 : float
        Propagation rate coefficient.
    k13 : float
        Propagation rate coefficient.
    k31 : float
        Propagation rate coefficient.
    k23 : float
        Propagation rate coefficient.
    k32 : float
        Propagation rate coefficient.

    Returns
    -------
    tuple[float | FloatArray, ...]
        Radical fractions, $(p_1, p_2, p_3)$.

    See also
    --------
    * [`radical_fractions_multi`](radical_fractions_multi.md):
      generic method for multicomponent systems.

    Examples
    --------
    >>> from polykin.copolymerization import radical_fractions_ternary
    >>> p1, p2, p3 = radical_fractions_ternary(
    ...              f1=0.5, f2=0.3, k12=500., k21=50.,
    ...              k13=30., k31=200., k23=300., k32=40.)
    >>> print(f"p1 = {p1:.2f}; p2 = {p2:.2f}; p3 = {p3:.2f}")
    p1 = 0.25; p2 = 0.48; p3 = 0.27
    """

    f1 = np.asarray(f1)
    f2 = np.asarray(f2)
    f3 = 1. - (f1 + f2)

    p1 = k21*k31*f1**2 + k21*k32*f1*f2 + k23*k31*f1*f3
    p2 = k12*k31*f1*f2 + k12*k32*f2**2 + k13*k32*f2*f3
    p3 = k12*k23*f2*f3 + k13*k21*f1*f3 + k13*k23*f3**2

    denominator = p1 + p2 + p3

    p1 /= denominator
    p2 /= denominator
    p3 /= denominator

    return (p1, p2, p3)


def radical_fractions_multi(f: FloatVectorLike,
                            k: FloatSquareMatrix,
                            ) -> FloatVector:
    r"""Calculate the radical fractions for a multicomponent system.

    In a multicomponent system, the radical fractions $p_i$ can be determined
    by solving the following set of linear algebraic equations:

    $$ \sum_{j\ne i}^N k_{ij} p_i f_j = \sum_{j\ne i}^N k_{ji} p_j f_i $$

    where $k_{ij}$ are the cross-propagation rate coefficients and $f_i$ are
    the monomer compositions. Note that the homo-propagation rate coefficients
    $k_{ii}$ do not appear in the equations. For this reason, radical fractions
    cannot be evaluated from reactivity ratios alone.

    Parameters
    ----------
    f : FloatVectorLike (N)
        Vector of instantaneous monomer compositions, $f_i$.
    k : FloatSquareMatrix (N, N)
        Matrix of cross-propagation rate coefficients. The diagonal elements
        $k_{ii}$ are not used.

    Returns
    -------
    FloatVector (N)
        Vector of radical fractions, $p_i$.

    See also
    --------
    * [`radical_fractions_ternary`](radical_fractions_ternary.md):
      specific method for terpolymer systems.

    Examples
    --------
    >>> from polykin.copolymerization import radical_fractions_multi
    >>> import numpy as np

    Define the cross-propagation coefficient matrix.
    >>> k = np.zeros((3, 3))
    >>> k[0, 1] = 500.
    >>> k[1, 0] = 50.
    >>> k[0, 2] = 30.
    >>> k[2, 0] = 200.
    >>> k[1, 2] = 300.
    >>> k[2, 1] = 40.

    Evaluate the radical fractions at f1=0.5, f2=0.3, f3=0.2.
    >>> f = [0.5, 0.3, 0.2]
    >>> p = radical_fractions_multi(f, k)
    >>> p
    array([0.25012791, 0.47956341, 0.27030868])
    """
    f = np.asarray(f)

    A = k.T * f[:, np.newaxis]
    x = A.sum(axis=0)
    np.fill_diagonal(A, A.diagonal() - x)
    A[-1, :] = 1.
    b = np.zeros(A.shape[0])
    b[-1] = 1.
    p = np.linalg.solve(A, b)

    return p


def monomer_drift_multi(f0: FloatVectorLike,
                        x: FloatVectorLike,
                        r: FloatSquareMatrix,
                        atol: float = 1e-4,
                        rtol: float = 1e-4
                        ) -> FloatMatrix:
    r"""Compute the monomer composition drift for a multicomponent system.

    In a closed system, the drift in monomer composition is given by
    the solution of the following system of differential equations:

    $$ \frac{\textup{d} f_i}{\textup{d}x} = \frac{f_i - F_i}{1 - x} $$

    with initial condition $f_i(0)=f_{i,0}$, where $f_i$ and $F_i$ are,
    respectively, the instantaneous comonomer and copolymer composition of
    monomer $i$, and $x$ is the total molar monomer conversion.

    Parameters
    ----------
    f0 : FloatVectorLike (N)
        Vector of initial instantaneous comonomer compositions.
    x : FloatVectorLike (M)
        Vector of total monomer conversion values where the drift is to be
        evaluated.
    r : FloatSquareMatrix (N, N)
        Matrix of reactivity ratios, $r_{ij}=k_{ii}/k_{ij}$.
    atol : float
        Absolute tolerance of the ODE solver.
    rtol : float
        Relative tolerance of the ODE solver.

    Returns
    -------
    FloatMatrix (M, N)
        Matrix of monomer fraction of monomer $i$ at the specified total
        monomer conversions, $f_i(x_j)$.

    See also
    --------
    * [`inst_copolymer_multi`](inst_copolymer_multi.md): 
      instantaneous copolymer composition.
    * [`monomer_drift_multi`](monomer_drift_multi.md):
      specific method for binary systems.

    Examples
    --------
    >>> from polykin.copolymerization import monomer_drift_multi
    >>> import numpy as np

    Define reactivity ratio matrix.
    >>> r = np.ones((3, 3))
    >>> r[0, 1] = 0.2
    >>> r[1, 0] = 2.3
    >>> r[0, 2] = 3.0
    >>> r[2, 0] = 0.9
    >>> r[1, 2] = 0.4
    >>> r[2, 1] = 1.5

    Evaluate monomer drift.
    >>> f0 = [0.5, 0.3, 0.2]
    >>> x = [0.1, 0.5, 0.9, 0.99]
    >>> f = monomer_drift_multi(f0, x, r)
    >>> f
    array([[5.19272893e-01, 2.87851432e-01, 1.92875675e-01],
           [6.38613228e-01, 2.04334321e-01, 1.57052451e-01],
           [8.31122266e-01, 5.58847454e-03, 1.63289259e-01],
           [4.98294381e-01, 1.22646553e-07, 5.01705497e-01]])

    """

    N = len(f0)
    if r.shape != (N, N):
        raise ShapeError("Shape mismatch between `f0` and `r`.")

    def dfdx(x: float, f: FloatVector) -> FloatVector:
        F = inst_copolymer_multi(f=np.append(f, 1. - f.sum()), r=r)
        return (f - F[:-1]) / (1. - x + eps)

    sol = solve_ivp(dfdx,
                    t_span=(0., max(x)),
                    y0=f0[:-1],
                    t_eval=x,
                    atol=atol,
                    rtol=rtol,
                    method='LSODA')

    if sol.success:
        f = np.empty((len(x), N))
        f[:, :-1] = np.transpose(sol.y)
        f[:, -1] = 1. - np.sum(f[:, :-1], axis=1)
    else:
        raise ODESolverError(sol.message)

    return f


def transitions_multi(f: FloatVectorLike,
                      r: FloatSquareMatrix
                      ) -> FloatSquareMatrix:
    r"""Calculate the instantaneous transition probabilities for a
    multicomponent system.

    For a multicomponent system, the transition probabilities are given
    by:

    $$ P_{ij} = \frac{r_{ij}^{-1} f_j}
                     {\displaystyle \sum_{k=1}^{N} r_{ik}^{-1} f_k} $$

    where $f_i$ is the molar fraction of monomer $i$ and $r_{ij}=k_{ii}/k_{ij}$
    is the multicomponent reactivity ratio matrix.

    **References**

    *   NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 178.

    Parameters
    ----------
    f : FloatVectorLike (N)
        Vector of instantaneous monomer compositions, $f_i$.
    r : FloatSquareMatrix (N, N)
        Matrix of reactivity ratios, $r_{ij}=k_{ii}/k_{ij}$.

    Returns
    -------
    FloatSquareMatrix (N, N)
        Matrix of transition probabilities, $P_{ij}$.

    See also
    --------
    * [`inst_copolymer_multi`](inst_copolymer_multi.md):
      instantaneous copolymer composition.
    * [`sequence_multi`](sequence_multi.md):
      instantaneous sequence lengths.
    * [`tuples_multi`](tuples_multi.md):
      instantaneous tuple fractions.

    Examples
    --------
    >>> from polykin.copolymerization import transitions_multi
    >>> import numpy as np

    Define reactivity ratio matrix.
    >>> r = np.ones((3, 3))
    >>> r[0, 1] = 0.2
    >>> r[1, 0] = 2.3
    >>> r[0, 2] = 3.0
    >>> r[2, 0] = 0.9
    >>> r[1, 2] = 0.4
    >>> r[2, 1] = 1.5

    Evaluate transition probabilities.
    >>> f = [0.5, 0.3, 0.2]
    >>> P = transitions_multi(f, r)
    >>> P
    array([[0.24193548, 0.72580645, 0.03225806],
           [0.21367521, 0.29487179, 0.49145299],
           [0.58139535, 0.20930233, 0.20930233]])

    """
    f = np.asarray(f)

    # N = len(f)
    # P = np.empty((N, N))
    # for i in range(N):
    #     for j in range(N):
    #         P[i, j] = f[j]/r[i, j] / np.sum(f/r[i, :])
    P = (f/r) / np.sum(f/r, axis=1)[:, np.newaxis]

    return P


def sequence_multi(Pself: FloatVectorLike,
                   k: Optional[Union[int, IntArrayLike]] = None,
                   ) -> FloatArray:
    r"""Calculate the instantaneous sequence length probability or the
    number-average sequence length.

    For a multicomponent system, the probability of finding $k$ consecutive
    units of monomer $i$ in a chain is:

    $$ S_{i,k} = (1 - P_{ii})P_{ii}^{k-1} $$

    and the corresponding number-average sequence length is:

    $$ \bar{S}_i = \sum_{k=1}^{\infty} k S_{i,k} = \frac{1}{1 - P_{ii}} $$

    where $P_{ii}$ is the self-transition probability $i \rightarrow i$, which
    is a function of the monomer composition and the reactivity ratios.

    **References**

    * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
    process modeling, Wiley, 1996, p. 177.

    Parameters
    ----------
    Pself : FloatVectorLike (N)
        Vector of self-transition probabilities, $P_{ii}$, corresponding to
        the diagonal of the matrix of transition probabilities.
    k : int | IntArrayLike (M) | None
        Sequence length, i.e., number of consecutive units in a chain.
        If `None`, the number-average sequence length will be computed.

    Returns
    -------
    FloatArray (N, M)
        If `k is None`, the number-average sequence lengths, $\bar{S}_i$.
        Otherwise, the sequence probabilities, $S_{i,k}$.

    See also
    --------
    * [`transitions_multi`](transitions_multi.md):
      instantaneous transition probabilities.
    * [`tuples_multi`](tuples_multi.md):
      instantaneous tuple fractions.

    Examples
    --------
    >>> from polykin.copolymerization import sequence_multi
    >>> from polykin.copolymerization import transitions_multi
    >>> import numpy as np

    Define reactivity ratio matrix.
    >>> r = np.ones((3, 3))
    >>> r[0, 1] = 0.2
    >>> r[1, 0] = 2.3
    >>> r[0, 2] = 3.0
    >>> r[2, 0] = 0.9
    >>> r[1, 2] = 0.4
    >>> r[2, 1] = 1.5

    Evaluate self-transition probabilities.
    >>> f = [0.5, 0.3, 0.2]
    >>> Pself = transitions_multi(f, r).diagonal()
    >>> Pself
    array([0.24193548, 0.29487179, 0.20930233])

    Evaluate number-average sequence lengths for all monomers.
    >>> S = sequence_multi(Pself)
    >>> S
    array([1.31914894, 1.41818182, 1.26470588])

    Evaluate probabilities for certain sequence lengths.
    >>> S = sequence_multi(Pself, k=[1, 5])
    >>> S
    array([[0.75806452, 0.00259719],
           [0.70512821, 0.00533091],
           [0.79069767, 0.00151742]])

    """

    Pself = np.asarray(Pself)

    if k is None:
        S = 1/(1. - Pself + eps)
    else:
        if isinstance(k, (list, tuple)):
            k = np.array(k, dtype=np.int32)
        if isinstance(k, np.ndarray):
            Pself = Pself.reshape(-1, 1)
        S = (1. - Pself)*Pself**(k - 1)

    return S


def tuples_multi(P: FloatSquareMatrix,
                 n: int,
                 F: Optional[FloatVectorLike] = None
                 ) -> dict[tuple[int, ...], float]:
    r"""Calculate the instantaneous n-tuple fractions.

    For a multicomponent system, the probability of finding a specific sequence
    $ijk \cdots rs$ of repeating units is:

    $$ A_{ijk \cdots rs} = F_i P_{ij} P_{jk} \cdots P_{rs} $$

    where $F_i$ is the instantaneous copolymer composition, and $P_{ij}$ is
    the transition probability $i \rightarrow j$. Since the direction of chain
    growth does not play a role, symmetric sequences are combined under the
    sequence with lower index (e.g., $A_{112} \leftarrow A_{112} + A_{211}$).

    **References**

    *   NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 179.

    Parameters
    ----------
    P : FloatSquareMatrix (N, N)
        Matrix of transition probabilities, $P_{ij}$.
    n : int
        Tuple length, e.g. monads (1), diads (2), triads (3), etc.
    F : FloatVectorLike (N) | None
        Vector of instantaneous copolymer composition, $F_i$. If `None`,
        the value will be computed internally. When calculating tuples of
        various lengths, it is more efficient to precompute $F$ and use it in
        all tuple cases.

    Returns
    -------
    dict[tuple[int, ...], float]
        Tuple of molar fractions.

    See also
    --------
    * [`sequence_multi`](sequence_multi.md):
      instantaneous sequence lengths.
    * [`transitions_multi`](transitions_multi.md):
      instantaneous transition probabilities.

    Examples
    --------
    >>> from polykin.copolymerization import tuples_multi
    >>> from polykin.copolymerization import transitions_multi
    >>> import numpy as np

    Define reactivity ratio matrix.
    >>> r = np.ones((3, 3))
    >>> r[0, 1] = 0.2
    >>> r[1, 0] = 2.3
    >>> r[0, 2] = 3.0
    >>> r[2, 0] = 0.9
    >>> r[1, 2] = 0.4
    >>> r[2, 1] = 1.5

    Evaluate transition probabilities.
    >>> f = [0.5, 0.3, 0.2]
    >>> P = transitions_multi(f, r)

    Evaluate triad fractions.
    >>> A = tuples_multi(P, 3)
    >>> A[(0, 0, 0)]
    0.018811329044450834
    >>> A[(1, 0, 1)]
    0.06365013630778116

    """

    # Compute F if not given
    if F is None:
        F = inst_copolymer_multi(f=None, r=None, P=P)

    # Generate all tuple combinations
    N = P.shape[0]
    indexes = list(itertools.product(range(N), repeat=n))

    result = {}
    for idx in indexes:
        # Compute tuple probability
        P_product = 1.
        for j in range(n-1):
            P_product *= P[idx[j], idx[j+1]]
        A = F[idx[0]]*P_product
        # Add probability to dict, but combine symmetric tuples: 12 <- 12 + 21
        reversed_idx = idx[::-1]
        if reversed_idx in result.keys():
            result[reversed_idx] += A
        else:
            result[idx] = A

    return result


def convert_Qe_to_r(Qe_values: list[tuple[float, float]]
                    ) -> FloatSquareMatrix:
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
    FloatSquareMatrix (N, N)
        Reactivity ratio matrix.

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
