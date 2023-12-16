# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector, FloatSquareMatrix

import numpy as np

__all__ = ['quadratic_mixing_rule']


def quadratic_mixing_rule(y: FloatVector, Q: FloatSquareMatrix) -> float:
    r"""Calculate mixture parameter using quadratic mixing rule.

    $$ Q_m = sum_i sum_j y_i y_j Q_{ij} $$

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 75.

    Parameters
    ----------
    y : FloatVector
        Composition, usually molar or mass fractions.
    Q : FloatSquareMatrix
        Matrix of pure component and interaction parameters.

    Returns
    -------
    float
        Mixture parameter, $Q_m$. Unit = [Q].
    """
    # alternative loop implementation, if Q is symmetric
    # N = y.size
    # Qm = 0.
    # for i in range(N):
    #     Qm += y[i]**2 * Q[i, i]
    # for i in range(N):
    #     for j in range(i+1, N):
    #         Qm += 2*y[i]*y[j]*Q[i, j]
    Qm = np.sum(np.outer(y, y)*Q, dtype=np.float64)
    return Qm
