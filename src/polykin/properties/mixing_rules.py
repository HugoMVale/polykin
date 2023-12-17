# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector, FloatSquareMatrix

import numpy as np
from scipy.constants import R
from typing import Optional

__all__ = ['quadratic_mixing_rule',
           'pseudocritical_properties']

# %% Mixing rules


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


# %% Pseudocritical properties


def pseudocritical_properties(y: FloatVector,
                              Tc: FloatVector,
                              Pc: FloatVector,
                              Zc: FloatVector,
                              w: Optional[FloatVector] = None
                              ) -> tuple[float, float, float, float, float]:
    r"""Calculate the pseudocritial properties of a mixture to use in
    corresponding states correlations.

    $$ \begin{aligned}
        T_{cm} &= \sum_i y_i T_{ci} \\
        Z_{cm} &= \sum_i y_i Z_{ci} \\
        V_{cm} &= \sum_i y_i \frac{Z_{ci} R T_{ci}}{P_{ci}} \\
        P_{cm} &= \frac{Z_{cm} R T_{cm}}{V_{cm}} \\
        \omega_{cm} &= \sum_i y_i \omega_{ci}
    \end{aligned} $$

    where the meaning of the parameters is as defined below.

    Reference:

    * RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
    4th edition, 1986, p. 76-77.

    Parameters
    ----------
    y : FloatVector
        Mole fractions of all components. Unit = mol/mol.
    Tc : FloatVector
        Critical temperatures of all components. Unit = K.
    Pc : FloatVector
        Critical pressures of all components. Unit = Pa.
    Zc : FloatVector
        Critical compressibility factors of all components.
    w : FloatVector | None
        Acentric factors of all components.

    Returns
    -------
    tuple[float, float, float, float, float]
        Tuple of pseudocritial properties,
        $(T_{cm}, P_{cm}, V_{cm}, Z_{cm}, \omega_{cm})$.
    """

    Tc_mix = np.dot(y, Tc)
    Zc_mix = np.dot(y, Zc)
    Vc = Zc*R*Tc/Pc
    Vc_mix = np.dot(y, Vc)
    Pc_mix = R*Zc_mix*Tc_mix/Vc_mix

    if w is not None:
        w_mix = np.dot(y, w)
    else:
        w_mix = -1e99

    return (Tc_mix, Pc_mix, Vc_mix, Zc_mix, w_mix)
