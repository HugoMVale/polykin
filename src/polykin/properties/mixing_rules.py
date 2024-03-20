# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from typing import Optional

from numpy import dot, sqrt
from scipy.constants import R

from polykin.utils.types import FloatSquareMatrix, FloatVector

__all__ = ['quadratic_mixing',
           'geometric_interaction_mixing',
           'pseudocritical_properties']

# %% Mixing rules


def quadratic_mixing(y: FloatVector,
                     Q: FloatSquareMatrix
                     ) -> float:
    r"""Calculate a mixture parameter using a quadratic mixing rule.

    $$ Q_m = \sum_i \sum_j y_i y_j Q_{ij} $$

    !!! note

        For the sake of generality, no assumptions are made regarding the
        symmetry of $Q_{ij}$. The _full_ matrix must be supplied and will be
        used as given.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
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
    return dot(y, dot(y, Q))


def geometric_interaction_mixing(y: FloatVector,
                                 Q: FloatVector,
                                 k: Optional[FloatSquareMatrix] = None
                                 ) -> float:
    r"""Calculate a mixture parameter using a geometric average with
    interaction.

    $$ Q_m = \sum_i \sum_j y_i y_j (Q_i Q_j)^{1/2}(1-k_{ij}) $$

    with $k_{ii}=0$ and $k_{i,j}=k_{j,i}$.

    !!! note

        Only the entries above the main diagonal of $k_{i,j}$ are used.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
        4th edition, 1986, p. 75.

    Parameters
    ----------
    y : FloatVector
        Composition, usually molar or mass fractions.
    Q : FloatVector
        Pure component property.
    k : FloatSquareMatrix | None
        Binary interaction parameter matrix.

    Returns
    -------
    float
        Mixture parameter, $Q_m$. Unit = [Q].
    """
    if k is None:
        Qm = (dot(y, sqrt(Q)))**2
    else:
        N = y.size
        Qm = 0.
        for i in range(N):
            Qm += y[i]**2 * Q[i]
            for j in range(i+1, N):
                Qm += 2*y[i]*y[j]*sqrt(Q[i]*Q[j])*(1. - k[i, j])
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
        v_{cm} &= \sum_i y_i \frac{Z_{ci} R T_{ci}}{P_{ci}} \\
        P_{cm} &= \frac{Z_{cm} R T_{cm}}{v_{cm}} \\
        \omega_{cm} &= \sum_i y_i \omega_{ci}
    \end{aligned} $$

    where the meaning of the parameters is as defined below.

    **References**

    *   RC Reid, JM Prausniz, and BE Poling. The properties of gases & liquids
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
        $(T_{cm}, P_{cm}, v_{cm}, Z_{cm}, \omega_{cm})$.
    """

    Tc_mix = dot(y, Tc)
    Zc_mix = dot(y, Zc)
    vc = Zc*R*Tc/Pc
    vc_mix = dot(y, vc)
    Pc_mix = R*Zc_mix*Tc_mix/vc_mix

    if w is not None:
        w_mix = dot(y, w)
    else:
        w_mix = -1e99

    return (Tc_mix, Pc_mix, vc_mix, Zc_mix, w_mix)
