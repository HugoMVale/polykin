# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

import numpy as np
from numpy import dot, exp

from polykin.constants import gas_constant as R
from polykin.utils.tools import check_shape
from polykin.utils.typing import FloatVector, FloatVectorLike, override

from .base import MolecularACM

__all__ = ["ScatchardHildebrand", "ScatchardHildebrand_gamma"]


class ScatchardHildebrand(MolecularACM):
    r"""Scatchard-Hildebrand multicomponent activity coefficient model.

    This model is based on the following molar excess Gibbs energy expression:

    $$ g^{E} = \left( \sum_i x_i v_i \right)
               \left( \sum_i \phi_i (\delta_i - \bar{\delta})^2 \right) $$

    where $x_i$ are the mole fractions, $\delta_{i}$ are the solubility parameters, and
    $v_i$ are the molar volumes. Moreover, $\phi_i$ are the volume fractions defined as:

    $$ \phi_i = \frac{x_i v_i}{\sum_j x_j v_j} $$

    and $\bar{\delta}$ is the average solubility parameter:

    $$ \bar{\delta} = \sum_i \phi_i \delta_i $$

    !!! note

        The solubility parameters and the molar volumes can be expressed in any length
        units, as long as they are consistent with each other.

    **References**

    *   Prausnitz, J. M.; Lichtenthaler, R. N.; de Azevedo, E. G. Molecular Thermodynamics
        of Fluid-Phase Equilibria, 3rd ed.; Prentice Hall, 1999, p. 325.

    Parameters
    ----------
    N : int
        Number of components.
    delta : FloatVectorLike (N)
        Solubility parameters of all components [(J/L³)^(1/2)].
    v : FloatVectorLike (N)
        Molar volumes of all components [L³/mol].
    name : str
        Name of the model instance.

    See Also
    --------
    * [`ScatchardHildebrand_gamma`](ScatchardHildebrand_gamma.md):
      Related activity coefficient method.
    """

    delta: FloatVector
    v: FloatVector

    def __init__(
        self,
        N: int,
        delta: FloatVectorLike,
        v: FloatVectorLike,
        name: str = "",
    ) -> None:

        delta = np.asarray(delta, dtype=float)
        v = np.asarray(v, dtype=float)

        check_shape(delta, (N,), "delta")
        check_shape(v, (N,), "v")

        super().__init__(N, name)
        self.delta = delta
        self.v = v

    def gE(self, T: float, x: FloatVector) -> float:

        v = self.v
        delta = self.delta

        phi = x * v
        vm = phi.sum()
        phi /= vm

        δm = dot(phi, delta)

        return vm * dot(phi, (delta - δm) ** 2)

    @override
    def gamma(self, T: float, x: FloatVector) -> FloatVector:
        return ScatchardHildebrand_gamma(T, x, self.delta, self.v)


def ScatchardHildebrand_gamma(
    T: float,
    x: FloatVector,
    delta: FloatVector,
    v: FloatVector,
) -> FloatVector:
    r"""Calculate the activity coefficients of a multicomponent mixture according to the
    Scatchard-Hildebrand model.

    $$ \ln{\gamma_i} = \frac{v_i}{R T} (\delta_i - \bar{\delta})^2 $$

    with:

    \begin{aligned}
    \phi_i       &= \frac{x_i v_i}{\sum_j x_j v_j}
    \bar{\delta} &= \sum_j \phi_j \delta_j \\
    \end{aligned}

    where $x_i$ are the mole fractions, $\delta_{i}$ are the solubility parameters, and
    $v_i$ are the molar volumes.

    **References**

    *   Prausnitz, J. M.; Lichtenthaler, R. N.; de Azevedo, E. G. Molecular Thermodynamics
        of Fluid-Phase Equilibria, 3rd ed.; Prentice Hall, 1999, p. 325.

    Parameters
    ----------
    T : float
        Temperature [K].
    x : FloatVector (N)
        Mole fractions of all components [mol/mol].
    delta : FloatVector (N)
        Solubility parameters of all components [(J/L³)^(1/2)].
    v : FloatVector (N)
        Molar volumes of all components [L³/mol].

    Returns
    -------
    FloatVector (N)
        Activity coefficients of all components.

    See Also
    --------
    * [`ScatchardHildebrand`](ScatchardHildebrand.md): Related class.
    """
    phi = x * v
    phi /= phi.sum()
    δm = dot(phi, delta)
    return exp(v / (R * T) * (delta - δm) ** 2)
