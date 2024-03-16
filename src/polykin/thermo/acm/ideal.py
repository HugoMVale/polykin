# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np

from polykin.utils.types import FloatVector

from .base import ACM

__all__ = ['IdealSolution']


class IdealSolution(ACM):
    r"""[Ideal solution](https://en.wikipedia.org/wiki/Ideal_solution) model.

    This model is based on the following trivial molar excess Gibbs energy
    expression:

    $$ g^{E} = 0 $$

    """

    def __init__(self) -> None:
        """Construct `IdealSolution`."""

    def gE(self, T: float, x: FloatVector) -> float:
        r"""Molar excess Gibbs energy, $g^{E}$.

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        float
            Molar excess Gibbs energy. Unit = J/mol.
        """
        return 0.

    def gamma(self, T: float, x: FloatVector) -> FloatVector:
        r"""Activity coefficients, $\gamma_i$.

        $$ \gamma_i = 1 $$

        Parameters
        ----------
        T : float
            Temperature. Unit = K.
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.

        Returns
        -------
        FloatVector
            Activity coefficients of all components.
        """
        return np.ones_like(x)
