# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np

from polykin.utils.types import FloatVector

from .base import ActivityCoefficientModel

__all__ = ['IdealSolution']


class IdealSolution(ActivityCoefficientModel):
    r"""Ideal solution model."""

    def gE(self, T: float, x: FloatVector) -> float:
        r"""Molar excess Gibbs energy, $g^{E}$.

        $$ g^{E} = 0 $$

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
            Activity coefficients.
        """
        return np.ones_like(x)
