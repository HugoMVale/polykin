# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np

from polykin.utils.types import FloatVector

from .base import ActivityCoefficientModel

__all__ = ['IdealSolution']


class IdealSolution(ActivityCoefficientModel):
    r"""Ideal solution model."""

    def gE(self, x: FloatVector, T: float) -> float:
        r"""Molar excess Gibbs energy, $g^{E}$.

        $$ g^{E} = 0 $$

        Parameters
        ----------
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.
        T : float
            Temperature. Unit = K.

        Returns
        -------
        float
            Excess Gibbs energy. Unit = J/mol.
        """
        return 0.

    def gamma(self, x: FloatVector, T: float) -> FloatVector:
        r"""Activity coefficients, $\gamma_i$.

        $$ \gamma_i = 1 $$

        Parameters
        ----------
        x : FloatVector
            Mole fractions of all components. Unit = mol/mol.
        T : float
            Temperature. Unit = K.

        Returns
        -------
        FloatVector
            Activity coefficients.
        """
        return np.ones_like(x)
