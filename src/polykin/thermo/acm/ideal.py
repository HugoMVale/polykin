# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import numpy as np

from polykin.utils.types import FloatVector

from .base import SmallSpeciesActivityModel

__all__ = ['IdealSolution']


class IdealSolution(SmallSpeciesActivityModel):
    r"""[Ideal solution](https://en.wikipedia.org/wiki/Ideal_solution) model.

    This model is based on the following trivial molar excess Gibbs energy
    expression:

    $$ g^{E} = 0 $$

    Parameters
    ----------
    N : int
        Number of components.
    name : str
        Name.
    """

    def __init__(self,
                 N: int,
                 name: str = ''
                 ) -> None:

        super().__init__(N, name)

    def gE(self, T: float, x: FloatVector) -> float:
        return 0.

    def gamma(self, T: float, x: FloatVector) -> FloatVector:
        return np.ones(self.N)
