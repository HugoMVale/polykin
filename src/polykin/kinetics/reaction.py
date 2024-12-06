# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union, Sequence

import numpy as np

from polykin.kinetics import Arrhenius
from polykin.utils.types import FloatVector

__all__ = ['Substance',
           'Variable',
           'Distribution',
           'Propagation',
           'ReactionSystem']


class Species():

    idx: int
    name: str
    mass: float
    note: str


class Scalar(Species):
    pass


class Substance(Scalar):

    def __init__(self,
                 *,
                 name: str,
                 mass: float = 0.,
                 note: str = ''
                 ) -> None:
        self.name = name
        self.mass = mass
        self.note = note


class Variable(Scalar):

    def __init__(self,
                 *,
                 name: str,
                 note: str = ''
                 ) -> None:
        self.name = name
        self.mass = 0.
        self.note = note


class Distribution(Species):

    mass_fragment: float

    def __init__(self,
                 *,
                 name: str,
                 mass: float = 0.,
                 mass_fragment: float = 0.,
                 note: str = ''
                 ) -> None:
        self.name = name
        self.mass = mass
        self.mass_fragment = mass_fragment
        self.note = note

# %%


class Reaction(ABC):

    active: bool
    reactants: list[Species]
    products: list[Species]
    k: Arrhenius
    DH: float
    nr: int
    np: int
    name: str
    note: str
    stoich_reactants: FloatVector
    stoich_products: FloatVector

    def __init__(self,
                 reactants: list[Species],
                 products: list[Species],
                 k: Arrhenius,
                 DH: float,
                 name: str,
                 note: str,
                 ) -> None:
        self.active = True
        self.reactants = reactants
        self.products = products
        self.k = k
        self.nr = len(reactants)
        self.np = len(products)
        self.name = name
        self.note = note
        self.DH = DH

    @abstractmethod
    def eval(self,
             T: float,
             x: FloatVector,
             rx: FloatVector,
             ) -> None:
        pass

    def rate(self, C: FloatVector, T: float) -> float:
        """Calculate the rate of reaction."""
        rate = self.k(T)
        for reactant in self.reactants:
            rate *= C[reactant.idx]
        return rate  # type: ignore

    def __add__(self,
                other: Reaction
                ) -> ReactionSystem:
        if isinstance(other, Reaction):
            return ReactionSystem([self, other])
        else:
            return NotImplemented


class ReactionSystem():

    dim: int
    scalars: list[Substance]
    distributions: list[Distribution]
    indexes: dict

    def __init__(self,
                 reactions: list[Reaction]
                 ) -> None:

        # Check for repeated reactions
        if len(reactions) != len(set(reactions)):
            raise ValueError("`ReactionSystem` contains repeated `Reactions`.")
        else:
            self.reactions = reactions

        # Collect species
        scalars = []
        distributions = []
        for reaction in reactions:
            for s in reaction.species():
                if isinstance(s, Scalar):
                    scalars.append(s)
                elif isinstance(s, Distribution):
                    distributions.append(s)
                else:
                    raise ValueError("This should not happen!")
        self.scalars = list(set(scalars))
        self.distributions = list(set(distributions))
        self.dim = len(scalars) + 3*len(distributions)

        # assign idx for each species

        return

    # @property
    # def reactions(self)->list[Reaction]:
    #     return self._reactions

    # @reactions.setter
    # def reactions(self, reactions):
    #     self.__reactions = reactions

    def __add__(self,
                other: Union[Reaction, ReactionSystem]
                ) -> ReactionSystem:
        if isinstance(other, Reaction):
            self.reactions.append(other)
            return self
        elif isinstance(other, ReactionSystem):
            return ReactionSystem(self.reactions + other.reactions)
        else:
            return NotImplemented

    def rates(self,
              x: FloatVector,
              T: float
              ) -> FloatVector:

        # check shape x

        # rx could also be given for update
        rx = np.zeros(self.dim)

        for reaction in self.reactions:
            reaction.eval(T, x, rx)

        return rx


class Propagation(Reaction):
    """Propagation.

    P(n) + M --> P(n+1) + (A)
    """

    M: Scalar
    P: Distribution
    A: Optional[Scalar]

    def __init__(self,
                 *,
                 M: Scalar,
                 P: Distribution,
                 k: Arrhenius,
                 A: Optional[Scalar] = None,
                 DH: float = 0,
                 name: str = '',
                 note: str = ''
                 ) -> None:
        self.M = M
        self.P = P
        self.A = A
        products: list[Species] = [P]
        if A is not None:
            products.append(A)
        super().__init__([M, P], products, k, DH, name, note)

    def eval(self,
             T: float,
             x: FloatVector,
             rx: FloatVector,
             ) -> None:

        iM = self.M.idx
        iP = self.P.idx
        M = x[iM]
        P0 = x[iP]
        P1 = x[iP + 1]
        # P2 = x[iP + 2]

        k = self.k(T)

        r = k*M*P0
        rx[iM] += -r
        # rx[iP0] += 0.
        rx[iP + 1] += r
        rx[iP + 2] += k*M*(2*P1 + P0)

        if self.A:
            rx[self.A.idx] += r

        # dQ = r*self.DH
        return


class Termination(Reaction):
    """Termination.

    P(n) + Q(m) --> R(n+m) + (A)
    """

    P: Distribution
    Q: Distribution
    R: Distribution
    A: Optional[Scalar]

    def __init__(self,
                 *,
                 P: Distribution,
                 Q: Distribution,
                 R: Distribution,
                 k: Arrhenius,
                 A: Optional[Scalar] = None,
                 DH: float = 0,
                 name: str = '',
                 note: str = ''
                 ) -> None:
        self.P = P
        self.Q = Q
        self.R = R
        self.A = A
        products: list[Species] = [R]
        if A is not None:
            products.append(A)
        super().__init__([P, Q], products, k, DH, name, note)

# %% aux functions


def exclude_None(a: Iterable):
    return (item for item in a if item is not None)

# quem é que procura os indices? Reaction or ReactionNetwork?
