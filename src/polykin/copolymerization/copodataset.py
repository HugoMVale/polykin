# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from abc import ABC
from dataclasses import dataclass
from typing import Any, Literal, Union

from polykin.utils.math import convert_FloatOrVectorLike_to_FloatOrVector
from polykin.utils.tools import check_shapes
from polykin.utils.types import FloatVector, FloatVectorLike

# from dataclasses import dataclass


__all__ = ['MayoDataset',
           'DriftDataset',
           'kpDataset',
           'CopoDataset_Ff',
           'CopoDataset_fx',
           'CopoDataset_Fx']


class CopoDataset(ABC):

    varmap: dict[str, str]

    def __init__(self,
                 M1: str,
                 M2: str,
                 x: FloatVectorLike,
                 y: FloatVectorLike,
                 sigma_x: Union[float, FloatVectorLike],
                 sigma_y: Union[float, FloatVectorLike],
                 weight: float,
                 T: float,
                 Tunit: Literal['C', 'K'],
                 name: str,
                 source: str,
                 ) -> None:
        """Construct `CopoDataset` with the given parameters."""

        if not M1 or not M2 or M1.lower() == M2.lower():
            raise ValueError(
                "`M1` and `M2` must be non-empty and different.")
        else:
            self.M1 = M1
            self.M2 = M2

        _x, _y, _sigma_x, _sigma_y = \
            convert_FloatOrVectorLike_to_FloatOrVector(
                [x, y, sigma_x, sigma_y])

        check_shapes([_x, _y], [_sigma_x, _sigma_y])

        self.x = _x
        self.y = _y
        self.sigma_x = _sigma_x
        self.sigma_y = _sigma_y
        self.weight = weight
        self.T = T
        self.Tunit = Tunit
        self.name = name
        self.source = source

    def getvar(self,
               varname: str,
               M: str = ''
               ) -> Any:
        x = getattr(self, self.varmap[varname])

        if M:
            if M == self.M1:
                pass
            elif M == self.M2:
                x = 1 - x
            else:
                raise ValueError
        return x


class MayoDataset(CopoDataset):
    r"""Dataset of binary instantaneous copolymerization data.

    Container for binary instantaneous copolymerization data, $F_1$ vs $f_1$,
    as usually obtained from low-conversion experiments.

    Parameters
    ----------
    M1 : str
        Name of M1.
    M2 : str
        Name of M2.
    f1 : FloatVectorLike
        Vector of monomer composition, $f_1$.
    F1 : FloatVectorLike
        Vector of instantaneous copolymer composition, $F_1$.
    sigma_f1: float | FloatVectorLike
        Absolute standard deviation of $f_1$
        ($\sigma_{f_1} \equiv \sigma_{f_2}$).
    sigma_F1: float | FloatVectorLike
        Absolute standard deviation of $F_1$
        ($\sigma_{F_1} \equiv \sigma_{F_2}$).
    weight: float
        Relative weight of dataset for fitting.
    T : float
        Temperature. Unit = `Tunit`.
    Tunit : Literal['C', 'K']
        Temperature unit.
    name: str
        Name of dataset.
    source: str
        Source of dataset.
    """
    varmap = {'f': 'x', 'F': 'y'}

    def __init__(self,
                 M1: str,
                 M2: str,
                 f1: FloatVectorLike,
                 F1: FloatVectorLike,
                 sigma_f1: Union[float, FloatVectorLike] = 1e-2,
                 sigma_F1: Union[float, FloatVectorLike] = 5e-2,
                 weight: float = 1,
                 T: float = 298.,
                 Tunit: Literal['C', 'K'] = 'K',
                 name: str = '',
                 source: str = '',
                 ) -> None:
        """Construct `MayoDataset` with the given parameters."""
        super().__init__(M1, M2, f1, F1, sigma_f1, sigma_F1, weight, T, Tunit,
                         name, source)


class DriftDataset(CopoDataset):
    r"""Dataset of binary monomer drift copolymerization data.

    Container for binary monomer drift copolymerization data, $f_1$ vs $x$.

    Parameters
    ----------
    M1 : str
        Name of M1.
    M2 : str
        Name of M2.
    x : FloatVectorLike
        Vector of total molar conversion, $x$.
    f1 : FloatVectorLike
        Vector of monomer composition, $f_1$.
    sigma_x: float | FloatVectorLike
        Absolute standard deviation of $x$.
    sigma_f1: float | FloatVectorLike
        Absolute standard deviation of $f_1$.
    weight: float
        Relative weight of dataset for fitting.
    T : float
        Temperature. Unit = `Tunit`.
    Tunit : Literal['C', 'K']
        Temperature unit.
    name: str
        Name of dataset.
    source: str
        Source of dataset.
    """

    varmap = {'x': 'x', 'f': 'y'}

    def __init__(self,
                 M1: str,
                 M2: str,
                 x: FloatVectorLike,
                 f1: FloatVectorLike,
                 sigma_x: Union[float, FloatVectorLike] = 5e-2,
                 sigma_f1: Union[float, FloatVectorLike] = 5e-2,
                 weight: float = 1,
                 T: float = 298.,
                 Tunit: Literal['C', 'K'] = 'K',
                 name: str = '',
                 source: str = '',
                 ) -> None:
        """Construct `DriftDataset` with the given parameters."""
        super().__init__(M1, M2, x, f1, sigma_x, sigma_f1, weight, T, Tunit,
                         name, source)


class kpDataset(CopoDataset):
    r"""Dataset of average propagation rate coefficient data.

    Container for average propagation rate coefficient as a function of monomer
    composition, $k_p$ vs $f_1$.

    Parameters
    ----------
    M1 : str
        Name of M1.
    M2 : str
        Name of M2.
    f1 : FloatVectorLike
        Vector of monomer composition, $f_1$.
    kp : FloatVectorLike
        Vector of average propagation rate coefficient, $\bar{k}_p$.
        Unit = L/(mol·s).
    sigma_f1: float | FloatVectorLike
        Absolute standard deviation of $f_1$.
    sigma_kp: float | FloatVectorLike
        Absolute standard deviation of $\bar{k}_p$. Unit = L/(mol·s).
    weight: float
        Relative weight of dataset for fitting.
    T : float
        Temperature. Unit = `Tunit`.
    Tunit : Literal['C', 'K']
        Temperature unit.
    name: str
        Name of dataset.
    source: str
        Source of dataset.
    """

    varmap = {'f': 'x', 'kp': 'y'}

    def __init__(self,
                 M1: str,
                 M2: str,
                 f1: FloatVectorLike,
                 kp: FloatVectorLike,
                 sigma_f1: Union[float, FloatVectorLike] = 5e-2,
                 sigma_kp: Union[float, FloatVectorLike] = 1e2,
                 weight: float = 1,
                 T: float = 298.,
                 Tunit: Literal['C', 'K'] = 'K',
                 name: str = '',
                 source: str = '',
                 ) -> None:
        """Construct `DriftDataset` with the given parameters."""
        super().__init__(M1, M2, f1, kp, sigma_f1, sigma_kp, weight, T, Tunit,
                         name, source)


@dataclass(frozen=True)
class CopoDataset_Ff():
    """Dataclass for instantaneous copolymerization data of the form F(f)."""
    name: str
    f1: FloatVector
    F1: FloatVector
    scale_f1: Union[FloatVector, float] = 1.0
    scale_F1: Union[FloatVector, float] = 1.0
    weight: float = 1.0


@dataclass(frozen=True)
class CopoDataset_fx():
    """Dataclass for drift copolymerization data of the form f1(x)."""
    name: str
    f10: float
    x: FloatVector
    f1: FloatVector
    # scale_x: Union[FloatVector, float] = 1.0
    scale_f1: Union[FloatVector, float] = 1.0
    weight: float = 1.0


@dataclass(frozen=True)
class CopoDataset_Fx():
    """Dataclass for drift copolymerization data of the form F1(x)."""
    name: str
    f10: float
    x: FloatVector
    F1: FloatVector
    # scale_x: Union[FloatVector, float] = 1.0
    scale_F1: Union[FloatVector, float] = 1.0
    weight: float = 1.0
