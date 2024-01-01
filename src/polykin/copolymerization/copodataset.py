# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from abc import ABC
from typing import Any, Literal

from polykin.math import convert_FloatOrVectorLike_to_FloatOrVector
from polykin.types import FloatOrVectorLike, FloatVectorLike
from polykin.utils import check_shapes

# from dataclasses import dataclass


__all__ = ['MayoDataset',
           'DriftDataset',
           'kpDataset']

# %% Dataclasses


# @dataclass(frozen=True)
# class InstantaneousCopoData():
#     r"""Binary instantaneous copolymerization dataclass.

#     Container for binary instantaneous copolymerization data, $F_1$ vs $f_1$,
#     as usually obtained from low-conversion experiments.

#     Parameters
#     ----------
#     M1 : str
#         Name of M1.
#     M2 : str
#         Name of M2.
#     f1 : FloatVectorLike
#         Vector of monomer molar composition, $f_1$.
#     F1 : FloatVectorLike
#         Vector of copolymer molar composition, $F_1$.
#     sigma_f: float | FloatVectorLike
#         Absolute standard deviation of $f_i$
#         ($\sigma_{f_1} \equiv \sigma_{f_2}$).
#     sigma_F: float | FloatVectorLike
#         Absolute standard deviation of $F_i$
#         ($\sigma_{F_1} \equiv \sigma_{F_2}$).
#     name: str
#         Name of dataset.
#     reference: str
#         Reference of dataset.

#     """
#     M1: str
#     M2: str
#     f1: FloatVectorLike
#     F1: FloatVectorLike
#     sigma_f: FloatOrVectorLike = 1e-3
#     sigma_F: FloatOrVectorLike = 5e-2
#     name: str = ''
#     reference: str = ''

#     def __post_init__(self):
#         if not self.M1 or not self.M2 or self.M1.lower() == self.M2.lower():
#             raise ValueError(
#                 "`M1` and `M2` must be non-empty and different.")
#         for attr in ['f1', 'F1', 'sigma_f', 'sigma_F']:
#             x = getattr(self, attr)
#             if isinstance(x, list):
#                 x = np.array(x, dtype=np.float64)
#                 object.__setattr__(self, attr, x)
#             utils.check_bounds(x, 0., 1., attr)
#         if len(self.f1) != len(self.F1):
#             raise ShapeError(
#                 "`f1` and `F1` must be vectors of the same length.")
#         for attr in ['sigma_f', 'sigma_F']:
#             x = getattr(self, attr)
#             if isinstance(x, np.ndarray) and (len(x) != len(self.f1)):
#                 raise ShapeError(
#                     f"`{attr}` must have the same length as `f1` and `F1`.")

# %% CopoDataset : dataclass version


# @dataclass(frozen=True)
# class MayoDataset():
#     r"""Dataset of binary instantaneous copolymerization data.

#     Container for binary instantaneous copolymerization data, $F_1$ vs $f_1$,
#     as usually obtained from low-conversion experiments.

#     Parameters
#     ----------
#     M1 : str
#         Name of M1.
#     M2 : str
#         Name of M2.
#     f1 : FloatVectorLike
#         Vector of monomer composition, $f_1$.
#     F1 : FloatVectorLike
#         Vector of instantaneous copolymer composition, $F_1$.
#     sigma_f1: float | FloatVectorLike
#         Absolute standard deviation of $f_1$
#         ($\sigma_{f_1} \equiv \sigma_{f_2}$).
#     sigma_F1: float | FloatVectorLike
#         Absolute standard deviation of $F_1$
#         ($\sigma_{F_1} \equiv \sigma_{F_2}$).
#     weight: float
#         Relative weight of dataset for fitting.
#     T : float
#         Temperature. Unit = `Tunit`.
#     Tunit : Literal['C', 'K']
#         Temperature unit.
#     name: str
#         Name of dataset.
#     source: str
#         Source of dataset.
#     """
#     M1: str
#     M2: str
#     f1: FloatVectorLike
#     F1: FloatVectorLike
#     sigma_f1: FloatOrVectorLike = 1e-3
#     sigma_F1: FloatOrVectorLike = 5e-2
#     weight: float = 1
#     T: float = 298.
#     Tunit: Literal['C', 'K'] = 'K'
#     name: str = ''
#     source: str = ''


# class DriftDataset():
#     r"""Dataset of binary monomer drift copolymerization data.

#     Container for binary monomer drift copolymerization data, $f_1$ vs $x$.

#     Parameters
#     ----------
#     M1 : str
#         Name of M1.
#     M2 : str
#         Name of M2.
#     x : FloatVectorLike
#         Vector of total molar conversion, $x$.
#     f1 : FloatVectorLike
#         Vector of monomer composition, $f_1$.
#     sigma_x: float | FloatVectorLike
#         Absolute standard deviation of $x$.
#     sigma_f1: float | FloatVectorLike
#         Absolute standard deviation of $f_1$.
#     weight: float
#         Relative weight of dataset for fitting.
#     T : float
#         Temperature. Unit = `Tunit`.
#     Tunit : Literal['C', 'K']
#         Temperature unit.
#     name: str
#         Name of dataset.
#     source: str
#         Source of dataset.
#     """
#     M1: str
#     M2: str
#     x: FloatVectorLike
#     f1: FloatVectorLike
#     sigma_x: FloatOrVectorLike = 5e-2
#     sigma_f1: FloatOrVectorLike = 5e-2
#     weight: float = 1
#     T: float = 298.
#     Tunit: Literal['C', 'K'] = 'K'
#     name: str = ''
#     source: str = ''


# class kpDataset():
#     r"""Dataset of average propagation rate coefficient data.

#     Container for average propagation rate coefficient as a function of monomer
#     composition, $k_p$ vs $f_1$.

#     Parameters
#     ----------
#     M1 : str
#         Name of M1.
#     M2 : str
#         Name of M2.
#     f1 : FloatVectorLike
#         Vector of monomer composition, $f_1$.
#     kp : FloatVectorLike
#         Vector of average propagation rate coefficient, $\bar{k}_p$.
#         Unit = L/(mol·s).
#     sigma_f1: float | FloatVectorLike
#         Absolute standard deviation of $f_1$.
#     sigma_kp: float | FloatVectorLike
#         Absolute standard deviation of $\bar{k}_p$. Unit = L/(mol·s).
#     weight: float
#         Relative weight of dataset for fitting.
#     T : float
#         Temperature. Unit = `Tunit`.
#     Tunit : Literal['C', 'K']
#         Temperature unit.
#     name: str
#         Name of dataset.
#     source: str
#         Source of dataset.
#     """
#     M1: str
#     M2: str
#     f1: FloatVectorLike
#     kp: FloatVectorLike
#     sigma_f1: FloatOrVectorLike = 5e-2
#     sigma_kp: FloatOrVectorLike = 1e2
#     weight: float = 1
#     T: float = 298.
#     Tunit: Literal['C', 'K'] = 'K'
#     name: str = ''
#     source: str = ''

# %% Class version


class CopoDataset(ABC):

    varmap: dict[str, str]

    def __init__(self,
                 M1: str,
                 M2: str,
                 x: FloatVectorLike,
                 y: FloatVectorLike,
                 sigma_x: FloatOrVectorLike,
                 sigma_y: FloatOrVectorLike,
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
                 sigma_f1: FloatOrVectorLike = 1e-3,
                 sigma_F1: FloatOrVectorLike = 5e-2,
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
                 sigma_x: FloatOrVectorLike = 5e-2,
                 sigma_f1: FloatOrVectorLike = 5e-2,
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
                 sigma_f1: FloatOrVectorLike = 5e-2,
                 sigma_kp: FloatOrVectorLike = 1e2,
                 weight: float = 1,
                 T: float = 298.,
                 Tunit: Literal['C', 'K'] = 'K',
                 name: str = '',
                 source: str = '',
                 ) -> None:
        """Construct `DriftDataset` with the given parameters."""
        super().__init__(M1, M2, f1, kp, sigma_f1, sigma_kp, weight, T, Tunit,
                         name, source)
