# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from polykin.utils.math import eps
from polykin.utils.tools import (check_bounds, convert_check_pressure,
                                 convert_check_temperature)
from polykin.utils.types import FloatArray, FloatArrayLike

from ..equations.base import PropertyEquation

__all__ = []

# %% Parameter tables

table_parameters: dict[str, Optional[pd.DataFrame]] = {}


def load_PVT_parameters(method: str) -> pd.DataFrame:
    "Load table with PVT parameters for a given equation."
    global table_parameters
    table = table_parameters.get(method, None)
    if table is None:
        filepath = (Path(__file__).parent).joinpath(method + '_parameters.tsv')
        table = pd.read_csv(filepath, sep=r"\s+")
        table.set_index('Polymer', inplace=True)
        table_parameters[method] = table
    return table


# %% PolymerPVTEquation

class PolymerPVTEquation(PropertyEquation):
    r"""_Abstract_ polymer PVT equation, $\hat{V}(T, P)$"""

    Trange: tuple[Union[float, FloatArray], Union[float, FloatArray]]
    Prange: tuple[Union[float, FloatArray], Union[float, FloatArray]]
    symbol = r"$\hat{V}$"
    unit = "m³/kg"

    def __init__(self,
                 Tmin: float,
                 Tmax: float,
                 Pmin: float,
                 Pmax: float,
                 name: str
                 ) -> None:

        # Check bounds
        check_bounds(Tmin, 0, np.inf, 'Tmin')
        check_bounds(Tmax, 0, np.inf, 'Tmax')
        check_bounds(Tmax-Tmin, eps, np.inf, 'Tmax-Tmin')
        check_bounds(Pmin, 0, np.inf, 'Pmin')
        check_bounds(Pmax, 0, np.inf, 'Pmax')
        check_bounds(Pmax-Pmin, eps, np.inf, 'Pmax-Pmin')

        self.Trange = (Tmin, Tmax)
        self.Prange = (Pmin, Pmax)
        self.name = name

    def V(self,
          T: Union[float, FloatArrayLike],
          P: Union[float, FloatArrayLike],
          Tunit: Literal['C', 'K'] = 'K',
          Punit: Literal['bar', 'MPa', 'Pa'] = 'Pa'
          ) -> Union[float, FloatArray]:
        r"""Evaluate the specific volume, $\hat{V}$, at given temperature and
        pressure, including unit conversion and range check.

        Parameters
        ----------
        T : float | FloatArrayLike
            Temperature.
            Unit = `Tunit`.
        P : float | FloatArrayLike
            Pressure.
            Unit = `Punit`.
        Tunit : Literal['C', 'K']
            Temperature unit.
        Punit : Literal['bar', 'MPa', 'Pa']
            Pressure unit.

        Returns
        -------
        float | FloatArray
            Specific volume.
            Unit = m³/kg.
        """
        TK = convert_check_temperature(T, Tunit, self.Trange)
        Pa = convert_check_pressure(P, Punit, self.Prange)
        return self.eval(TK, Pa)

    @abstractmethod
    def eval(self,
             T: Union[float, FloatArray],
             P: Union[float, FloatArray]
             ) -> Union[float, FloatArray]:
        r"""Evaluate specific volume, $\hat{V}$, at given SI conditions without
        unit conversions or checks.

        Parameters
        ----------
        T : float | FloatArray
            Temperature.
            Unit = K.
        P : float | FloatArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        float | FloatArray
            Specific volume.
            Unit = m³/kg.
        """
        pass

    @abstractmethod
    def alpha(self,
              T: Union[float, FloatArray],
              P: Union[float, FloatArray]
              ) -> Union[float, FloatArray]:
        r"""Calculate thermal expansion coefficient, $\alpha$.

        $$\alpha=\frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_{P}$$

        Parameters
        ----------
        T : float | FloatArray
            Temperature.
            Unit = K.
        P : float | FloatArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        float | FloatArray
            Thermal expansion coefficient, $\alpha$.
        """
        pass

    @abstractmethod
    def beta(self,
             T: Union[float, FloatArray],
             P: Union[float, FloatArray]
             ) -> Union[float, FloatArray]:
        r"""Calculate isothermal compressibility coefficient, $\beta$.

        $$\beta=-\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_{T}$$

        Parameters
        ----------
        T : float | FloatArray
            Temperature.
            Unit = K.
        P : float | FloatArray
            Pressure.
            Unit = Pa.

        Returns
        -------
        float | FloatArray
            Isothermal compressibility coefficient, $\beta$.
        """
        pass

    @classmethod
    def from_database(cls,
                      name: str
                      ) -> Optional[PolymerPVTEquation]:
        r"""Construct `PolymerPVTEquation` with parameters from the database.

        Parameters
        ----------
        name : str
            Polymer code name.
        """
        table = load_PVT_parameters(method=cls.__name__)
        try:
            mask = table.index == name
            parameters = table[mask].iloc[0, :].to_dict()
            return cls(**parameters, name=name)
        except IndexError:
            print(
                f"Error: '{name}' does not exist in polymer database.\n"
                f"Valid names are: {table.index.to_list()}")

    @classmethod
    def get_database(cls) -> pd.DataFrame:
        r"""Get database with parameters for the respective PVT equation.

        | Method          | Reference                            |
        | :-----------    | ------------------------------------ |
        | Flory           | [2] Table 4.1.7  (p. 72-73)          |
        | Hartmann-Haque  | [2] Table 4.1.11 (p. 85-86)          |
        | Sanchez-Lacombe | [2] Table 4.1.9  (p. 78-79)          |
        | Tait            | [1] Table 3B-1 (p. 41)               |

        **References**

        1.  Danner, Ronald P., and Martin S. High. Handbook of polymer
            solution thermodynamics. John Wiley & Sons, 2010.
        2.  Caruthers et al. Handbook of Diffusion and Thermal Properties of
            Polymers and Polymer Solutions. AIChE, 1998.
        """
        return load_PVT_parameters(method=cls.__name__)
