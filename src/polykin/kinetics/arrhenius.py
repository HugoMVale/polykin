# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from __future__ import annotations

from typing import Union

import numpy as np
from numpy import exp

from polykin.kinetics.base import KineticCoefficientT
from polykin.utils.exceptions import ShapeError
from polykin.utils.math import convert_FloatOrArrayLike_to_FloatOrArray
from polykin.utils.tools import check_bounds, check_shapes
from polykin.utils.types import FloatArray, FloatArrayLike

__all__ = ['Arrhenius']


class Arrhenius(KineticCoefficientT):
    r"""[Arrhenius](https://en.wikipedia.org/wiki/Arrhenius_equation) kinetic
    rate coefficient.

    This class implements the following temperature dependence:

    $$
    k(T)=k_0
    \exp\left[-\frac{E_a}{R}\left(\frac{1}{T}-\frac{1}{T_0}\right)\right]
    $$

    where $T_0$ is a reference temperature, $E_a$ is the activation energy,
    and $k_0=k(T_0)$. In the limit $T\rightarrow+\infty$, the usual
    form of the Arrhenius equation with $k_0=A$ is recovered.

    Parameters
    ----------
    k0 : float | FloatArrayLike
        Coefficient value at the reference temperature, $k_0=k(T_0)$.
        Unit = `unit`.
    EaR : float | FloatArrayLike
        Energy of activation, $E_a/R$.
        Unit = K.
    T0 : float | FloatArrayLike
        Reference temperature, $T_0$.
        Unit = K.
    Tmin : float | FloatArrayLike
        Lower temperature bound.
        Unit = K.
    Tmax : float | FloatArrayLike
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of coefficient.
    symbol : str
        Symbol of coefficient $k$.
    name : str
        Name.

    See also
    --------
    * [`Eyring`](Eyring.md): alternative method.

    Examples
    --------
    Define and evaluate the propagation rate coefficient of styrene.
    >>> from polykin.kinetics import Arrhenius 
    >>> kp = Arrhenius(
    ...     10**7.63,       # pre-exponential factor
    ...     32.5e3/8.314,   # Ea/R, K
    ...     Tmin=261., Tmax=366.,
    ...     symbol='k_p',
    ...     unit='L/mol/s',
    ...     name='kp of styrene')
    >>> kp(25.,'C') 
    86.28385101961442
    """

    _pinfo = {'k0': ('#', True), 'EaR': ('K', True), 'T0': ('K', False)}

    def __init__(self,
                 k0: Union[float, FloatArrayLike],
                 EaR: Union[float, FloatArrayLike],
                 T0: Union[float, FloatArrayLike] = np.inf,
                 Tmin: Union[float, FloatArrayLike] = 0.0,
                 Tmax: Union[float, FloatArrayLike] = np.inf,
                 unit: str = '-',
                 symbol: str = 'k',
                 name: str = ''
                 ) -> None:

        # Convert lists to arrays
        k0, EaR, T0, Tmin, Tmax = \
            convert_FloatOrArrayLike_to_FloatOrArray([k0, EaR, T0, Tmin, Tmax])

        # Check shapes
        self._shape = check_shapes([k0, EaR], [T0, Tmin, Tmax])

        # Check bounds
        check_bounds(k0, 0., np.inf, 'k0')
        check_bounds(EaR, -np.inf, np.inf, 'EaR')
        check_bounds(T0, 0., np.inf, 'T0')

        self.p = {'k0': k0, 'EaR': EaR, 'T0': T0}
        super().__init__((Tmin, Tmax), unit, symbol, name)

    @staticmethod
    def equation(T: Union[float, FloatArray],
                 k0: Union[float, FloatArray],
                 EaR: Union[float, FloatArray],
                 T0: Union[float, FloatArray],
                 ) -> Union[float, FloatArray]:
        r"""Arrhenius equation.

        Parameters
        ----------
        T : float | FloatArray
            Temperature.
            Unit = K.
        k0 : float | FloatArray
            Coefficient value at the reference temperature, $k_0=k(T_0)$.
            Unit = Any.
        EaR : float | FloatArray
            Energy of activation, $E_a/R$.
            Unit = K.
        T0 : float | FloatArray
            Reference temperature, $T_0$.
            Unit = K.

        Returns
        -------
        float | FloatArray
            Coefficient value. Unit = [k0].
        """
        return k0 * exp(-EaR*(1/T - 1/T0))

    @property
    def A(self) -> Union[float, FloatArray]:
        r"""Pre-exponential factor, $A=k_0 e^{E_a/(R T_0)}$."""
        return self.__call__(np.inf)

    def __mul__(self,
                other: Union[int, float, Arrhenius]
                ) -> Arrhenius:
        """Multipy Arrhenius coefficient(s).

        Create a new Arrhenius coefficient from a product of two Arrhenius
        coefficients with identical shapes or a product of an Arrhenius
        coefficient and a numerical constant.

        Parameters
        ----------
        other : int | float | Arrhenius
            Another Arrhenius coefficient or number.

        Returns
        -------
        Arrhenius
            Product coefficient.
        """
        if isinstance(other, Arrhenius):
            if self._shape == other._shape:
                return Arrhenius(k0=self.A*other.A,
                                 EaR=self.p['EaR'] + other.p['EaR'],
                                 Tmin=np.maximum(
                                     self.Trange[0], other.Trange[0]),
                                 Tmax=np.minimum(
                                     self.Trange[1], other.Trange[1]),
                                 unit=f"{self.unit}·{other.unit}",
                                 symbol=f"{self.symbol}·{other.symbol}",
                                 name=f"{self.name}·{other.name}")
            else:
                raise ShapeError(
                    "Product of array-like coefficients requires identical shapes.")  # noqa: E501
        elif isinstance(other, (int, float)):
            return Arrhenius(k0=self.p['k0']*other,
                             EaR=self.p['EaR'],
                             T0=self.p['T0'],
                             Tmin=self.Trange[0],
                             Tmax=self.Trange[1],
                             unit=self.unit,
                             symbol=f"{str(other)}·{self.symbol}",
                             name=f"{str(other)}·{self.name}")
        else:
            return NotImplemented

    def __rmul__(self,
                 other: Union[int, float, Arrhenius]
                 ) -> Arrhenius:
        return self.__mul__(other)

    def __truediv__(self,
                    other: Union[int, float, Arrhenius]
                    ) -> Arrhenius:
        """Divide Arrhenius coefficient(s).

        Create a new Arrhenius coefficient from a division of two Arrhenius
        coefficients with identical shapes or a division involving an Arrhenius
        coefficient and a numerical constant.

        Parameters
        ----------
        other : int | float | Arrhenius
            Another Arrhenius coefficient or number.

        Returns
        -------
        Arrhenius
            Quotient coefficient.
        """
        if isinstance(other, Arrhenius):
            if self._shape == other._shape:
                return Arrhenius(k0=self.A/other.A,
                                 EaR=self.p['EaR'] - other.p['EaR'],
                                 Tmin=np.maximum(
                                     self.Trange[0], other.Trange[0]),
                                 Tmax=np.minimum(
                                     self.Trange[1], other.Trange[1]),
                                 unit=f"{self.unit}/{other.unit}",
                                 symbol=f"{self.symbol}/{other.symbol}",
                                 name=f"{self.name}/{other.name}")
            else:
                raise ShapeError(
                    "Division of array-like coefficients requires identical shapes.")  # noqa: E501
        elif isinstance(other, (int, float)):
            return Arrhenius(k0=self.p['k0']/other,
                             EaR=self.p['EaR'],
                             T0=self.p['T0'],
                             Tmin=self.Trange[0],
                             Tmax=self.Trange[1],
                             unit=self.unit,
                             symbol=f"{self.symbol}/{str(other)}",
                             name=f"{self.name}/{str(other)}")
        else:
            return NotImplemented

    def __rtruediv__(self,
                     other: Union[int, float]
                     ) -> Arrhenius:
        if isinstance(other, (int, float)):
            return Arrhenius(k0=other/self.p['k0'],
                             EaR=-self.p['EaR'],
                             T0=self.p['T0'],
                             Tmin=self.Trange[0],
                             Tmax=self.Trange[1],
                             unit=f"1/{self.unit}",
                             symbol=f"{str(other)}/{self.symbol}",
                             name=f"{str(other)}/{self.name}")
        else:
            return NotImplemented

    def __pow__(self,
                other: Union[int, float]
                ) -> Arrhenius:
        """Power of an Arrhenius coefficient.

        Create a new Arrhenius coefficient from the exponentiation of an
        Arrhenius coefficient.

        Parameters
        ----------
        other : int | float
            Exponent.

        Returns
        -------
        Arrhenius
            Power coefficient.
        """
        if isinstance(other, (int, float)):
            return Arrhenius(k0=self.p['k0']**other,
                             EaR=self.p['EaR']*other,
                             T0=self.p['T0'],
                             Tmin=self.Trange[0],
                             Tmax=self.Trange[1],
                             unit=f"({self.unit})^{str(other)}",
                             symbol=f"({self.symbol})^{str(other)}",
                             name=f"{self.name}^{str(other)}"
                             )
        else:
            return NotImplemented

    def __rpow__(self, other):
        return NotImplemented
