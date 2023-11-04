# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.utils import check_shapes, check_bounds, convert_list_to_array, \
    FloatOrArray, FloatOrArrayLike, ShapeError
from polykin.properties.base import PropertyEquationT

import numpy as np
from scipy.constants import h, R, Boltzmann as kB
from typing import Optional


__all__ = ['Arrhenius', 'Eyring']


class KineticCoefficientT(PropertyEquationT):

    _shape: Optional[tuple]

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        """Shape of underlying parameter array."""
        return self._shape


class Arrhenius(KineticCoefficientT):
    r"""[Arrhenius](https://en.wikipedia.org/wiki/Arrhenius_equation) kinetic
    rate coefficient.

    This class implements the following temperature dependence:

    $$
    k(T)=k_0
    \exp\left[-\frac{E_a}{R}\left(\frac{1}{T}-\frac{1}{T_0}\right)\right]
    $$

    where $T_0$ is a convenient reference temperature, $E_a$ is the activation
    energy, and $k_0=k(T_0)$. In the limit $T\rightarrow+\infty$, the usual
    form of the Arrhenius equation with $k_0=A$ is recovered.

    Parameters
    ----------
    k0 : FloatOrArrayLike
        Coefficient value at the reference temperature, $k_0=k(T_0)$.
        Unit = `unit`.
    EaR : FloatOrArrayLike
        Energy of activation, $E_a/R$.
        Unit = K.
    T0 : FloatOrArrayLike
        Reference temperature, $T_0$.
        Unit = K.
    Tmin : FloatOrArrayLike
        Lower temperature bound.
        Unit = K.
    Tmax : FloatOrArrayLike
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of coefficient.
    symbol : str
        Symbol of coefficient $k$.
    name : str
        Name.
    """

    _pnames = (('k0', 'EaR'), ('T0',))
    _punits = ('#', 'K', 'K')

    def __init__(self,
                 k0: FloatOrArrayLike,
                 EaR: FloatOrArrayLike,
                 T0: FloatOrArrayLike = np.inf,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = np.inf,
                 unit: str = '-',
                 symbol: str = 'k',
                 name: str = ''
                 ) -> None:
        """Construct `Arrhenius` with the given parameters."""

        # Convert lists to arrays
        k0, EaR, T0, Tmin, Tmax = \
            convert_list_to_array([k0, EaR, T0, Tmin, Tmax])

        # Check shapes
        self._shape = check_shapes([k0, EaR], [T0, Tmin, Tmax])

        # Check bounds
        check_bounds(k0, 0., np.inf, 'k0')
        check_bounds(EaR, -np.inf, np.inf, 'EaR')
        check_bounds(T0, 0., np.inf, 'T0')

        self.pvalues = (k0, EaR, T0)
        super().__init__((Tmin, Tmax), unit, symbol, name)

    @staticmethod
    def equation(T: FloatOrArray,
                 k0: FloatOrArray,
                 EaR: FloatOrArray,
                 T0: FloatOrArray,
                 ) -> FloatOrArray:
        r"""Arrhenius equation.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        k0 : FloatOrArray
            Coefficient value at the reference temperature, $k_0=k(T_0)$.
            Unit = Any.
        EaR : FloatOrArray
            Energy of activation, $E_a/R$.
            Unit = K.
        T0 : FloatOrArray
            Reference temperature, $T_0$.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Coefficient value. Unit = [k0].
        """
        return k0 * np.exp(-EaR*(1/T - 1/T0))

    @property
    def A(self) -> FloatOrArray:
        r"""Pre-exponential factor, $A=k_0 e^{E_a/(R T_0)}$."""
        return self.__call__(np.inf)

    def __mul__(self, other):
        """Multipy Arrhenius coefficient(s).

        Create a new Arrhenius coefficient from a product of two Arrhenius
        coefficients with identical shapes or a product of an Arrhenius
        coefficient and a numerical constant.

        Parameters
        ----------
        other : Arrhenius | float | int
            Another Arrhenius coefficient or number.

        Returns
        -------
        Arrhenius
            Product coefficient.
        """
        if isinstance(other, Arrhenius):
            if self._shape == other._shape:
                return Arrhenius(k0=self.A*other.A,
                                 EaR=self.pvalues[1] + other.pvalues[1],
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
            return Arrhenius(k0=self.pvalues[0]*other,
                             EaR=self.pvalues[1],
                             T0=self.pvalues[2],
                             Tmin=self.Trange[0],
                             Tmax=self.Trange[1],
                             unit=self.unit,
                             symbol=f"{str(other)}·{self.symbol}",
                             name=f"{str(other)}·{self.name}")
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide Arrhenius coefficient(s).

        Create a new Arrhenius coefficient from a division of two Arrhenius
        coefficients with identical shapes or a division involving an Arrhenius
        coefficient and a numerical constant.

        Parameters
        ----------
        other : Arrhenius | float | int
            Another Arrhenius coefficient or number.

        Returns
        -------
        Arrhenius
            Quotient coefficient.
        """
        if isinstance(other, Arrhenius):
            if self._shape == other._shape:
                return Arrhenius(k0=self.A/other.A,
                                 EaR=self.pvalues[1] - other.pvalues[1],
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
            return Arrhenius(k0=self.pvalues[0]/other,
                             EaR=self.pvalues[1],
                             T0=self.pvalues[2],
                             Tmin=self.Trange[0],
                             Tmax=self.Trange[1],
                             unit=self.unit,
                             symbol=f"{self.symbol}/{str(other)}",
                             name=f"{self.name}/{str(other)}")
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Arrhenius(k0=other/self.pvalues[0],
                             EaR=-self.pvalues[1],
                             T0=self.pvalues[2],
                             Tmin=self.Trange[0],
                             Tmax=self.Trange[1],
                             unit=f"1/{self.unit}",
                             symbol=f"{str(other)}/{self.symbol}",
                             name=f"{str(other)}/{self.name}")
        else:
            return NotImplemented

    def __pow__(self, other):
        """Power of an Arrhenius coefficient.

        Create a new Arrhenius coefficient from the exponentiation of an
        Arrhenius coefficient.

        Parameters
        ----------
        other : float | int
            Exponent.

        Returns
        -------
        Arrhenius
            Power coefficient.
        """
        if isinstance(other, (int, float)):
            return Arrhenius(k0=self.pvalues[0]**other,
                             EaR=self.pvalues[1]*other,
                             T0=self.pvalues[2],
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


class Eyring(KineticCoefficientT):
    r"""[Eyring](https://en.wikipedia.org/wiki/Eyring_equation) kinetic rate
    coefficient.

    This class implements the following temperature dependence:

    $$
    k(T) = \dfrac{\kappa k_B T}{h}
           \exp\left(\frac{\Delta S^\ddagger}{R}\right)
           \exp\left(-\frac{\Delta H^\ddagger}{R T}\right)
    $$

    where $\kappa$ is the transmission coefficient, $\Delta S^\ddagger$ is
    the entropy of activation, and $\Delta H^\ddagger$ is the enthalpy of
    activation. The unit of $k$ is physically set to s$^{-1}$.

    Parameters
    ----------
    DSa : FloatOrArrayLike
        Entropy of activation, $\Delta S^\ddagger$.
        Unit = J/(mol·K).
    DHa : FloatOrArrayLike
        Enthalpy of activation, $\Delta H^\ddagger$.
        Unit = J/mol.
    kappa : FloatOrArrayLike
        Transmission coefficient.
    Tmin : FloatOrArrayLike
        Lower temperature bound.
        Unit = K.
    Tmax : FloatOrArrayLike
        Upper temperature bound.
        Unit = K.
    symbol : str
        Symbol of coefficient $k$.
    name : str
        Name.
    """

    _pnames = (('DSa', 'DHa'), ('kappa',))
    _punits = ('J/(mol·K)', 'J/mol', '')

    def __init__(self,
                 DSa: FloatOrArrayLike,
                 DHa: FloatOrArrayLike,
                 kappa: FloatOrArrayLike = 1.0,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = np.inf,
                 symbol: str = 'k',
                 name: str = ''
                 ) -> None:
        """Construct `Eyring` with the given parameters."""

        # Convert lists to arrays
        DSa, DHa, kappa, Tmin, Tmax = \
            convert_list_to_array([DSa, DHa, kappa, Tmin, Tmax])

        # Check shapes
        self._shape = check_shapes([DSa, DHa], [kappa, Tmin, Tmax])

        # Check bounds
        check_bounds(DSa, 0., np.inf, 'DSa')
        check_bounds(DHa, 0., np.inf, 'DHa')
        check_bounds(kappa, 0., 1., 'kappa')

        self.pvalues = (DSa, DHa, kappa)
        super().__init__((Tmin, Tmax), '1/s', symbol, name)

    @staticmethod
    def equation(T: FloatOrArray,
                 DSa: FloatOrArray,
                 DHa: FloatOrArray,
                 kappa: FloatOrArray,
                 ) -> FloatOrArray:
        r"""Eyring equation.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        DSa : FloatOrArray
            Entropy of activation, $\Delta S^\ddagger$.
            Unit = J/(mol·K).
        DHa : FloatOrArray
            Enthalpy of activation, $\Delta H^\ddagger$.
            Unit = J/mol.
        kappa : FloatOrArray
            Transmission coefficient.

        Returns
        -------
        FloatOrArray
            Coefficient value. Unit = 1/s.
        """
        return kappa * kB*T/h * np.exp((DSa - DHa/T)/R)
