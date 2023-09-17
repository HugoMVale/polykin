from polykin.utils import check_type, check_shapes, check_bounds, \
    FloatOrArray, FloatOrArrayLike, ShapeError, \
    eps
from polykin.coefficients.baseclasses import CoefficientT

import numpy as np
from scipy.constants import h, R, Boltzmann as kB


class Arrhenius(CoefficientT):
    r"""[Arrhenius](https://en.wikipedia.org/wiki/Arrhenius_equation) kinetic
    rate coefficient.

    This coefficient implements the following temperature dependence:

    $$ k(T)=k_0\exp\left[-\frac{E_a}{R}\left(\frac{1}{T}-\frac{1}{T_0} \\
        \right)\right] $$

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

        # Convert lists to arrays
        if isinstance(k0, list):
            k0 = np.array(k0, dtype=np.float64)
        if isinstance(EaR, list):
            EaR = np.array(EaR, dtype=np.float64)
        if isinstance(T0, list):
            T0 = np.array(T0, dtype=np.float64)
        if isinstance(Tmin, list):
            Tmin = np.array(Tmin, dtype=np.float64)
        if isinstance(Tmax, list):
            Tmax = np.array(Tmax, dtype=np.float64)

        # Check shapes
        self._shape = check_shapes([k0, EaR], [T0, Tmin, Tmax])

        # Check bounds
        check_bounds(k0, 0, np.inf, 'k0')
        check_bounds(EaR, -np.inf, np.inf, 'EaR')
        check_bounds(T0, 0, np.inf, 'T0')
        check_bounds(Tmin, 0, np.inf, 'Tmin')
        check_bounds(Tmax, 0, np.inf, 'Tmax')
        check_bounds(Tmax-Tmin, eps, np.inf, 'Tmax-Tmin')

        # Check types
        check_type(unit, str, 'unit')
        check_type(symbol, str, 'symbol')

        self.k0 = k0
        self.EaR = EaR
        self.T0 = T0
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.unit = unit
        self.symbol = symbol
        self.name = name

    def __repr__(self) -> str:
        return \
            f"name:      {self.name}\n" \
            f"symbol:    {self.symbol}\n" \
            f"unit:      {self.unit}\n" \
            f"k0:        {self.k0}\n" \
            f"Ea/R [K]:  {self.EaR}\n" \
            f"T0   [K]:  {self.T0}\n" \
            f"Tmin [K]:  {self.Tmin}\n" \
            f"Tmax [K]:  {self.Tmax}"

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
                                 EaR=self.EaR + other.EaR,
                                 Tmin=np.maximum(self.Tmin, other.Tmin),
                                 Tmax=np.minimum(self.Tmax, other.Tmax),
                                 unit=f"{self.unit}·{other.unit}",
                                 symbol=f"{self.symbol}·{other.symbol}",
                                 name=f"{self.name}·{other.name}")
            else:
                raise ShapeError(
                    "Product of array-like coefficients requires identical shapes.")  # noqa: E501
        elif isinstance(other, (int, float)):
            return Arrhenius(k0=self.k0*other,
                             EaR=self.EaR,
                             T0=self.T0,
                             Tmin=self.Tmin,
                             Tmax=self.Tmax,
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
                                 EaR=self.EaR - other.EaR,
                                 Tmin=np.maximum(self.Tmin, other.Tmin),
                                 Tmax=np.minimum(self.Tmax, other.Tmax),
                                 unit=f"{self.unit}/{other.unit}",
                                 symbol=f"{self.symbol}/{other.symbol}",
                                 name=f"{self.name}/{other.name}")
            else:
                raise ShapeError(
                    "Division of array-like coefficients requires identical shapes.")  # noqa: E501
        elif isinstance(other, (int, float)):
            return Arrhenius(k0=self.k0/other,
                             EaR=self.EaR,
                             T0=self.T0,
                             Tmin=self.Tmin,
                             Tmax=self.Tmax,
                             unit=self.unit,
                             symbol=f"{self.symbol}/{str(other)}",
                             name=f"{self.name}/{str(other)}")
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Arrhenius(k0=other/self.k0,
                             EaR=-self.EaR,
                             T0=self.T0,
                             Tmin=self.Tmin,
                             Tmax=self.Tmax,
                             unit=f"1/{self.unit}",
                             symbol=f"{str(other)}/{self.symbol}",
                             name=f"{str(other)}/{self.name}")
        else:
            return NotImplemented

    @property
    def A(self) -> FloatOrArray:
        r"""Pre-exponential factor, $A=k_0 e^{E_a/(R T_0)}$."""
        return self.eval(np.inf)

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        return self.k0 * np.exp(-self.EaR*(1/T - 1/self.T0))


class Eyring(CoefficientT):
    r"""[Eyring](https://en.wikipedia.org/wiki/Eyring_equation) kinetic rate
    coefficient.

    This coefficient implements the following temperature dependence:

    $$ k(T)=\dfrac{\kappa k_B T}{h} \\
        \exp\left(\frac{\Delta S^\ddagger}{R}\right) \\
        \exp\left(-\frac{\Delta H^\ddagger}{R T}\right)$$

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

    unit: str = '1/s'

    def __init__(self,
                 DSa: FloatOrArrayLike,
                 DHa: FloatOrArrayLike,
                 kappa: FloatOrArrayLike = 1.0,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = np.inf,
                 symbol: str = 'k',
                 name: str = ''
                 ) -> None:

        # Convert lists to arrays
        if isinstance(DSa, list):
            DSa = np.array(DSa, dtype=np.float64)
        if isinstance(DHa, list):
            DHa = np.array(DHa, dtype=np.float64)
        if isinstance(kappa, list):
            kappa = np.array(kappa, dtype=np.float64)
        if isinstance(Tmin, list):
            Tmin = np.array(Tmin, dtype=np.float64)
        if isinstance(Tmax, list):
            Tmax = np.array(Tmax, dtype=np.float64)

        # Check shapes
        self._shape = check_shapes([DSa, DHa], [kappa, Tmin, Tmax])

        # Check bounds
        check_bounds(DSa, 0, np.inf, 'DSa')
        check_bounds(DHa, 0, np.inf, 'DHa')
        check_bounds(kappa, 0, 1, 'kappa')
        check_bounds(Tmin, 0, np.inf, 'Tmin')
        check_bounds(Tmax, 0, np.inf, 'Tmax')
        check_bounds(Tmax-Tmin, eps, np.inf, 'Tmax-Tmin')

        # check types
        check_type(symbol, str, 'symbol')

        self.DSa = DSa
        self.DHa = DHa
        self.kappa = kappa
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.symbol = symbol
        self.name = name

    def __repr__(self) -> str:
        return \
            f"name:             {self.name}\n" \
            f"symbol:           {self.symbol}\n" \
            f"unit:             {self.unit}\n" \
            f"DSa [J/(mol·K)]:  {self.DSa}\n" \
            f"DHa [J/mol]:      {self.DHa}\n" \
            f"kappa [—]:        {self.kappa}\n" \
            f"Tmin [K]:         {self.Tmin}\n" \
            f"Tmax [K]:         {self.Tmax}"

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        return self.kappa * kB*T/h * np.exp((self.DSa - self.DHa/T)/R)