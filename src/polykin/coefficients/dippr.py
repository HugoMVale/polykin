from polykin.utils import check_type, check_shapes, \
    FloatOrArray, FloatOrArrayLike
from polykin.coefficients.baseclasses import CoefficientT

import numpy as np


class DIPPR(CoefficientT):
    """_Abstract_ class for all
    [DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)
    temperature-dependent equations."""
    pass


class DIPPRP5(DIPPR):
    """_Abstract_ class for DIPPR equations with 5 parameters (A-E)."""

    def __init__(self,
                 A: FloatOrArrayLike,
                 B: FloatOrArrayLike,
                 C: FloatOrArrayLike,
                 D: FloatOrArrayLike,
                 E: FloatOrArrayLike,
                 Tmin: FloatOrArrayLike,
                 Tmax: FloatOrArrayLike,
                 unit,
                 symbol,
                 name
                 ) -> None:

        # Convert lists to arrays
        if isinstance(A, list):
            A = np.array(A, dtype=np.float64)
        if isinstance(B, list):
            B = np.array(B, dtype=np.float64)
        if isinstance(C, list):
            C = np.array(C, dtype=np.float64)
        if isinstance(D, list):
            D = np.array(D, dtype=np.float64)
        if isinstance(E, list):
            E = np.array(E, dtype=np.float64)
        if isinstance(Tmin, list):
            Tmin = np.array(Tmin, dtype=np.float64)
        if isinstance(Tmax, list):
            Tmax = np.array(Tmax, dtype=np.float64)

        # Check shapes
        self._shape = check_shapes([A, B, C, D, E], [Tmin, Tmax])

        # Check types
        check_type(unit, str, 'unit')
        check_type(symbol, str, 'symbol')

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
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
            f"A:         {self.A}\n" \
            f"B:         {self.B}\n" \
            f"C:         {self.C}\n" \
            f"D:         {self.D}\n" \
            f"E:         {self.E}\n" \
            f"Tmin [K]:  {self.Tmin}\n" \
            f"Tmax [K]:  {self.Tmax}"


class DIPPRP4(DIPPRP5):
    """_Abstract_ class for DIPPR equations with 4 parameters (A-D)."""

    def __init__(self,
                 A: FloatOrArrayLike,
                 B: FloatOrArrayLike,
                 C: FloatOrArrayLike,
                 D: FloatOrArrayLike,
                 Tmin: FloatOrArrayLike,
                 Tmax: FloatOrArrayLike,
                 unit,
                 symbol,
                 name
                 ) -> None:

        if isinstance(A, (list, np.ndarray)):
            E = [0.0]*len(A)
        else:
            E = 0.0
        super().__init__(A, B, C, D, E, Tmin, Tmax, unit, symbol, name)

    def __repr__(self) -> str:
        return \
            f"name:      {self.name}\n" \
            f"symbol:    {self.symbol}\n" \
            f"unit:      {self.unit}\n" \
            f"A:         {self.A}\n" \
            f"B:         {self.B}\n" \
            f"C:         {self.C}\n" \
            f"D:         {self.D}\n" \
            f"Tmin [K]:  {self.Tmin}\n" \
            f"Tmax [K]:  {self.Tmax}"


class DIPPR100(DIPPRP5):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-100 equation.

    This equation implements the following temperature dependence:

    $$ Y = A + B T + C T^2 + D T^3 + E T^4 $$

    where $A$ to $E$ are constant parameters and $T$ is the absolute
    temperature.

    Parameters
    ----------
    A : FloatOrArrayLike
        Parameter of equation.
    B : FloatOrArrayLike
        Parameter of equation.
    C : FloatOrArrayLike
        Parameter of equation.
    D : FloatOrArrayLike
        Parameter of equation.
    E : FloatOrArrayLike
        Parameter of equation.
    Tmin : FloatOrArrayLike
        Lower temperature bound.
        Unit = K.
    Tmax : FloatOrArrayLike
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of output variable $Y$.
    symbol : str
        Symbol of output variable $Y$.
    name : str
        Name.
    """

    def __init__(self,
                 A: FloatOrArrayLike = 0.,
                 B: FloatOrArrayLike = 0.,
                 C: FloatOrArrayLike = 0.,
                 D: FloatOrArrayLike = 0.,
                 E: FloatOrArrayLike = 0.,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = np.inf,
                 unit: str = '-',
                 symbol: str = 'Y',
                 name: str = ''
                 ) -> None:

        super().__init__(A, B, C, D, E, Tmin, Tmax, unit, symbol, name)

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate correlation at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Property value, $Y$.
        """
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        return A + B * T + C * T**2 + D * T**3 + E * T**4


class DIPPR101(DIPPRP5):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-101 equation.

    This equation implements the following temperature dependence:

    $$ Y = \exp{\left(A + B / T + C \ln(T) + D T^E\right)} $$

    where $A$ to $E$ are constant parameters and $T$ is the absolute
    temperature.

    Parameters
    ----------
    A : FloatOrArrayLike
        Parameter of equation.
    B : FloatOrArrayLike
        Parameter of equation.
    C : FloatOrArrayLike
        Parameter of equation.
    D : FloatOrArrayLike
        Parameter of equation.
    E : FloatOrArrayLike
        Parameter of equation.
    Tmin : FloatOrArrayLike
        Lower temperature bound.
        Unit = K.
    Tmax : FloatOrArrayLike
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of output variable $Y$.
    symbol : str
        Symbol of output variable $Y$.
    name : str
        Name.
    """

    def __init__(self,
                 A: FloatOrArrayLike,
                 B: FloatOrArrayLike,
                 C: FloatOrArrayLike = 0.,
                 D: FloatOrArrayLike = 0.,
                 E: FloatOrArrayLike = 0.,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = np.inf,
                 unit: str = '-',
                 symbol: str = 'Y',
                 name: str = ''
                 ) -> None:

        super().__init__(A, B, C, D, E, Tmin, Tmax, unit, symbol, name)

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate correlation at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.

        Returns
        -------
        FloatOrArray
            Property value, $Y$.
        """
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        return np.exp(A + B / T + C * np.log(T) + D * T**E)


class DIPPR105(DIPPRP4):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-105 equation.

    This equation implements the following temperature dependence:

    $$ Y = \frac{A}{B^{ \left( 1 + (1 - T / C)^D \right) }} $$

    where $A$ to $D$ are constant parameters and $T$ is the absolute
    temperature.

    Parameters
    ----------
    A : FloatOrArrayLike
        Parameter of equation.
    B : FloatOrArrayLike
        Parameter of equation.
    C : FloatOrArrayLike
        Parameter of equation.
    D : FloatOrArrayLike
        Parameter of equation.
    Tmin : FloatOrArrayLike
        Lower temperature bound.
        Unit = K.
    Tmax : FloatOrArrayLike
        Upper temperature bound.
        Unit = K.
    unit : str
        Unit of output variable $Y$.
    symbol : str
        Symbol of output variable $Y$.
    name : str
        Name.
    """

    def __init__(self,
                 A: FloatOrArrayLike,
                 B: FloatOrArrayLike,
                 C: FloatOrArrayLike,
                 D: FloatOrArrayLike,
                 Tmin: FloatOrArrayLike = 0.0,
                 Tmax: FloatOrArrayLike = np.inf,
                 unit: str = '-',
                 symbol: str = 'Y',
                 name: str = ''
                 ) -> None:

        super().__init__(A, B, C, D, Tmin, Tmax, unit, symbol, name)

    def eval(self, T: FloatOrArray) -> FloatOrArray:
        """Evaluate correlation at given SI conditions, without unit
        conversions or checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature. 
            Unit = K.

        Returns
        -------
        FloatOrArray
            Property value, $Y$.
        """
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        return A / B**(1 + (1 - T / C)**D)
