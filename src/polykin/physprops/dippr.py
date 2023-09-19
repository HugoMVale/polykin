from polykin.utils import check_type, check_shapes, convert_list_to_array, \
    FloatOrArray, FloatOrArrayLike
from polykin.correlation import CorrelationT

import numpy as np

__all__ = ['DIPPR100', 'DIPPR101', 'DIPPR102', 'DIPPR104', 'DIPPR105',
           'DIPPR106']


class DIPPR(CorrelationT):
    """_Abstract_ class for all
    [DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)
    temperature-dependent equations."""
    pass


class DIPPRP4(DIPPR):
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

        # Convert lists to arrays
        A, B, C, D, Tmin, Tmax = \
            convert_list_to_array([A, B, C, D, Tmin, Tmax])

        # Check shapes
        self._shape = check_shapes([A, B, C, D], [Tmin, Tmax])

        # Check types
        check_type(unit, str, 'unit')
        check_type(symbol, str, 'symbol')

        self.A = A
        self.B = B
        self.C = C
        self.D = D
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
            f"Tmin [K]:  {self.Tmin}\n" \
            f"Tmax [K]:  {self.Tmax}"


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
        A, B, C, D, E, Tmin, Tmax = \
            convert_list_to_array([A, B, C, D, E, Tmin, Tmax])

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


class DIPPR102(DIPPRP4):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-102 equation.

    This equation implements the following temperature dependence:

    $$ Y = \frac{A T^B}{ 1 + C/T + D/T^2} $$

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
                 C: FloatOrArrayLike = 0.,
                 D: FloatOrArrayLike = 0.,
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
        return (A * T**B) / (1 + C/T + D/T**2)


class DIPPR104(DIPPRP5):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-104 equation.

    This equation implements the following temperature dependence:

    $$ Y = A + B/T + C/T^3 + D/T^8 + E/T^9 $$

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
        return A + B/T + C/T**3 + D/T**8 + E/T**9


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


class DIPPR106(DIPPR):
    r"""[DIPPR](https://de.wikipedia.org/wiki/DIPPR-Gleichungen)-106 equation.

    This equation implements the following temperature dependence:

    $$ Y = A (1 - T_r)^{B + C T_r + D T_r^2 + E T_r^3} $$

    where $A$ to $E$ are constant parameters, $T$ is the absolute temperature,
    $T_c$ is the critical temperature and $T_r = T/T_c$ is the reduced
    temperature.

    Parameters
    ----------
    Tc : FloatOrArrayLike
        Critical temperature.
        Unit = K.
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
                 Tc: FloatOrArrayLike,
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

        # Convert lists to arrays
        A, B, C, D, E, Tc, Tmin, Tmax = \
            convert_list_to_array([A, B, C, D, E, Tc, Tmin, Tmax])

        # Check shapes
        self._shape = check_shapes([A, B, C, D, E, Tc], [Tmin, Tmax])

        # Check types
        check_type(unit, str, 'unit')
        check_type(symbol, str, 'symbol')

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.Tc = Tc
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
            f"Tc [K]:    {self.Tc}\n" \
            f"Tmin [K]:  {self.Tmin}\n" \
            f"Tmax [K]:  {self.Tmax}"

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
        Tr = T/self.Tc
        return A * (1 - Tr) ** (B + Tr*(C + Tr*(D + E*Tr)))
