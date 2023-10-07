# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.utils import check_type, check_bounds, convert_check_temperature,\
    FloatOrArray, FloatOrArrayLike, IntOrArrayLike, IntOrArray
from polykin.kinetics.thermal import Arrhenius, Eyring

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Literal


__all__ = ['TerminationCompositeModel', 'PropagationHalfLength']


class KineticCoefficientCLD(ABC):
    """_Abstract_ chain-length dependent (CLD) coefficient."""

    name: str

    @abstractmethod
    def __call__(self, T, i, *args) -> FloatOrArray:
        pass

    @abstractmethod
    def eval(self, T, i, *args) -> FloatOrArray:
        pass


class TerminationCompositeModel(KineticCoefficientCLD):
    r"""Composite model for the termination rate coefficient between two
    radicals.

    This model implements the chain-length dependence proposed by
    [Smith & Russel (2003)](https://doi.org/10.1002/mats.200390029):

    $$ k_t(i,j)=\sqrt{k_t(i,i) k_t(j,j)} $$

    with:

    $$ k_t(i,i)=\begin{cases}
    k_t(1,1)i^{-\alpha_S}& \text{if } i \leq i_{crit} \\
    k_t(1,1)i_{crit}^{-(\alpha_S-\alpha_L)}i^{-\alpha_L} & \text{if }i>i_{crit}
    \end{cases}
    $$

    where $k_t(1,1)$ is the temperature-dependent termination rate coefficient
    between two monomeric radicals, $i_{crit}$ is the critical chain length,
    $\alpha_S$ is the short-chain exponent, and $\alpha_L$ is the long-chain
    exponent.

    Parameters
    ----------
    k11 : Arrhenius | Eyring
        Temperature-dependent termination rate coefficient between two
        monomeric radicals, $k_t(1,1)$.
    icrit : int
        Critical chain length, $i_{crit}$.
    aS : float
        Short-chain exponent, $\alpha_S$.
    aL : float
        Long-chain exponent, $\alpha_L$.
    name : str
        Name.
    """

    k11: Union[Arrhenius, Eyring]
    icrit: int
    aS: float
    aL: float
    symbol: str = 'k_t (i,j)'

    def __init__(self,
                 k11: Union[Arrhenius, Eyring],
                 icrit: int,
                 aS: float = 0.5,
                 aL: float = 0.2,
                 name: str = ''
                 ) -> None:

        check_type(k11, (Arrhenius, Eyring), 'k11')
        check_bounds(icrit, 1, 200, 'icrit')
        check_bounds(aS, 0, 1, 'alpha_short')
        check_bounds(aL, 0, 0.5, 'alpha_long')

        self.k11 = k11
        self.icrit = icrit
        self.aS = aS
        self.aL = aL
        self.name = name

    def __repr__(self) -> str:
        return (
            f"name:      {self.name}\n"
            f"icrit:     {self.icrit}\n"
            f"aS:        {self.aS}\n"
            f"aL:        {self.aL}\n"
            "- k11 -\n" + self.k11.__repr__()
        )

    def eval(self,
             T: FloatOrArray,
             i: IntOrArray,
             j: IntOrArray
             ) -> FloatOrArray:
        r"""Evaluate kinetic coefficient.

        Direct evaluation at given SI conditions, without unit conversions or
        checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        i : IntOrArray
            Chain length of 1st radical.
        j : IntOrArray
            Chain length of 2nd radical.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """

        k11 = self.k11.eval(T)
        aS = self.aS
        aL = self.aL
        icrit = self.icrit

        def ktii(i):
            return np.where(i <= icrit,
                            k11*i**(-aS),
                            k11*icrit**(-aS+aL)*i**(-aL))

        return np.sqrt(ktii(i)*ktii(j))

    def __call__(self,
                 T: FloatOrArrayLike,
                 i: IntOrArrayLike,
                 j: IntOrArrayLike,
                 Tunit: Literal['C', 'K'] = 'K'
                 ) -> FloatOrArray:
        r"""Evaluate kinetic coefficient.

        Evaluation at given conditions, including unit conversion and range
        check.

        Parameters
        ----------
        T : FloatOrArrayLike
            Temperature.
            Unit = `Tunit`.
        i : IntOrArrayLike
            Chain length of 1st radical.
        j : IntOrArrayLike
            Chain length of 2nd radical.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """
        TK = convert_check_temperature(T, Tunit, self.k11.Tmin, self.k11.Tmax)
        if isinstance(i, (list, tuple)):
            i = np.array(i, dtype=np.int32)
        if isinstance(j, (list, tuple)):
            j = np.array(j, dtype=np.int32)
        return self.eval(TK, i, j)


class PropagationHalfLength(KineticCoefficientCLD):
    r"""Half-length model for the decay of the propagation rate coefficient
    with chain length.

    This model implements the chain-length dependence proposed by
    [Smith et al. (2005)](https://doi.org/10.1016/j.eurpolymj.2004.09.002):

    $$ k_p(i) = k_p \left[ 1+ (C - 1)\exp{\left (-\frac{\ln 2}{i_{1/2}} (i-1) \
    \right )} \right] $$

    where $k_p=k_p(\infty)$ is the long-chain value of the propagation rate
    coefficient, $C\ge 1$ is the ratio $k_p(1)/k_p$ and $(i_{1/2}+1)$ is the
    hypothetical chain-length at which the difference $k_p(1) - k_p$ is halved.

    Parameters
    ----------
    kp : Arrhenius | Eyring
        Long-chain value of the propagation rate coefficient, $k_p$.
    C : float
        Ratio of the propagation coefficients of a monomeric radical and a
        long-chain radical, $C$.
    ihalf : float
        Half-length, $i_{i/2}$.
    name : str
        Name.
    """

    kp: Union[Arrhenius, Eyring]
    C: float
    ihalf: float
    symbol: str = 'k_p(i)'

    def __init__(self,
                 kp: Union[Arrhenius, Eyring],
                 C: float = 10.,
                 ihalf: float = 1.0,
                 name: str = ''
                 ) -> None:

        check_type(kp, (Arrhenius, Eyring), 'kp')
        check_bounds(C, 1., 100., 'C')
        check_bounds(ihalf, 0.1, 10, 'ihalf')

        self.kp = kp
        self.C = C
        self.ihalf = ihalf
        self.name = name

    def __repr__(self) -> str:
        return (
            f"name:      {self.name}\n"
            f"C:         {self.C}\n"
            f"ihalf:     {self.ihalf}\n"
            "- kp -\n" + self.kp.__repr__()
        )

    def eval(self,
             T: FloatOrArray,
             i: IntOrArray,
             ) -> FloatOrArray:
        r"""Evaluate kinetic coefficient.

        Direct evaluation at given SI conditions, without unit conversions or
        checks.

        Parameters
        ----------
        T : FloatOrArray
            Temperature.
            Unit = K.
        i : IntOrArray
            Chain length of radical.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """
        kp = self.kp.eval(T)
        C = self.C
        ihalf = self.ihalf

        return kp*(1 + (C - 1)*np.exp(-np.log(2)*(i - 1)/ihalf))

    def __call__(self,
                 T: FloatOrArrayLike,
                 i: IntOrArrayLike,
                 Tunit: Literal['C', 'K'] = 'K'
                 ) -> FloatOrArray:
        r"""Evaluate kinetic coefficient.

        Evaluation at given conditions, including unit conversion and range
        check.

        Parameters
        ----------
        T : FloatOrArrayLike
            Temperature.
            Unit = `Tunit`.
        i : IntOrArrayLike
            Chain length of radical.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        FloatOrArray
            Coefficient value.
        """
        TK = convert_check_temperature(T, Tunit, self.kp.Tmin, self.kp.Tmax)
        if isinstance(i, (list, tuple)):
            i = np.array(i, dtype=np.int32)
        return self.eval(TK, i)
