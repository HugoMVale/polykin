# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np

from polykin.kinetics import Arrhenius
from polykin.utils.math import eps
from polykin.utils.tools import check_bounds
from polykin.utils.types import FloatArray, FloatArrayLike, IntArrayLike

from .terminal import CopoModel, TerminalModel

__all__ = ['PenultimateModel',
           'ImplicitPenultimateModel']

# %% Models

# %% Penultimate model


class PenultimateModel(CopoModel):
    r"""Penultimate binary copolymerization model.

    According to this model, the reactivity of a macroradical depends on the
    nature of the _penultimate_ and _terminal_ repeating units. A binary system
    is, thus, described by eight propagation reactions:

    \begin{matrix}
    P^{\bullet}_{11} + M_1 \overset{k_{111}}{\rightarrow} P^{\bullet}_{11} \\
    P^{\bullet}_{11} + M_2 \overset{k_{112}}{\rightarrow} P^{\bullet}_{12} \\
    P^{\bullet}_{12} + M_1 \overset{k_{121}}{\rightarrow} P^{\bullet}_{21} \\
    P^{\bullet}_{12} + M_2 \overset{k_{122}}{\rightarrow} P^{\bullet}_{22} \\
    P^{\bullet}_{21} + M_1 \overset{k_{211}}{\rightarrow} P^{\bullet}_{11} \\
    P^{\bullet}_{21} + M_2 \overset{k_{212}}{\rightarrow} P^{\bullet}_{12} \\
    P^{\bullet}_{22} + M_1 \overset{k_{221}}{\rightarrow} P^{\bullet}_{21} \\
    P^{\bullet}_{22} + M_2 \overset{k_{222}}{\rightarrow} P^{\bullet}_{22} \\
    \end{matrix}

    where $k_{iii}$ are the homo-propagation rate coefficients and $k_{ijk}$
    are the cross-propagation coefficients. The six cross-propagation
    coefficients are specified via an equal number of reactivity ratios, which
    are divided in two categories. There are four monomer reactivity ratios,
    defined as $r_{11}=k_{111}/k_{112}$, $r_{12}=k_{122}/k_{121}$,
    $r_{21}=k_{211}/k_{212}$ and $r_{22}=k_{222}/k_{221}$. Additionally, there
    are two radical reactivity ratios defined as $s_1=k_{211}/k_{111}$ and
    $s_2=k_{122}/k_{222}$. The latter influence the average propagation rate
    coefficient, but have no effect on the copolymer composition.

    Parameters
    ----------
    r11 : float
        Monomer reactivity ratio, $r_{11}=k_{111}/k_{112}$.
    r12 : float
        Monomer reactivity ratio, $r_{12}=k_{122}/k_{121}$.
    r21 : float
        Monomer reactivity ratio, $r_{21}=k_{211}/k_{212}$.
    r22 : float
        Monomer reactivity ratio, $r_{22}=k_{222}/k_{221}$.
    s1 : float
        Radical reactivity ratio, $s_1=k_{211}/k_{111}$.
    s2 : float
        Radical reactivity ratio, $s_2=k_{122}/k_{222}$.
    k1 : Arrhenius | None
        Homopropagation rate coefficient of M1, $k_1 \equiv k_{111}$.
    k2 : Arrhenius | None
        Homopropagation rate coefficient of M2, $k_2 \equiv k_{222}$.
    M1 : str
        Name of M1.
    M2 : str
        Name of M2.
    name : str
        Name.
    """

    r11: float
    r12: float
    r21: float
    r22: float
    s1: float
    s2: float

    _pnames = ('r11', 'r12', 'r21', 'r22', 's1', 's2')

    def __init__(self,
                 r11: float,
                 r12: float,
                 r21: float,
                 r22: float,
                 s1: float,
                 s2: float,
                 k1: Optional[Arrhenius] = None,
                 k2: Optional[Arrhenius] = None,
                 M1: str = 'M1',
                 M2: str = 'M2',
                 name: str = ''
                 ) -> None:

        check_bounds(r11, 0., np.inf, 'r11')
        check_bounds(r12, 0., np.inf, 'r12')
        check_bounds(r21, 0., np.inf, 'r21')
        check_bounds(r22, 0., np.inf, 'r22')
        check_bounds(s1, 0., np.inf, 's1')
        check_bounds(s2, 0., np.inf, 's2')

        self.r11 = r11
        self.r12 = r12
        self.r21 = r21
        self.r22 = r22
        self.s1 = s1
        self.s2 = s2
        super().__init__(k1, k2, M1, M2, name)

    def ri(self,
            f1: Union[float, FloatArray]
           ) -> tuple[Union[float, FloatArray], Union[float, FloatArray]]:
        r"""Pseudoreactivity ratios.

        In the penultimate model, the pseudoreactivity ratios depend on the
        instantaneous comonomer composition according to:

        \begin{aligned}
           \bar{r}_1 &= r_{21}\frac{f_1 r_{11} + f_2}{f_1 r_{21} + f_2} \\
           \bar{r}_2 &= r_{12}\frac{f_2 r_{22} + f_1}{f_2 r_{12} + f_1}
        \end{aligned}

        where $r_{ij}$ are the monomer reactivity ratios.

        Parameters
        ----------
        f1 : float | FloatArray
            Molar fraction of M1.

        Returns
        -------
        tuple[FloatOrArray, FloatOrArray]
            Pseudoreactivity ratios, ($\bar{r}_1$, $\bar{r}_2$).
        """
        f2 = 1. - f1
        r11 = self.r11
        r12 = self.r12
        r21 = self.r21
        r22 = self.r22
        r1 = r21*(f1*r11 + f2)/(f1*r21 + f2)
        r2 = r12*(f2*r22 + f1)/(f2*r12 + f1)
        return (r1, r2)

    def kii(self,
            f1: Union[float, FloatArray],
            T: float,
            Tunit: Literal['C', 'K'] = 'K',
            ) -> tuple[Union[float, FloatArray], Union[float, FloatArray]]:
        r"""Pseudohomopropagation rate coefficients.

        In the penultimate model, the pseudohomopropagation rate coefficients
        depend on the instantaneous comonomer composition according to:

        \begin{aligned}
        \bar{k}_{11} &= k_{111}\frac{f_1 r_{11} + f_2}{f_1 r_{11} + f_2/s_1} \\
        \bar{k}_{22} &= k_{222}\frac{f_2 r_{22} + f_1}{f_2 r_{22} + f_1/s_2}
        \end{aligned}

        where $r_{ij}$ are the monomer reactivity ratios, $s_i$ are the radical
        reactivity ratios, and $k_{iii}$ are the homopropagation rate
        coefficients.

        Parameters
        ----------
        f1 : float | FloatArray
            Molar fraction of M1.
        T : float
            Temperature. Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        tuple[float | FloatArray, float | FloatArray]
            Pseudohomopropagation rate coefficients,
            ($\bar{k}_{11}$, $\bar{k}_{22}$).
        """
        if self.k1 is None or self.k2 is None:
            raise ValueError(
                "To use this feature, `k1` and `k2` cannot be `None`.")
        f2 = 1. - f1
        r11 = self.r11
        r22 = self.r22
        s1 = self.s1
        s2 = self.s2
        k1 = self.k1(T, Tunit)
        k2 = self.k2(T, Tunit)
        k11 = k1*(f1*r11 + f2)/(f1*r11 + f2/s1)
        k22 = k2*(f2*r22 + f1)/(f2*r22 + f1/s2)
        return (k11, k22)

    def transitions(self,
                    f1: Union[float, FloatArrayLike]
                    ) -> dict[str, Union[float, FloatArray]]:
        r"""Calculate the instantaneous transition probabilities.

        For a binary system, the transition probabilities are given by:

        \begin{aligned}
            P_{iii} &= \frac{r_{ii} f_i}{r_{ii} f_i + (1 - f_i)} \\
            P_{jii} &= \frac{r_{ji} f_i}{r_{ji} f_i + (1 - f_i)} \\
            P_{iij} &= 1 -  P_{iii} \\
            P_{jij} &= 1 -  P_{jii}
        \end{aligned}

        where $i,j=1,2, i \neq j$.

        **References**

        * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 181.

        Parameters
        ----------
        f1 : float | FloatArrayLike
            Molar fraction of M1.

        Returns
        -------
        dict[str, float | FloatArray]
            Transition probabilities,
            {'111': $P_{111}$, '211': $P_{211}$, '121': $P_{121}$, ... }.
        """

        if isinstance(f1, (list, tuple)):
            f1 = np.array(f1, dtype=np.float64)
        check_bounds(f1, 0., 1., 'f1')

        f2 = 1. - f1
        r11 = self.r11
        r12 = self.r12
        r21 = self.r21
        r22 = self.r22

        P111 = r11*f1/(r11*f1 + f2)
        P211 = r21*f1/(r21*f1 + f2)
        P112 = 1. - P111
        P212 = 1. - P211

        P222 = r22*f2/(r22*f2 + f1)
        P122 = r12*f2/(r12*f2 + f1)
        P221 = 1. - P222
        P121 = 1. - P122

        result = {'111': P111,
                  '211': P211,
                  '112': P112,
                  '212': P212,
                  '222': P222,
                  '122': P122,
                  '221': P221,
                  '121': P121}

        return result

    def triads(self,
               f1: Union[float, FloatArrayLike],
               ) -> dict[str, Union[float, FloatArray]]:
        r"""Calculate the instantaneous triad fractions.

        For a binary system, the triad fractions are given by:

        \begin{aligned}
            F_{iii} &\propto  P_{jii} \frac{P_{iii}}{1 - P_{iii}} \\
            F_{iij} &\propto 2 P_{jii} \\
            F_{jij} &\propto 1 - P_{jii}
        \end{aligned}

        where $P_{ijk}$ is the transition probability
        $i \rightarrow j \rightarrow k$, which is a function of the monomer
        composition and the reactivity ratios.

        **References**

        * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 181.

        Parameters
        ----------
        f1 : float | FloatArrayLike
            Molar fraction of M1.

        Returns
        -------
        dict[str, float | FloatArray]
            Triad fractions,
            {'111': $F_{111}$, '112': $F_{112}$, '212': $F_{212}$, ... }.
        """

        P111, P211, _, _, P222, P122, _, _ = self.transitions(f1).values()

        F111 = P211*P111/(1. - P111 + eps)
        F112 = 2*P211
        F212 = 1. - P211
        Fsum = F111 + F112 + F212
        F111 /= Fsum
        F112 /= Fsum
        F212 /= Fsum

        F222 = P122*P222/(1. - P222 + eps)
        F221 = 2*P122
        F121 = 1. - P122
        Fsum = F222 + F221 + F121
        F222 /= Fsum
        F221 /= Fsum
        F121 /= Fsum

        result = {'111': F111,
                  '112': F112,
                  '212': F212,
                  '222': F222,
                  '221': F221,
                  '121': F121}

        return result

    def sequence(self,
                 f1: Union[float, FloatArrayLike],
                 k: Optional[Union[int, IntArrayLike]] = None,
                 ) -> dict[str, Union[float, FloatArray]]:
        r"""Calculate the instantaneous sequence length probability or the
        number-average sequence length.

        For a binary system, the probability of finding $k$ consecutive units
        of monomer $i$ in a chain is:

        $$ S_{i,k} = \begin{cases}
            1 - P_{jii} & \text{if } k = 1 \\
            P_{jii}(1 - P_{iii})P_{iii}^{k-2}& \text{if } k \ge 2
        \end{cases} $$

        and the corresponding number-average sequence length is:

        $$ \bar{S}_i = \sum_k k S_{i,k} = 1 + \frac{P_{jii}}{1 - P_{iii}} $$

        where $P_{ijk}$ is the transition probability
        $i \rightarrow j \rightarrow k$, which is a function of the monomer
        composition and the reactivity ratios.

        **References**

        * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 180.

        Parameters
        ----------
        f1 : float | FloatArrayLike
            Molar fraction of M1.
        k : int | IntArrayLike | None
            Sequence length, i.e., number of consecutive units in a chain.
            If `None`, the number-average sequence length will be computed.

        Returns
        -------
        dict[str, float | FloatArray]
            If `k is None`, the number-average sequence lengths,
            {'1': $\bar{S}_1$, '2': $\bar{S}_2$}. Otherwise, the
            sequence probabilities, {'1': $S_{1,k}$, '2': $S_{2,k}$}.
        """

        P111, P211, _, _, P222, P122, _, _ = self.transitions(f1).values()

        if k is None:
            S1 = 1. + P211/(1. - P111 + eps)
            S2 = 1. + P122/(1. - P222 + eps)
        else:
            if isinstance(k, (list, tuple)):
                k = np.array(k, dtype=np.int32)
            S1 = np.where(k == 1, 1. - P211, P211*(1. - P111)*P111**(k - 2))
            S2 = np.where(k == 1, 1. - P122, P122*(1. - P222)*P222**(k - 2))

        return {'1': S1, '2': S2}
# %% Implicit penultimate model


class ImplicitPenultimateModel(TerminalModel):
    r"""Implicit penultimate binary copolymerization model.

    This model is a special case of the general (explicit) penultimate model,
    with a smaller number of independent parameters. As in the explicit
    version, the reactivity of a macroradical depends on the nature of the
    _penultimate_ and _terminal_ repeating units. A binary system is, thus,
    described by eight propagation reactions:

    \begin{matrix}
    P^{\bullet}_{11} + M_1 \overset{k_{111}}{\rightarrow} P^{\bullet}_{11} \\
    P^{\bullet}_{11} + M_2 \overset{k_{112}}{\rightarrow} P^{\bullet}_{12} \\
    P^{\bullet}_{12} + M_1 \overset{k_{121}}{\rightarrow} P^{\bullet}_{21} \\
    P^{\bullet}_{12} + M_2 \overset{k_{122}}{\rightarrow} P^{\bullet}_{22} \\
    P^{\bullet}_{21} + M_1 \overset{k_{211}}{\rightarrow} P^{\bullet}_{11} \\
    P^{\bullet}_{21} + M_2 \overset{k_{212}}{\rightarrow} P^{\bullet}_{12} \\
    P^{\bullet}_{22} + M_1 \overset{k_{221}}{\rightarrow} P^{\bullet}_{21} \\
    P^{\bullet}_{22} + M_2 \overset{k_{222}}{\rightarrow} P^{\bullet}_{22} \\
    \end{matrix}

    where $k_{iii}$ are the homo-propagation rate coefficients and $k_{ijk}$
    are the cross-propagation coefficients. The six cross-propagation
    coefficients are specified via just four reactivity ratios, which
    are divided in two categories. There are two monomer reactivity
    ratios, which are defined as $r_1=k_{111}/k_{112}=k_{211}/k_{212}$ and
    $r_2=k_{222}/k_{221}=k_{122}/k_{121}$. Additionally, there
    are two radical reactivity ratios defined as $s_1=k_{211}/k_{111}$ and
    $s_2=k_{122}/k_{222}$. The latter influence the average propagation rate
    coefficient, but have no effect on the copolymer composition.

    Parameters
    ----------
    r1 : float
        Monomer reactivity ratio, $r_1=k_{111}/k_{112}=k_{211}/k_{212}$.
    r2 : float
        Monomer reactivity ratio, $r_2=k_{222}/k_{221}=k_{122}/k_{121}$.
    s1 : float
        Radical reactivity ratio, $s_1=k_{211}/k_{111}$.
    s2 : float
        Radical reactivity ratio, $s_2=k_{122}/k_{222}$.
    k1 : Arrhenius | None
        Homopropagation rate coefficient of M1, $k_1 \equiv k_{111}$.
    k2 : Arrhenius | None
        Homopropagation rate coefficient of M2, $k_2 \equiv k_{222}$.
    M1 : str
        Name of M1.
    M2 : str
        Name of M2.
    name : str
        Name.
    """

    r1: float
    r2: float
    s1: float
    s2: float

    _pnames = ('r1', 'r2', 's1', 's2')

    def __init__(self,
                 r1: float,
                 r2: float,
                 s1: float,
                 s2: float,
                 k1: Optional[Arrhenius] = None,
                 k2: Optional[Arrhenius] = None,
                 M1: str = 'M1',
                 M2: str = 'M2',
                 name: str = ''
                 ) -> None:

        check_bounds(s1, 0., np.inf, 's1')
        check_bounds(s2, 0., np.inf, 's2')

        self.s1 = s1
        self.s2 = s2
        super().__init__(r1, r2, k1, k2, M1, M2, name)

    def kii(self,
            f1: Union[float, FloatArray],
            T: float,
            Tunit: Literal['C', 'K'] = 'K',
            ) -> tuple[Union[float, FloatArray], Union[float, FloatArray]]:
        r"""Pseudo-homopropagation rate coefficients.

        In the implicit penultimate model, the pseudohomopropagation rate
        coefficients depend on the instantaneous comonomer composition
        according to:

        \begin{aligned}
        \bar{k}_{11} &= k_{111} \frac{f_1 r_1 + f_2}{f_1 r_1 + f_2/s_1} \\
        \bar{k}_{22} &= k_{222} \frac{f_2 r_2 + f_1}{f_2 r_2 + f_1/s_2}
        \end{aligned}

        where $r_i$ are the monomer reactivity ratios, $s_i$ are the radical
        reactivity ratios, and $k_{iii}$ are the homopropagation rate
        coefficients.

        Parameters
        ----------
        f1 : float | FloatArray
            Molar fraction of M1.
        T : float
            Temperature. Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        tuple[float | FloatArray, float | FloatArray]
            Tuple of pseudohomopropagation rate coefficients,
            ($\bar{k}_{11}$, $\bar{k}_{22}$).
        """
        if self.k1 is None or self.k2 is None:
            raise ValueError(
                "To use this feature, `k1` and `k2` cannot be `None`.")
        f2 = 1. - f1
        r1 = self.r1
        r2 = self.r2
        s1 = self.s1
        s2 = self.s2
        k1 = self.k1(T, Tunit)
        k2 = self.k2(T, Tunit)
        k11 = k1*(f1*r1 + f2)/(f1*r1 + f2/s1)
        k22 = k2*(f2*r2 + f1)/(f2*r2 + f1/s2)
        return (k11, k22)
