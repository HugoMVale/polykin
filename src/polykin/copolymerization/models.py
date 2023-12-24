# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector, FloatOrVectorLike, \
    FloatOrArray, FloatOrArrayLike
from polykin.utils import eps
from polykin.utils import check_bounds, check_in_set, custom_repr
from polykin.kinetics import Arrhenius

import numpy as np
from typing import Optional, Literal
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from abc import ABC, abstractmethod

__all__ = ['TerminalModel',
           'PenultimateModel',
           'ImplicitPenultimateModel']

# %% Models


class CopoModel(ABC):

    name: str
    M1: str
    M2: str
    _pnames: tuple[str, ...]

    def __init__(self,
                 M1: str,
                 M2: str,
                 name: str
                 ) -> None:
        """Construct `CopoModel` with the given parameters."""

        if M1 and M2 and M1.lower() != M2.lower():
            self.M1 = M1
            self.M2 = M2
        else:
            raise ValueError("`M1` and `M2` must be non-empty and different.")

        self.name = name

    def __repr__(self) -> str:
        return custom_repr(self, ('name', 'M1', 'M2') + self._pnames)

    @staticmethod
    def equation_F1(f1: FloatOrArray,
                    r1: FloatOrArray,
                    r2: FloatOrArray
                    ) -> FloatOrArray:
        r"""Instantaneous copolymer composition equation.

        For a binary system, the instantaneous copolymer composition is related
        to the comonomer composition by:

        $$ F_1=\frac{r_1 f_1^2 + f_1 f_2}{r_1 f_1^2 + 2 f_1 f_2 + r_2 f_2^2} $$

        where $F_i$ and $f_i$ are, respectively, the instantaneous copolymer
        and comonomer composition of monomer $i$, and $r_i$ are the
        reactivity ratios. Although the equation is written using terminal
        model notation, it is equally applicable in the frame of the
        penultimate model if $r_i \rightarrow \bar{r}_i$.

        Parameters
        ----------
        f1 : FloatOrArray
            Molar fraction of M1.
        r1 : FloatOrArray
            Reactivity ratio of M1, $r_1$ or $\bar{r}_1$.
        r2 : FloatOrArray
            Reactivity ratio of M2, $r_2$ or $\bar{r}_2$.

        Returns
        -------
        FloatOrArray
            Instantaneous copolymer composition, $F_1$.
        """
        f2 = 1 - f1
        return (r1*f1**2 + f1*f2)/(r1*f1**2 + 2*f1*f2 + r2*f2**2)

    @staticmethod
    def equation_kp(f1: FloatOrArray,
                    r1: FloatOrArray,
                    r2: FloatOrArray,
                    k11: FloatOrArray,
                    k22: FloatOrArray
                    ) -> FloatOrArray:
        r"""Average propagation rate coefficient equation.

        For a binary system, the instantaneous average propagation rate
        coefficient is related to the instantaneous comonomer composition by:

        $$ \bar{k}_p = \frac{r_1 f_1^2 + r_2 f_2^2 + 2f_1 f_2}
           {(r_1 f_1/k_{11}) + (r_2 f_2/k_{22})} $$

        where $f_i$ is the instantaneous comonomer composition of monomer $i$,
        $r_i$ are the reactivity ratios, and $k_{ii}$ are the homo-propagation
        rate coefficients. Although the equation is written using terminal
        model notation, it is equally applicable in the frame of the
        penultimate model if $r_i \rightarrow \bar{r}_i$ and
        $k_{ii} \rightarrow \bar{k}_{ii}$.

        Parameters
        ----------
        f1 : FloatOrArray
            Molar fraction of M1.
        r1 : float
            Reactivity ratio of M1, $r_1$ or $\bar{r}_1$.
        r2 : float
            Reactivity ratio of M2, $r_2$ or $\bar{r}_2$.
        k11 : float
            Propagation rate coefficient of M1, $k_{11}$ or $\bar{k}_{11}$.
            Unit = L/(mol·s)
        k22 : float
            Propagation rate coefficient of M2, $k_{22}$ or $\bar{k}_{22}$.
            Unit = L/(mol·s)

        Returns
        -------
        FloatOrArray
            Average propagation rate coefficient. Unit = L/(mol·s)
        """
        f2 = 1 - f1
        return (r1*f1**2 + r2*f2**2 + 2*f1*f2)/((r1*f1/k11) + (r2*f2/k22))

    @abstractmethod
    def ri(self, f1: FloatOrArray) -> tuple[FloatOrArray, FloatOrArray]:
        pass

    @abstractmethod
    def kii(self,
            f1: FloatOrArray,
            T: float,
            Tunit,
            ) -> tuple[FloatOrArray, FloatOrArray]:
        pass

    def F1(self,
           f1: FloatOrArrayLike
           ) -> FloatOrArray:
        r"""Calculate the instantaneous copolymer composition, $F_1$.

        Parameters
        ----------
        f1 : FloatOrArrayLike
            Molar fraction of M1.

        Returns
        -------
        FloatOrArray
            Instantaneous copolymer composition, $F_1$.
        """

        if isinstance(f1, (list, tuple)):
            f1 = np.array(f1, dtype=np.float64)
        check_bounds(f1, 0., 1., 'f1')
        return self.equation_F1(f1, *self.ri(f1))

    def kp(self,
           f1: FloatOrArrayLike,
           T: float,
           Tunit: Literal['C', 'K'] = 'K'
           ) -> FloatOrArray:
        r"""Calculate the average propagation rate coefficient equation,
        $\bar{k}_p$.

        Parameters
        ----------
        f1 : FloatOrArrayLike
            Molar fraction of M1.
        T : float
            Temperature. Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        FloatOrArray
            Average propagation rate coefficient. Unit = L/(mol·s)
        """

        if isinstance(f1, (list, tuple)):
            f1 = np.array(f1, dtype=np.float64)
        check_bounds(f1, 0., 1., 'f1')
        return self.equation_kp(f1, *self.ri(f1), *self.kii(f1, T, Tunit))

    @property
    def azeo(self) -> Optional[float]:
        r"""Calculate the azeotrope composition.

        Returns
        -------
        float | None
            If an azeotrope exists, it returns its composition in terms of
            $f_1$.
        """

        # Check if system is trivial
        ri_check = np.array(self.ri(0.1) + self.ri(0.9))
        if (np.all(np.isclose(ri_check, 1., atol=1e-2))):
            print("Warning: Trivial system with r1~r2~1.")
            return None

        def fzero(f1):
            return self.equation_F1(f1, *self.ri(f1)) - f1

        try:
            solution = root_scalar(f=fzero,
                                   bracket=(1e-4, 0.9999),
                                   method='brentq')
            if solution.converged:
                result = solution.root
            else:
                print(solution.flag)
                result = None
        except ValueError:
            result = None

        return result

    def drift(self,
              f10: FloatOrVectorLike,
              x: FloatOrVectorLike
              ) -> FloatVector:
        r"""Calculate drift of comonomer composition in a closed system for a
        given total monomer conversion.

        In a closed binary system, the drift in monomer composition is given by
        the solution of the following differential equation:

        $$ \frac{\textup{d} f_1}{\textup{d}x} = \frac{f_1 - F_1}{1 - x} $$

        with initial condition $f_1(0)=f_{1,0}$, where $f_1$ and $F_1$ are,
        respectively, the instantaneous comonomer and copolymer composition of
        M1, and $x$ is the total molar monomer conversion.

        Parameters
        ----------
        f10 : FloatOrVectorLike
            Initial molar fraction of M1, $f_{1,0}=f_1(0)$.
        x : FloatOrVectorLike
            Total molar monomer conversion.

        Returns
        -------
        FloatVector
            Monomer fraction of M1 at a given conversion, $f_1(x)$.
        """
        if isinstance(f10, (int, float)):
            f10 = [f10]
        if isinstance(x, (int, float)):
            x = [x]

        def df1dx(x, f1):
            return (f1 - self.equation_F1(f1, *self.ri(f1)))/(1 - x + eps)

        sol = solve_ivp(df1dx,
                        (0, max(x)),
                        f10,
                        method='LSODA',
                        t_eval=x,
                        vectorized=True,
                        rtol=1e-4)
        if sol.success:
            result = sol.y
            result = np.maximum(0, result)
            if result.shape[0] == 1:
                result = result[0]
        else:
            result = np.empty_like(x)
            result[:] = np.nan
            print(sol.message)

        return result

    def plot(self,
             kind: Literal['Mayo', 'kp', 'drift'] = 'Mayo',
             M: Literal[1, 2] = 1,
             f0: Optional[FloatOrVectorLike] = None,
             T: Optional[float] = None,
             Tunit: Literal['C', 'K'] = 'K',
             title: Optional[str] = None,
             axes: Optional[Axes] = None,
             return_objects: bool = False
             ) -> Optional[tuple[Optional[Figure], Axes]]:
        r"""Generate a plot of instantaneous copolymer composition, monomer
        composition drift, or average propagation rate coefficient.

        Parameters
        ----------
        kind : Literal['Mayo', 'kp', 'drift']
            Kind of plot to be generated.
        M : Literal[1, 2]
            Index of the monomer to be used in input argument `f0` and in
            output results. Specifically, if `M=i`, then `f0` stands for
            $f_i(0)$ and plots will be generated in terms of $f_i$ and $F_i$.
        f0 : FloatOrVectorLike | None
            Initial monomer composition, $f_i(0)$, as required for a monomer
            composition drift plot.
        T : float | None
            Temperature. Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.
        title : str | None
            Title of plot. If `None`, a default title with the monomer names
            will be used.
        axes : Axes | None
            Matplotlib Axes object.
        return_objects : bool
            If `True`, the Figure and Axes objects are returned (for saving or
            further manipulations).

        Returns
        -------
        tuple[Figure | None, Axes] | None
            Figure and Axes objects if return_objects is `True`.
        """
        check_in_set(M, {1, 2}, 'M')
        check_in_set(kind, {'Mayo', 'kp', 'drift'}, 'kind')

        label = None
        if axes is None:
            fig, ax = plt.subplots()
            if title is None:
                titles = {'Mayo': "Mayo-Lewis plot",
                          'drift': "Monomer composition drift",
                          'kp': "Average propagation coefficient"}
                title = titles[kind] + f" {self.M1}(1)-{self.M2}(2)"
            if title:
                fig.suptitle(title)
        else:
            ax = axes
            fig = None
            if self.name:
                label = self.name

        if kind == 'Mayo':

            ax.set_xlabel(fr"$f_{M}$")
            ax.set_ylabel(fr"$F_{M}$")
            ax.set_ylim(0, 1)

            ax.plot((0., 1.), (0., 1.), color='black', linewidth=0.5)

            x = np.linspace(0, 1, 1000)
            y = self.F1(x)
            if M == 2:
                x[:] = 1 - x
                y[:] = 1 - y  # type: ignore
            ax.plot(x, y, label=label)

        elif kind == 'drift':

            if f0 is None:
                raise ValueError("`f0` is required for a `drift` plot.")
            else:
                if isinstance(f0, (int, float)):
                    f0 = [f0]
                f0 = np.array(f0, dtype=np.float64)
                check_bounds(f0, 0., 1., 'f0')

            ax.set_xlabel("Total molar monomer conversion, " + r"$x$")
            ax.set_ylabel(fr"$f_{M}$")
            ax.set_ylim(0, 1)

            x = np.linspace(0, 1, 1000)
            if M == 2:
                f0[:] = 1 - f0
            y = self.drift(f0, x)
            if M == 2:
                y[:] = 1 - y
            if y.ndim == 1:
                y = y[np.newaxis, :]
            for i in range(len(f0)):
                ax.plot(x, y[i, :], label=label)

        elif kind == 'kp':

            if T is None:
                raise ValueError("`T` is required for a `kp` plot.")

            ax.set_xlabel(fr"$f_{M}$")
            ax.set_ylabel(r"$\bar{k}_p$")

            x = np.linspace(0, 1, 1000)
            y = self.kp(x, T, Tunit)
            if M == 2:
                y[:] = 1 - y  # type: ignore
            ax.plot(x, y, label=label)

        ax.set_xlim(0, 1)
        ax.grid(True)

        if axes is not None:
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        if return_objects:
            return (fig, ax)

# %% Terminal model


class TerminalModel(CopoModel):
    r"""Terminal binary copolymerization model.

    According to this model, the reactivity of a macroradical depends
    uniquely on the nature of the _terminal_ repeating unit. A binary system
    is, thus, described by four propagation reactions:

    $$
    \begin{matrix}
    P^{\bullet}_1 + M_1 \overset{k_{11}}{\rightarrow} P^{\bullet}_1 \\
    P^{\bullet}_1 + M_2 \overset{k_{12}}{\rightarrow} P^{\bullet}_2 \\
    P^{\bullet}_2 + M_1 \overset{k_{21}}{\rightarrow} P^{\bullet}_1 \\
    P^{\bullet}_2 + M_2 \overset{k_{22}}{\rightarrow} P^{\bullet}_2
    \end{matrix}
    $$

    where $k_{ii}$ are the homo-propagation rate coefficients and $k_{ij}$ are
    the cross-propagation coefficients. The two reactivity ratios are
    defined as $r_1=k_{11}/k_{12}$ and $r_2=k_{22}/k_{21}$.

    Parameters
    ----------
    r1 : float
        Reactivity ratio of M1, $r_1=k_{11}/k_{12}$.
    r2 : float
        Reactivity ratio of M2, $r_2=k_{22}/k_{21}$.
    k11 : Arrhenius | None
        Propagation rate coefficient of M1.
    k22 : Arrhenius | None
        Propagation rate coefficient of M2.
    M1 : str
        Name of M1.
    M2 : str
        Name of M2.
    name : str
        Name.
    """

    r1: float
    r2: float
    k11: Optional[Arrhenius]
    k22: Optional[Arrhenius]

    _pnames = ('r1', 'r2', 'k11', 'k22')

    def __init__(self,
                 r1: float,
                 r2: float,
                 k11: Optional[Arrhenius] = None,
                 k22: Optional[Arrhenius] = None,
                 M1: str = 'M1',
                 M2: str = 'M2',
                 name: str = ''
                 ) -> None:
        """Construct `TerminalCopoModel` with the given parameters."""

        check_bounds(r1, 0., np.inf, 'r1')
        check_bounds(r2, 0., np.inf, 'r2')

        # Perhaps this could be upgraded to exception, but I don't want to be
        # too restrictive (one does find literature data with (r1,r2)>1)
        if r1 > 1. and r2 > 1.:
            print(
                f"Warning: `r1`={r1} and `r2`={r2} are both greater than 1, which is deemed physically impossible.")

        self.r1 = r1
        self.r2 = r2
        self.k11 = k11
        self.k22 = k22
        super().__init__(M1, M2, name)

    @property
    def azeo(self) -> Optional[float]:
        r"""Calculate the azeotrope composition.

        An azeotrope (i.e., a point where $F_1=f_1$) only exists if both
        reactivity ratios are smaller than unity. In that case, the azeotrope
        composition is given by:

        $$ f_{1,azeo} = \frac{1 - r_2}{(2 - r_1 - r_2)} $$

        where $r_1$ and $r_2$ are the reactivity ratios.

        Returns
        -------
        float | None
            If an azeotrope exists, it returns its composition in terms of
            $f_1$.
        """
        r1 = self.r1
        r2 = self.r2
        if r1 < 1. and r2 < 1.:
            result = (1 - r2)/(2 - r1 - r2)
        else:
            result = None
        return result

    def ri(self, _) -> tuple[FloatOrArray, FloatOrArray]:
        return (self.r1, self.r2)

    def kii(self,
            _,
            T: float,
            Tunit,
            ) -> tuple[FloatOrArray, FloatOrArray]:
        if self.k11 is None or self.k22 is None:
            raise ValueError(
                "To use this feature, `k11` and `k22` cannot be `None`.")
        return (self.k11(T, Tunit), self.k22(T, Tunit))

# %% Penultimate model


class PenultimateModel(CopoModel):
    r"""Penultimate binary copolymerization model.

    According to this model, the reactivity of a macroradical depends on the
    nature of _penultimate_ and _terminal_ repeating units. A binary system
    is, thus, described by eight propagation reactions:

    $$
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
    $$

    where $k_{iii}$ are the homo-propagation rate coefficients and $k_{ijk}$
    are the cross-propagation coefficients. The four monomer reactivity ratios
    are defined as $r_{11}=k_{111}/k_{112}$, $r_{12}=k_{122}/k_{121}$,
    $r_{21}=k_{211}/k_{212}$ and $r_{22}=k_{222}/k_{221}$. The two radical
    reactivity ratios are defined as $s_1=k_{211}/k_{111}$ and
    $s_2=k_{122}/k_{222}$.

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
    k111 : Arrhenius | None
        Propagation rate coefficient of M1.
    k222 : Arrhenius | None
        Propagation rate coefficient of M2.
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
    k111: Optional[Arrhenius]
    k222: Optional[Arrhenius]

    _pnames = ('r11', 'r12', 'r21', 'r22', 's1', 's2', 'k111', 'k222')

    def __init__(self,
                 r11: float,
                 r12: float,
                 r21: float,
                 r22: float,
                 s1: float,
                 s2: float,
                 k111: Optional[Arrhenius] = None,
                 k222: Optional[Arrhenius] = None,
                 M1: str = 'M1',
                 M2: str = 'M2',
                 name: str = ''
                 ) -> None:
        """Construct `PenultimateModel` with the given parameters."""

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
        self.k111 = k111
        self.k222 = k222
        super().__init__(M1, M2, name)

    def ri(self,
            f1: FloatOrArray
           ) -> tuple[FloatOrArray, FloatOrArray]:
        r"""Pseudo-reactivity ratios.

        In the penultimate model, the pseudo-reactivity ratios depend on the
        instantaneous comonomer composition according to:

        $$ \begin{aligned}
           \bar{r}_1 &= r_{21}\frac{f_1 r_{11} + f_2}{f_1 r_{21} + f_2} \\
           \bar{r}_2 &= r_{12}\frac{f_2 r_{22} + f_1}{f_2 r_{12} + f_1}
        \end{aligned} $$

        where $r_{ij}$ are the monomer reactivity ratios.

        Parameters
        ----------
        f1 : FloatOrArray
            Molar fraction of M1.

        Returns
        -------
        tuple[FloatOrArray, FloatOrArray]
            Tuple of average reactivity ratios, ($\bar{r}_1$, $\bar{r}_2$).
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
            f1: FloatOrArray,
            T: float,
            Tunit,
            ) -> tuple[FloatOrArray, FloatOrArray]:
        r"""Pseudo-homopropagation rate coefficients.

        In the penultimate model, the pseudo-homopropagation rate coefficients
        on the instantaneous comonomer composition according to:

        $$ \begin{aligned}
        \bar{k}_{11} = k_{111} \frac{f_1 r_{11} + f_2}{f_1 r_{11} + f_2/s_1} \\
        \bar{k}_{22} = k_{222} \frac{f_2 r_{22} + f_1}{f_2 r_{22} + f_1/s_2}
        \end{aligned} $$

        where $r_{ij}$ are the monomer reactivity ratios, $s_i$ are the radical
        reactivity ratios, and $k_{iii}$ are the homopropagation rate
        coefficients.

        Parameters
        ----------
        f1 : FloatOrArray
            Molar fraction of M1.
        T : float
            Temperature. Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        tuple[FloatOrArray, FloatOrArray]
            Tuple of average propagation rate coefficients,
            ($\bar{k}_{11}$, $\bar{k}_{22}$).
        """
        if self.k111 is None or self.k222 is None:
            raise ValueError(
                "To use this feature, `k111` and `k222` cannot be `None`.")
        f2 = 1. - f1
        r11 = self.r11
        r22 = self.r22
        s1 = self.s1
        s2 = self.s2
        k111 = self.k111(T, Tunit)
        k222 = self.k222(T, Tunit)
        k11 = k111*(f1*r11 + f2)/(f1*r11 + f2/s1)
        k22 = k222*(f2*r22 + f1)/(f2*r22 + f1/s2)
        return (k11, k22)

# %%


class ImplicitPenultimateModel(TerminalModel):
    r"""Implicit penultimate binary copolymerization model.

    According to this model, the reactivity of a macroradical depends on the
    nature of _penultimate_ and _terminal_ repeating units. A binary system
    is, thus, described by eight propagation reactions:

    $$
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
    $$

    where $k_{iii}$ are the homo-propagation rate coefficients and $k_{ijk}$
    are the cross-propagation coefficients. In contrast to the full (explicit)
    penultimate model, the implicit version only has two monomer reactivity
    ratios, which are defined as $r_1=k_{111}/k_{112}=k_{211}/k_{212}$ and
    $r_2=k_{222}/k_{221}=k_{122}/k_{121}$. The two radical
    reactivity ratios are defined as $s_1=k_{211}/k_{111}$ and
    $s_2=k_{122}/k_{222}$.

    Parameters
    ----------
    r1 : float
        Monomer reactivity ratio, $r_1=k_{111}/k_{112}=k_{211}/k_{212}$.
    r21 : float
        Monomer reactivity ratio, $r_2=k_{222}/k_{221}=k_{122}/k_{121}$.
    s1 : float
        Radical reactivity ratio, $s_1=k_{211}/k_{111}$.
    s2 : float
        Radical reactivity ratio, $s_2=k_{122}/k_{222}$.
    k111 : Arrhenius | None
        Propagation rate coefficient of M1.
    k222 : Arrhenius | None
        Propagation rate coefficient of M2.
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
    k111: Optional[Arrhenius]
    k222: Optional[Arrhenius]

    _pnames = ('r1', 'r2', 's1', 's2', 'k111', 'k222')

    def __init__(self,
                 r1: float,
                 r2: float,
                 s1: float,
                 s2: float,
                 k111: Optional[Arrhenius] = None,
                 k222: Optional[Arrhenius] = None,
                 M1: str = 'M1',
                 M2: str = 'M2',
                 name: str = ''
                 ) -> None:
        """Construct `ImplicitPenultimateModel` with the given parameters."""

        check_bounds(r1, 0., np.inf, 'r1')
        check_bounds(r2, 0., np.inf, 'r2')
        check_bounds(s1, 0., np.inf, 's1')
        check_bounds(s2, 0., np.inf, 's2')

        # Perhaps this could be upgraded to exception, but I don't want to be
        # too restrictive (one does find literature data with (r1,r2)>1)
        if r1 > 1. and r2 > 1.:
            print(
                f"Warning: `r1`={r1} and `r2`={r2} are both greater than 1, which is deemed physically impossible.")

        self.r1 = r1
        self.r2 = r2
        self.s1 = s1
        self.s2 = s2
        self.k111 = k111
        self.k222 = k222
        super(TerminalModel, self).__init__(M1, M2, name)

    def kii(self,
            f1: FloatOrArray,
            T: float,
            Tunit,
            ) -> tuple[FloatOrArray, FloatOrArray]:
        r"""Pseudo-homopropagation rate coefficients.

        In the implicit penultimate model, the pseudo-homopropagation rate
        coefficients on the instantaneous comonomer composition according to:

        $$ \begin{aligned}
        \bar{k}_{11} = k_{111} \frac{f_1 r_1 + f_2}{f_1 r_1 + f_2/s_1} \\
        \bar{k}_{22} = k_{222} \frac{f_2 r_2 + f_1}{f_2 r_2 + f_1/s_2}
        \end{aligned} $$

        where $r_i$ are the monomer reactivity ratios, $s_i$ are the radical
        reactivity ratios, and $k_{iii}$ are the homopropagation rate
        coefficients.

        Parameters
        ----------
        f1 : FloatOrArray
            Molar fraction of M1.
        T : float
            Temperature. Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        tuple[FloatOrArray, FloatOrArray]
            Tuple of average propagation rate coefficients,
            ($\bar{k}_{11}$, $\bar{k}_{22}$).
        """
        if self.k111 is None or self.k222 is None:
            raise ValueError(
                "To use this feature, `k111` and `k222` cannot be `None`.")
        f2 = 1. - f1
        r1 = self.r1
        r2 = self.r2
        s1 = self.s1
        s2 = self.s2
        k111 = self.k111(T, Tunit)
        k222 = self.k222(T, Tunit)
        k11 = k111*(f1*r1 + f2)/(f1*r1 + f2/s1)
        k22 = k222*(f2*r2 + f1)/(f2*r2 + f1/s2)
        return (k11, k22)
