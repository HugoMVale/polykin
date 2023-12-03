# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.types import FloatVector, FloatVectorLike, FloatOrVectorLike, \
    FloatOrArray, FloatOrArrayLike
from polykin.utils import ShapeError, eps
from polykin import utils
from polykin.kinetics import Arrhenius

from dataclasses import dataclass
from operator import itemgetter
import numpy as np
from typing import Union, Optional, Literal, Any
from scipy.stats import linregress
from scipy.stats.distributions import t
from scipy.optimize import curve_fit, root_scalar
from scipy import odr
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse, Patch
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from abc import ABC, abstractmethod

__all__ = ['InstantaneousCopoData',
           'CopoFitResult',
           'TerminalModel',
           'PenultimateModel']

# %% Dataclasses


@dataclass(frozen=True)
class InstantaneousCopoData():
    r"""Binary instantaneous copolymerization dataclass.

    Container for binary instantaneous copolymerization data, $F_1$ vs $f_1$,
    as usually obtained from low-conversion experiments.

    Parameters
    ----------
    M1 : str
        Name of M1.
    M2 : str
        Name of M2.
    f1 : FloatVectorLike
        Vector of monomer molar composition, $f_1$.
    F1 : FloatVectorLike
        Vector of copolymer molar composition, $F_1$.
    sigma_f: float | FloatVectorLike
        Absolute standard deviation of $f_i$
        ($\sigma_{f_1} \equiv \sigma_{f_2}$).
    sigma_F: float | FloatVectorLike
        Absolute standard deviation of $F_i$
        ($\sigma_{F_1} \equiv \sigma_{F_2}$).
    name: str
        Name of dataset.
    reference: str
        Reference of dataset.

    """
    M1: str
    M2: str
    f1: FloatVectorLike
    F1: FloatVectorLike
    sigma_f: FloatOrVectorLike = 1e-3
    sigma_F: FloatOrVectorLike = 5e-2
    name: str = ''
    reference: str = ''

    def __post_init__(self):
        if not self.M1 or not self.M2 or self.M1.lower() == self.M2.lower():
            raise ValueError(
                "`M1` and `M2` must be non-empty and different.")
        for attr in ['f1', 'F1', 'sigma_f', 'sigma_F']:
            x = getattr(self, attr)
            if isinstance(x, list):
                x = np.array(x, dtype=np.float64)
                object.__setattr__(self, attr, x)
            utils.check_bounds(x, 0., 1., attr)
        if len(self.f1) != len(self.F1):
            raise ShapeError(
                "`f1` and `F1` must be vectors of the same length.")
        for attr in ['sigma_f', 'sigma_F']:
            x = getattr(self, attr)
            if isinstance(x, np.ndarray) and (len(x) != len(self.f1)):
                raise ShapeError(
                    f"`{attr}` must have the same length as `f1` and `F1`.")

    def __repr__(self):
        return (
            f"M1:        {self.M1}\n"
            f"M2:        {self.M2}\n"
            f"f1:        {self.f1}\n"
            f"F1:        {self.F1}\n"
            f"sigma_f:   {self.sigma_f}\n"
            f"sigma_F:   {self.sigma_F}\n"
            f"name:      {self.name}\n"
            f"reference: {self.reference}"
        )


@dataclass(frozen=True)
class CopoFitResult():
    """Something"""
    M1: str
    M2: str
    r1: Optional[float] = None
    r2: Optional[float] = None
    sigma_r1: Optional[float] = None
    sigma_r2: Optional[float] = None
    error95_r1: Optional[float] = None
    error95_r2: Optional[float] = None
    cov: Optional[Any] = None
    method: str = ''

    def __repr__(self):
        s1 = \
            f"method:     {self.method}\n" \
            f"M1:         {self.M1}\n" \
            f"M2:         {self.M2}\n" \
            f"r1:         {self.r1:.2E}\n" \
            f"r2:         {self.r2:.2E}\n"
        if self.sigma_r1 is not None:
            s2 = \
                f"sigma_r1:   {self.sigma_r1:.2E}\n" \
                f"sigma_r2:   {self.sigma_r2:.2E}\n" \
                f"error95_r1: {self.error95_r1:.2E}\n" \
                f"error95_r2: {self.error95_r2:.2E}\n" \
                f"cov:        {self.cov}\n"
        else:
            s2 = ""
        return s1 + s2

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
        string1 = (
            f"name: {self.name}\n"
            f"M1:   {self.M1}\n"
            f"M2:   {self.M2}"
        )
        string2 = ""
        for pname in self._pnames:
            string2 += "\n" + f"{pname}:" + " " * \
                (5 - len(pname)) + f"{getattr(self, pname)}"
        return string1 + string2

    @staticmethod
    def equation_F1(f1: FloatOrArray,
                    r1: FloatOrArray,
                    r2: FloatOrArray
                    ) -> FloatOrArray:
        r"""Instantaneous copolymer composition equation.

        For a binary system, the instantaneous copolymer composition is related
        to the comonomer composition by:

        $$ F_1=\frac{r_1 f_1^2 + f_1 f_2}{r_1 f_1^2 + 2 f_1 f_2 + r_2 f_2^2} $$

        where $F_1$ and $f_1$ are, respectively, the instantaneous copolymer
        and comonomer composition of M1, and $r_1$ and $r_2$ are the
        reactivity ratios (usually assumed composition independent).

        Parameters
        ----------
        f1 : FloatOrArray
            Molar fraction of M1.
        r1 : FloatOrArray
            Reactivity ratio of M1.
        r2 : FloatOrArray
            Reactivity ratio of M2.

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

        $$ \bar{k}_p = \frac{r_1 f_1^2 + r_2 f_2^2 + 2f_1 f_2}
           {(r_1 f_1/k_{11}) + (r_2 f_2/k_{22})} $$

        Parameters
        ----------
        f1 : FloatOrArray
            Molar fraction of M1.
        r1 : float
            Reactivity ratio of M1.
        r2 : float
            Reactivity ratio of M2.
        k11 : float
            Propagation rate coefficient of M1. Unit = L/(mol·s)
        k22 : float
            Propagation rate coefficient of M2. Unit = L/(mol·s)

        Returns
        -------
        FloatOrArray
            Average propagation rate coefficient. Unit = L/(mol·s)
        """
        f2 = 1 - f1
        return (r1*f1**2 + r2*f2**2 + 2*f1*f2)/((r1*f1/k11) + (r2*f2/k22))

    @abstractmethod
    def _ri(self, f1: FloatOrArray) -> tuple[FloatOrArray, FloatOrArray]:
        pass

    @abstractmethod
    def _kii(self,
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

        if isinstance(f1, list):
            f1 = np.array(f1, dtype=np.float64)
        utils.check_bounds(f1, 0., 1., 'f1')
        return self.equation_F1(f1, *self._ri(f1))

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

        if isinstance(f1, list):
            f1 = np.array(f1, dtype=np.float64)
        utils.check_bounds(f1, 0., 1., 'f1')
        return self.equation_kp(f1, *self._ri(f1), *self._kii(f1, T, Tunit))

    @property
    def azeo(self) -> Optional[float]:
        r"""Calculate the azeotrope composition.

        Returns
        -------
        float | None
            If an azeotrope exists, it returns its composition in terms of
            $f_1$.
        """

        def fzero(f1):
            return self.equation_F1(f1, *self._ri(f1)) - f1

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

        def df1dx(x_, f1_):
            return (f1_ - self.equation_F1(f1_, *self._ri(f1_)))/(1 - x_ + eps)

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
            $f_{i,0}$ and plots will be generated in terms of $f_i$ and $F_i$.
        f0 : FloatOrVectorLike | None
            Initial monomer composition, as required for a monomer composition
            drift $f_i(x)$ plot.
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
        utils.check_in_set(M, {1, 2}, 'M')
        utils.check_in_set(kind, {'Mayo', 'kp', 'drift'}, 'kind')

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
                utils.check_bounds(f0, 0., 1., 'f0')

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

        utils.check_bounds(r1, 0., np.inf, 'r1')
        utils.check_bounds(r2, 0., np.inf, 'r2')
        self.r1 = r1
        self.r2 = r2
        self.k11 = k11
        self.k22 = k22

        # Perhaps this could be upgraded to exception, but I don't want to be
        # too restrictive (one does find literature data with (r1,r2)>1)
        if r1 > 1. and r2 > 1.:
            print(
                f"Warning: `r1`={r1} and `r2`={r2} are both greater than 1, which is deemed physically impossible.")

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

    def _ri(self,
            f1: FloatOrArray
            ) -> tuple[FloatOrArray, FloatOrArray]:
        return (self.r1, self.r2)

    def _kii(self,
             f1: FloatOrArray,
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
    reactivity ratios are as defined as $s_1=k_{211}/k_{111}$ and
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

        utils.check_bounds(r11, 0., np.inf, 'r11')
        utils.check_bounds(r12, 0., np.inf, 'r12')
        utils.check_bounds(r21, 0., np.inf, 'r21')
        utils.check_bounds(r22, 0., np.inf, 'r22')
        utils.check_bounds(s1, 0., np.inf, 's1')
        utils.check_bounds(s2, 0., np.inf, 's2')
        self.r11 = r11
        self.r12 = r12
        self.r21 = r21
        self.r22 = r22
        self.s1 = s1
        self.s2 = s2
        self.k111 = k111
        self.k222 = k222
        super().__init__(M1, M2, name)

    def _ri(self,
            f1: FloatOrArray
            ) -> tuple[FloatOrArray, FloatOrArray]:
        f2 = 1. - f1
        r11 = self.r11
        r12 = self.r12
        r21 = self.r21
        r22 = self.r22
        r1 = r21*(f1*r11 + f2)/(f1*r21 + f2)
        r2 = r12*(f2*r22 + f1)/(f2*r12 + f1)
        return (r1, r2)

    def _kii(self,
             f1: FloatOrArray,
             T: float,
             Tunit,
             ) -> tuple[FloatOrArray, FloatOrArray]:
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


# %% Analysis


class CopoAnalysis():

    def __init__(self,
                 data: Union[InstantaneousCopoData, list[InstantaneousCopoData]],
                 M1: Union[str, None] = None,
                 M2: Union[str, None] = None
                 ) -> None:

        # Recast data input as list and check content
        if not isinstance(data, list):
            data = [data]
        self.data = utils.check_type(data, InstantaneousCopoData, 'data',
                                     check_inside=True)

        # Define M1, M2
        if M1 and M2:
            if M1.lower() != M2.lower():
                self.M1 = M1
                self.M2 = M2
            else:
                raise ValueError(
                    f"M1='{M1}' and M2='{M2}' must be different monomers.")
        elif not (M1 or M2):
            self.M1 = data[0].M1
            self.M2 = data[0].M2
        else:
            raise ValueError(
                "`M1` and `M2` must be both defined or both undefined.")

        # Check monomer consistency in datasets
        valid_monomers = {self.M1, self.M2}
        for ds in self.data:
            ds_monomers = {ds.M1, ds.M2}
            if valid_monomers != ds_monomers:
                raise ValueError(
                    f"Monomers defined in dataset `{ds.name}` are invalid: {ds_monomers}!={valid_monomers}")

        self._data_fit = {}

    def plot(self,
             M: Literal[1, 2] = 1,
             title: Union[str, None] = None,
             axes: Union[Axes, None] = None,
             show: Literal['data', 'fit', 'all'] = 'all'
             ) -> Union[Figure, None]:

        if axes is None:
            fig, ax = plt.subplots()
            if title is None:
                title = f"Mayo plot {self.M1}(1)-{self.M2}(2)"
            if title:
                fig.suptitle(title)
        else:
            ax = axes
            fig = None

        if M == 1:
            Mx = self.M1
        elif M == 2:
            Mx = self.M2
        else:
            raise ValueError("Monomer index `M` must be 1 or 2.")

        ax.set_xlabel(fr"$f_{M}$")
        ax.set_ylabel(fr"$F_{M}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.plot((0, 1), (0, 1), color='black', linewidth=0.5, label=None)

        if show in {'data', 'all'}:
            for ds in self.data:
                x = ds.f1
                y = ds.F1
                if Mx == ds.M2:
                    x[:] = 1 - x
                    y[:] = 1 - y
                ax.scatter(x, y, label=ds.name)
        if show in {'fit', 'all'}:
            x = np.linspace(0, 1, 100
                            )
            pass

        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        return fig

    def fit(self,
            method: Literal['FR', 'NLLS', 'ODR'] = 'NLLS',
            alpha: float = 0.05,
            plot: bool = True
            ) -> CopoFitResult:

        method_names = {'FR': 'Finemann-Ross',
                        'NLLS': 'Non-linear least squares',
                        'ODR': 'Orthogonal distance regression'}

        # Concatenate all datasets and save to cache
        if not self._data_fit:
            f1 = np.array([])
            F1 = np.array([])
            sigma_f = np.array([])
            sigma_F = np.array([])
            for ds in self.data:
                ds_f1 = ds.f1
                ds_F1 = ds.F1
                npoints = len(ds_f1)
                if ds.M1 == self.M2 and ds.M2 == self.M1:
                    ds_f1 = 1 - ds_f1
                    ds_F1 = 1 - ds_F1
                f1 = np.concatenate([f1, ds_f1])
                F1 = np.concatenate([F1, ds_F1])
                if isinstance(ds.sigma_f, float):
                    ds_sigma_f = np.full(npoints, ds.sigma_f)
                else:
                    ds_sigma_f = ds.sigma_f
                if isinstance(ds.sigma_F, float):
                    ds_sigma_F = np.full(npoints, ds.sigma_F)
                else:
                    ds_sigma_F = ds.sigma_F
                sigma_f = np.concatenate([sigma_f, ds_sigma_f])
                sigma_F = np.concatenate([sigma_F, ds_sigma_F])

            # Remove invalid f, F values
            idx_valid = np.logical_and.reduce((f1 > 0, f1 < 1, F1 > 0, F1 < 1))
            f1 = f1[idx_valid]
            F1 = F1[idx_valid]
            sigma_f = sigma_f[idx_valid]
            sigma_F = sigma_F[idx_valid]

            # Store in cache
            self._data_fit.update({'f1': f1,
                                   'F1': F1,
                                   'sigma_f': sigma_f,
                                   'sigma_F': sigma_F})
        else:
            f1, F1, sigma_f, sigma_F = \
                itemgetter('f1', 'F1', 'sigma_f', 'sigma_F')(self._data_fit)

        # Finemann-Ross (either for itself or as initial guess for other methods)
        x, y = f1/(1 - f1), F1/(1 - F1)
        x, y = -y/x**2, (y - 1)/x
        solution = linregress(x, y)
        _r1, _r2 = solution.intercept, solution.slope  # type: ignore

        r1 = None
        r2 = None
        sigma_r1 = None
        sigma_r2 = None
        cov = None
        error95_r1 = None
        error95_r2 = None

        if method == 'FR':
            r1 = _r1
            r2 = _r2

        elif method == 'NLLS':
            solution = curve_fit(F1_inst,
                                 xdata=f1,
                                 ydata=F1,
                                 p0=(_r1, _r2),
                                 sigma=sigma_F,
                                 absolute_sigma=True,
                                 bounds=(0, np.inf),
                                 full_output=True)
            if solution[4]:
                r1, r2 = solution[0]
                cov = solution[1]
                # This next part is to be checked
                sigma_r1, sigma_r2 = np.sqrt(np.diag(cov))
                tval = t.ppf(1 - alpha/2, max(0, f1.size - cov.shape[0]))
                error95_r1 = sigma_r1*tval
                error95_r2 = sigma_r2*tval
            else:
                print("Fit error: ", solution[3])

        elif method == 'ODR':
            odr_Model = odr.Model(lambda beta, x: F1_inst(x, *beta))
            odr_Data = odr.RealData(x=f1, y=F1, sx=sigma_f, sy=sigma_F)
            odr_ODR = odr.ODR(odr_Data, odr_Model, beta0=(_r1, _r2))
            solution = odr_ODR.run()
            r1, r2 = solution.beta
            cov = np.array(solution.cov_beta)  # !!! not sure
            # This next part is to be checked + finished
            sigma_r1, sigma_r2 = solution.sd_beta
            error95_r1 = sigma_r1
            error95_r2 = sigma_r2

        else:
            utils.check_in_set(method, set(method_names.keys()), 'method')

        # Pack results into object
        result = CopoFitResult(M1=self.M1,
                               M2=self.M2,
                               r1=r1,
                               r2=r2,
                               sigma_r1=sigma_r1,
                               sigma_r2=sigma_r2,
                               error95_r1=error95_r1,
                               error95_r2=error95_r2,
                               cov=cov,
                               method=method_names[method])
        if r1 is not None and r2 is not None and cov is not None:
            draw_jcr(r1, r2, cov, alpha)
        return result


def draw_jcr(r1: float,
             r2: float,
             cov: np.ndarray,
             alpha: float = 0.05):

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$r_1$")
    ax.set_ylabel(r"$r_2$")
    ax.scatter(r1, r2, c='black', s=5)
    confidence_ellipse((r1, r2), cov, ax)
    return


def confidence_ellipse(center: tuple[float, float],
                       cov: np.ndarray,
                       ax: plt.Axes,
                       nstd: float = 1.96
                       ) -> Patch:

    pearson = cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])
    radius_x = np.sqrt(1 + pearson)
    radius_y = np.sqrt(1 - pearson)
    scale_x, scale_y = np.sqrt(np.diag(cov))*nstd

    ellipse = Ellipse((0, 0),
                      width=2*radius_x,
                      height=2*radius_y,
                      facecolor='none',
                      edgecolor='black')

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(*center)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
