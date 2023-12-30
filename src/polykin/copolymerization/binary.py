# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from abc import ABC, abstractmethod
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from polykin.kinetics import Arrhenius
from polykin.types import (FloatOrArray, FloatOrArrayLike, FloatOrVectorLike,
                           FloatVector, IntOrArrayLike)
from polykin.utils import (check_bounds, check_in_set, convert_to_vector,
                           custom_repr, eps)

from .copodataset import CopoDataset
from .multicomponent import convert_Qe_to_r

__all__ = ['TerminalModel',
           'PenultimateModel',
           'ImplicitPenultimateModel']

# %% Models


class CopoModel(ABC):

    k1: Optional[Arrhenius]
    k2: Optional[Arrhenius]
    M1: str
    M2: str
    name: str
    data: dict[str, list[CopoDataset]]

    _pnames: tuple[str, ...]

    def __init__(self,
                 k1: Optional[Arrhenius],
                 k2: Optional[Arrhenius],
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

        self.k1 = k1
        self.k2 = k2
        self.name = name
        self.data = {kind: [] for kind in ('Mayo', 'drift', 'kp')}

    def __repr__(self) -> str:
        return custom_repr(self, ('name', 'M1', 'M2', 'k1', 'k2')
                           + self._pnames)

    @abstractmethod
    def ri(self,
           f1: FloatOrArray
           ) -> tuple[FloatOrArray, FloatOrArray]:
        """Return the evaluated reactivity ratios at the given conditions."""
        pass

    @abstractmethod
    def kii(self,
            f1: FloatOrArray,
            T: float,
            Tunit,
            ) -> tuple[FloatOrArray, FloatOrArray]:
        """Return the evaluated homopropagation rate coefficients at the given
        conditions."""
        pass

    def F1(self,
           f1: FloatOrArrayLike
           ) -> FloatOrArray:
        r"""Calculate the instantaneous copolymer composition, $F_1$.

        The calculation is handled by
        [`inst_copolymer_binary`](TerminalModel.md#polykin.copolymerization.models.inst_copolymer_binary).

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
        return inst_copolymer_binary(f1, *self.ri(f1))

    def kp(self,
           f1: FloatOrArrayLike,
           T: float,
           Tunit: Literal['C', 'K'] = 'K'
           ) -> FloatOrArray:
        r"""Calculate the average propagation rate coefficient, $\bar{k}_p$.

        The calculation is handled by
        [`average_kp_binary`](TerminalModel.md#polykin.copolymerization.models.average_kp_binary).

        !!! note

            This feature requires the attributes `k11` and `k22` to be defined.

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
        return average_kp_binary(f1, *self.ri(f1), *self.kii(f1, T, Tunit))

    @property
    def azeotrope(self) -> Optional[float]:
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
            return inst_copolymer_binary(f1, *self.ri(f1)) - f1

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
        f10, x = convert_to_vector([f10, x], False)

        def df1dx(x, f1):
            return (f1-inst_copolymer_binary(f1, *self.ri(f1)))/(1. - x + eps)

        sol = solve_ivp(df1dx,
                        (0., max(x)),
                        f10,
                        method='LSODA',
                        t_eval=x,
                        vectorized=True,
                        rtol=1e-4)
        if sol.success:
            result = sol.y
            result = np.maximum(0., result)
            if result.shape[0] == 1:
                result = result[0]
        else:
            result = np.empty_like(x)
            result[:] = np.nan
            print(sol.message)

        return result

    def plot(self,
             kind: Literal['drift', 'kp', 'Mayo'],
             show: Literal['auto', 'all', 'data', 'model'] = 'auto',
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
        kind : Literal['drift', 'kp', 'Mayo']
            Kind of plot to be generated.
        show : Literal['auto', 'all', 'data', 'model']
            What informatation is to be plotted.
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
        check_in_set(show, {'auto', 'all', 'data', 'model'}, 'show')

        label_model = None
        if axes is None:
            fig, ax = plt.subplots()
            if title is None:
                titles = {'Mayo': "Mayo-Lewis diagram",
                          'drift': "Monomer composition drift",
                          'kp': "Average propagation coefficient"}
                title = titles[kind] + f" {self.M1}(1)-{self.M2}(2)"
            if title:
                fig.suptitle(title)
        else:
            ax = axes
            fig = None
            if self.name:
                label_model = self.name

        unit_range = (0., 1.)
        npoints = 1000
        data = None

        if kind == 'Mayo':

            ax.set_xlabel(fr"$f_{M}$")
            ax.set_ylabel(fr"$F_{M}$")
            ax.set_xlim(*unit_range)
            ax.set_ylim(*unit_range)

            ax.plot(unit_range, unit_range, color='black', linewidth=0.5)

            if show in {'auto', 'all', 'model'}:
                x = np.linspace(*unit_range, npoints, dtype=np.float64)
                y = self.F1(x)
                if M == 2:
                    x[:] = 1. - x
                    y[:] = 1. - y  # type: ignore
                ax.plot(x, y, label=label_model)

            if show in {'auto', 'all', 'data'}:
                data = self.data[kind]
                for ds in data:
                    x = ds.x
                    y = ds.y
                    if (M == 2) == (self.M1 == ds.M1):
                        x[:] = 1. - x
                        y[:] = 1. - y
                    ax.scatter(x, y,
                               label=ds.name if ds.name else None)

        elif kind == 'drift':

            ax.set_xlabel(r"Total molar monomer conversion, $x$")
            ax.set_ylabel(fr"$f_{M}$")
            ax.set_xlim(*unit_range)
            ax.set_ylim(*unit_range)

            if show == 'model' or (show == 'auto' and f0 is not None):

                if f0 is None:
                    raise ValueError("`f0` is required for a `drift` plot.")
                else:
                    if isinstance(f0, (int, float)):
                        f0 = [f0]
                    f0 = np.array(f0, dtype=np.float64)
                    check_bounds(f0, *unit_range, 'f0')

                x = np.linspace(*unit_range, 1000)
                if M == 2:
                    f0[:] = 1. - f0
                y = self.drift(f0, x)
                if M == 2:
                    y[:] = 1. - y
                if y.ndim == 1:
                    y = y[np.newaxis, :]
                for i in range(y.shape[0]):
                    ax.plot(x, y[i, :], label=label_model)

            if show in {'data', 'all'}:

                data = self.data[kind]
                for ds in data:
                    x = ds.x
                    y = ds.y
                    if (M == 2) == (self.M1 == ds.M1):
                        y[:] = 1. - y
                    ax.scatter(x, y,
                               label=ds.name if ds.name else None)

                    if show == 'all':
                        x = np.linspace(*unit_range, npoints)
                        y0 = ds.y0
                        if self.M1 != ds.M1:
                            y0 = 1. - y0
                        y = self.drift(y0, x)
                        if M == 2:
                            y[:] = 1. - y
                        ax.plot(x, y, label=label_model)

        elif kind == 'kp':

            ax.set_xlabel(fr"$f_{M}$")
            ax.set_ylabel(r"$\bar{k}_p$")
            ax.set_xlim(*unit_range)

            if show in ('model') or (show == 'auto' and T is not None):

                if T is None:
                    raise ValueError("`T` is required for a `kp` plot.")

                x = np.linspace(*unit_range, npoints)
                y = self.kp(x, T, Tunit)
                if M == 2:
                    x[:] = 1. - x
                ax.plot(x, y, label=label_model)

            if show in {'all', 'data'}:

                data = self.data[kind]
                for ds in data:
                    x = ds.x
                    y = ds.y
                    if (M == 2) == (self.M1 == ds.M1):
                        x[:] = 1. - x
                    ax.scatter(x, y,
                               label=ds.name if ds.name else None)
                    if show == 'all':
                        x = np.linspace(*unit_range, npoints)
                        T = ds.T
                        y = self.kp(x, T, Tunit)
                        if M == 2:
                            x[:] = 1. - x
                        ax.plot(x, y, label=label_model)

        ax.grid(True)

        if axes is not None or data is not None:
            ax.legend(bbox_to_anchor=(1.05, 1.), loc="upper left")

        if return_objects:
            return (fig, ax)

    def add_data(self,
                 data: Union[CopoDataset, list[CopoDataset]]
                 ) -> None:
        r"""Add a copolymerization dataset for subsequent analysis.

        Parameters
        ----------
        data : Union[CopoDataset, list[CopoDataset]]
            Experimental dataset(s).
        """
        if not isinstance(data, list):
            data = [data]

        valid_monomers = {self.M1, self.M2}
        for ds in data:
            ds_monomers = {ds.M1, ds.M2}
            if ds_monomers != valid_monomers:
                raise ValueError(
                    f"Monomers defined in dataset `{ds.name}` are invalid: "
                    f"{ds_monomers}!={valid_monomers}")
            self.data[ds.kind].append(ds)

# %% Terminal model


class TerminalModel(CopoModel):
    r"""Terminal binary copolymerization model.

    According to this model, the reactivity of a macroradical depends
    uniquely on the nature of the _terminal_ repeating unit. A binary system
    is, thus, described by four propagation reactions:

    \begin{matrix}
    P^{\bullet}_1 + M_1 \overset{k_{11}}{\rightarrow} P^{\bullet}_1 \\
    P^{\bullet}_1 + M_2 \overset{k_{12}}{\rightarrow} P^{\bullet}_2 \\
    P^{\bullet}_2 + M_1 \overset{k_{21}}{\rightarrow} P^{\bullet}_1 \\
    P^{\bullet}_2 + M_2 \overset{k_{22}}{\rightarrow} P^{\bullet}_2
    \end{matrix}

    where $k_{ii}$ are the homopropagation rate coefficients and $k_{ij}$ are
    the cross-propagation coefficients. The two cross-propagation coefficients
    are specified via an equal number of reactivity ratios, defined as
    $r_1=k_{11}/k_{12}$ and $r_2=k_{22}/k_{21}$.

    Parameters
    ----------
    r1 : float
        Reactivity ratio of M1, $r_1=k_{11}/k_{12}$.
    r2 : float
        Reactivity ratio of M2, $r_2=k_{22}/k_{21}$.
    k1 : Arrhenius | None
        Homopropagation rate coefficient of M1, $k_1 \equiv k_{11}$.
    k2 : Arrhenius | None
        Homopropagation rate coefficient of M2, $k_2 \equiv k_{22}$.
    M1 : str
        Name of M1.
    M2 : str
        Name of M2.
    name : str
        Name.
    """

    r1: float
    r2: float

    _pnames = ('r1', 'r2')

    def __init__(self,
                 r1: float,
                 r2: float,
                 k1: Optional[Arrhenius] = None,
                 k2: Optional[Arrhenius] = None,
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
                f"Warning: `r1`={r1} and `r2`={r2} are both greater than 1, "
                "which is deemed physically impossible.")

        self.r1 = r1
        self.r2 = r2
        super().__init__(k1, k2, M1, M2, name)

    @classmethod
    def from_Qe(cls,
                Qe1: tuple[float, float],
                Qe2: tuple[float, float],
                k1: Optional[Arrhenius] = None,
                k2: Optional[Arrhenius] = None,
                M1: str = 'M1',
                M2: str = 'M2',
                name: str = ''
                ):
        r"""_summary_

        Parameters
        ----------
        Qe1 : tuple[float, float]
            _description_
        Qe2 : tuple[float, float]
            _description_
        k1 : Optional[Arrhenius], optional
            _description_, by default None
        k2 : Optional[Arrhenius], optional
            _description_, by default None
        M1 : str, optional
            _description_, by default 'M1'
        M2 : str, optional
            _description_, by default 'M2'
        name : str, optional
            _description_, by default ''

        Returns
        -------
        _type_
            _description_
        """
        r = convert_Qe_to_r([Qe1, Qe2])
        r1 = r[0, 1]
        r2 = r[1, 0]
        return cls(r1, r2, k1, k2, M1, M2, name)

    @property
    def azeotrope(self) -> Optional[float]:
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
            result = (1. - r2)/(2. - r1 - r2)
        else:
            result = None
        return result

    def ri(self, _) -> tuple[FloatOrArray, FloatOrArray]:
        return (self.r1, self.r2)

    def kii(self,
            _,
            T: float,
            Tunit: Literal['C', 'K'] = 'K'
            ) -> tuple[FloatOrArray, FloatOrArray]:
        if self.k1 is None or self.k2 is None:
            raise ValueError(
                "To use this feature, `k1` and `k2` cannot be `None`.")
        return (self.k1(T, Tunit), self.k2(T, Tunit))

    def transitions(self,
                    f1: FloatOrArrayLike
                    ) -> dict[str, FloatOrArray]:
        r"""Calculate the instantaneous transition probabilities.

        For a binary system, the transition probabilities are given by:

        \begin{aligned}
            P_{ii} &= \frac{r_i f_i}{r_i f_i + (1 - f_i)} \\
            P_{ij} &= 1 - P_{ii}
        \end{aligned}

        where $i,j=1,2, i \neq j$.

        Reference:

        * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 178.

        Parameters
        ----------
        f1 : FloatOrArrayLike
            Molar fraction of M1.

        Returns
        -------
        dict[str, FloatOrArray]
            Transition probabilities, {'11': $P_{11}$, '12': $P_{12}$, ... }.
        """

        if isinstance(f1, (list, tuple)):
            f1 = np.array(f1, dtype=np.float64)
        check_bounds(f1, 0., 1., 'f')

        f2 = 1. - f1
        r1 = self.r1
        r2 = self.r2
        P11 = r1*f1/(r1*f1 + f2)
        P22 = r2*f2/(r2*f2 + f1)
        P12 = 1. - P11
        P21 = 1. - P22

        result = {'11': P11,
                  '12': P12,
                  '21': P21,
                  '22': P22}

        return result

    def triads(self,
               f1: FloatOrArrayLike
               ) -> dict[str, FloatOrArray]:
        r"""Calculate the instantaneous triad fractions.

        For a binary system, the triad fractions are given by:

        \begin{aligned}
            F_{iii} &= P_{ii}^2 \\
            F_{iij} &= 2 P_{ii} P_{ij} \\
            F_{jij} &= P_{ij}^2
        \end{aligned}

        where $P_{ij}$ is the transition probability $i \rightarrow j$, which
        is a function of the monomer composition and the reactivity ratios.

        Reference:

        * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 179.

        Parameters
        ----------
        f1 : FloatOrArrayLike
            Molar fraction of M1.

        Returns
        -------
        dict[str, FloatOrArray]
            Triad fractions,
            {'111': $F_{111}$, '112': $F_{112}$, '212': $F_{212}$, ... }.
        """

        P11, P12, P21, P22 = self.transitions(f1).values()

        F111 = P11**2
        F112 = 2*P11*P12
        F212 = P12**2

        F222 = P22**2
        F221 = 2*P22*P21
        F121 = P21**2

        result = {'111': F111,
                  '112': F112,
                  '212': F212,
                  '222': F222,
                  '221': F221,
                  '121': F121}

        return result

    def sequence(self,
                 f1: FloatOrArrayLike,
                 k: Optional[IntOrArrayLike] = None,
                 ) -> dict[str, FloatOrArray]:
        r"""Calculate the instantaneous sequence length probability or the
        number-average sequence length.

        For a binary system, the probability of finding $k$ consecutive units
        of monomer $i$ in a chain is:

        $$ S_{i,k} = (1 - P_{ii})P_{ii}^{k-1} $$

        and the corresponding number-average sequence length is:

        $$ \bar{S}_i = \sum_k k S_{i,k} = \frac{1}{1 - P_{ii}} $$

        where $P_{ii}$ is the transition probability $i \rightarrow i$, which
        is a function of the monomer composition and the reactivity ratios.

        Reference:

        * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 177.

        Parameters
        ----------
        k : int | None
            Sequence length, i.e., number of consecutive units in a chain.
            If `None`, the number-average sequence length will be computed.
        f1 : FloatOrArrayLike
            Molar fraction of M1.

        Returns
        -------
        dict[str, FloatOrArray]
            If `k is None`, the number-average sequence lengths,
            {'1': $\bar{S}_1$, '2': $\bar{S}_2$}. Otherwise, the
            sequence probabilities, {'1': $S_{1,k}$, '2': $S_{2,k}$}.
        """

        P11, _, _, P22 = self.transitions(f1).values()

        if k is None:
            result = {str(i + 1): 1/(1 - P + eps)
                      for i, P in enumerate([P11, P22])}
        else:
            if isinstance(k, (list, tuple)):
                k = np.array(k, dtype=np.int32)
            result = {str(i + 1): (1. - P)*P**(k - 1)
                      for i, P in enumerate([P11, P22])}

        return result
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
        super().__init__(k1, k2, M1, M2, name)

    def ri(self,
            f1: FloatOrArray
           ) -> tuple[FloatOrArray, FloatOrArray]:
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
        f1 : FloatOrArray
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
            f1: FloatOrArray,
            T: float,
            Tunit: Literal['C', 'K'] = 'K',
            ) -> tuple[FloatOrArray, FloatOrArray]:
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
        f1 : FloatOrArray
            Molar fraction of M1.
        T : float
            Temperature. Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        tuple[FloatOrArray, FloatOrArray]
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
                    f1: FloatOrArrayLike
                    ) -> dict[str, FloatOrArray]:
        r"""Calculate the instantaneous transition probabilities.

        For a binary system, the transition probabilities are given by:

        \begin{aligned}
            P_{iii} &= \frac{r_{ii} f_i}{r_{ii} f_i + (1 - f_i)} \\
            P_{jii} &= \frac{r_{ji} f_i}{r_{ji} f_i + (1 - f_i)} \\
            P_{iij} &= 1 -  P_{iii} \\
            P_{jij} &= 1 -  P_{jii}
        \end{aligned}

        where $i,j=1,2, i \neq j$.

        Reference:

        * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 181.

        Parameters
        ----------
        f1 : FloatOrArrayLike
            Molar fraction of M1.

        Returns
        -------
        dict[str, FloatOrArray]
            Transition probabilities,
            {'111': $P_{111}$, '211': $P_{211}$, '121': $P_{121}$, ... }.
        """

        if isinstance(f1, (list, tuple)):
            f1 = np.array(f1, dtype=np.float64)
        check_bounds(f1, 0., 1., 'f')

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
               f1: FloatOrArrayLike
               ) -> dict[str, FloatOrArray]:
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

        Reference:

        * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 181.

        Parameters
        ----------
        f1 : FloatOrArrayLike
            Molar fraction of M1.

        Returns
        -------
        dict[str, FloatOrArray]
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
                 f1: FloatOrArrayLike,
                 k: Optional[IntOrArrayLike] = None,
                 ) -> dict[str, FloatOrArray]:
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

        Reference:

        * NA Dotson, R Galván, RL Laurence, and M Tirrel. Polymerization
        process modeling, Wiley, 1996, p. 180.

        Parameters
        ----------
        k : int | None
            Sequence length, i.e., number of consecutive units in a chain.
            If `None`, the number-average sequence length will be computed.
        f1 : FloatOrArrayLike
            Molar fraction of M1.

        Returns
        -------
        dict[str, FloatOrArray]
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
        """Construct `ImplicitPenultimateModel` with the given parameters."""

        check_bounds(s1, 0., np.inf, 's1')
        check_bounds(s2, 0., np.inf, 's2')

        self.s1 = s1
        self.s2 = s2
        super().__init__(r1, r2, k1, k2, M1, M2, name)

    def kii(self,
            f1: FloatOrArray,
            T: float,
            Tunit: Literal['C', 'K'] = 'K',
            ) -> tuple[FloatOrArray, FloatOrArray]:
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
        f1 : FloatOrArray
            Molar fraction of M1.
        T : float
            Temperature. Unit = `Tunit`.
        Tunit : Literal['C', 'K']
            Temperature unit.

        Returns
        -------
        tuple[FloatOrArray, FloatOrArray]
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

# %% Auxiliary functions


def inst_copolymer_binary(f1: FloatOrArray,
                          r1: FloatOrArray,
                          r2: FloatOrArray
                          ) -> FloatOrArray:
    r"""Instantaneous copolymer composition equation (aka Mayo-Lewis equation).

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


def average_kp_binary(f1: FloatOrArray,
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
