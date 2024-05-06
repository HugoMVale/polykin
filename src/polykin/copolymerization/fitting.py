# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from numpy import dot
from scipy import odr
from scipy.optimize import minimize
from scipy.stats import linregress
from scipy.stats.distributions import t as tdist

from polykin.copolymerization.binary import (inst_copolymer_binary,
                                             monomer_drift_binary)
from polykin.copolymerization.copodataset import (CopoDataset_Ff,
                                                  CopoDataset_fx,
                                                  CopoDataset_Fx)
from polykin.math import confidence_ellipse, confidence_region, hessian2
from polykin.utils.exceptions import FitError
from polykin.utils.tools import pprint_matrix
from polykin.utils.types import Float2x2Matrix, FloatVectorLike

__all__ = ['fit_Finemann_Ross',
           'fit_copo_data',
           'CopoFitResult']


# %% CopoFitResult

@dataclass
class CopoFitResult():
    r"""Dataclass for copolymerization fit results.

    Attributes
    ----------
    method : str
        Name of the fit method.
    r1 : float
        Reactivity ratio of M1.
    r2: float
        Reactivity ratio of M2
    alpha : float
        Significance level.
    ci_r1 : float
        Confidence interval of r1.
    ci_r2: float
        Confidence interval of r2.
    se_r1 : float
        Standard error of r1.
    se_r2 : float
        Standard error of r2.
    cov : Float2x2Matrix
        Scaled variance-covariance matrix.
    plots : dict[str, tuple[Figure, Axes]]
        Dictionary of plots.
    M1 : str
        Name of M1.
    M2 : str
        Name of M2.
    """

    method: str
    r1: float
    r2: float
    alpha: float
    ci_r1: float
    ci_r2: float
    se_r1: float
    se_r2: float
    cov: Optional[Float2x2Matrix]
    plots: dict[str, tuple[Figure, Axes]]
    M1: str = 'M1'
    M2: str = 'M2'

    def __repr__(self):
        s1 = \
            f"method:  {self.method}\n" \
            f"M1:      {self.M1}\n" \
            f"M2:      {self.M2}\n" \
            f"r1:      {self.r1:.2E}\n" \
            f"r2:      {self.r2:.2E}\n"
        if self.se_r1 is not None:
            s2 = \
                f"alpha:   {self.alpha:.2f}\n" \
                f"se_r1:   {self.se_r1:.2E}\n" \
                f"se_r2:   {self.se_r2:.2E}\n" \
                f"ci_r1:   {self.ci_r1:.2E}\n" \
                f"ci_r2:   {self.ci_r2:.2E}\n"
        else:
            s2 = ""
        if self.cov is not None:
            s3 = f"cov:     {pprint_matrix(self.cov, nspaces=9)}\n"
        else:
            s3 = "\n"
        return s1 + s2 + s3


# %% Fit functions

def fit_Finemann_Ross(f1: FloatVectorLike,
                      F1: FloatVectorLike
                      ) -> tuple[float, float]:
    r"""Fit binary copolymer composition data using the Finemann-Ross method.

    $$ \left(\frac{x(y-1)}{y}\right) = -r_2 + r_1 \left(\frac{x^2}{y}\right) $$

    where $x = f_1/(1 - f_1)$, $y = F_1/(1 - F_1)$, $r_i$ are the reactivity
    ratios, $f_1$ is the monomer composition, and $F_1$ is the instantaneous
    copolymer composition.

    **Reference**

    *   Fineman, M.; Ross, S. D. J. Polymer Sci. 1950, 5, 259.

    Note
    ----
    The Finemann-Ross method relies on a linearization procedure that can lead
    to significant statistical bias. The method is provided for its historical
    significance, but should no longer be used for fitting reactivity ratios.


    Parameters
    ----------
    f1 : FloatVectorLike
        Vector of molar fraction of M1, $f_1$.
    F1 : FloatVectorLike
        Vector of instantaneous copolymer composition of M1, $F_1$.

    Returns
    -------
    tuple[float, float]
        Point estimates of the reactivity ratios $(r_1, r_2)$.

    See also
    --------
    * [`fit_copo_data`](fit_copo_data.md): alternative
      (recommended) method.  

    Examples
    --------
    >>> from polykin.copolymerization.fitting import fit_Finemann_Ross
    >>>
    >>> f1 = [0.186, 0.299, 0.527, 0.600, 0.700, 0.798]
    >>> F1 = [0.196, 0.279, 0.415, 0.473, 0.542, 0.634]
    >>>
    >>> r1, r2 = fit_Finemann_Ross(f1, F1)
    >>> print(f"r1={r1:.2f}, r2={r2:.2f}")
    r1=0.27, r2=0.84

    """

    f1 = np.asarray(f1)
    F1 = np.asarray(F1)

    # Variable transformation
    x = f1/(1. - f1)
    y = F1/(1. - F1)
    H = x**2/y
    G = x*(y - 1.)/y

    # Linear regression
    solution = linregress(H, G)
    r1 = solution.slope  # type: ignore
    r2 = - solution.intercept  # type: ignore

    return (r1, r2)


def fit_copo_data(data_Ff: list[CopoDataset_Ff] = [],
                  data_fx: list[CopoDataset_fx] = [],
                  data_Fx: list[CopoDataset_Fx] = [],
                  r_guess: tuple[float, float] = (1.0, 1.0),
                  method: Literal['NLLS', 'ODR'] = 'NLLS',
                  alpha: float = 0.05,
                  plot_data: bool = True,
                  JCR_linear: bool = True,
                  JCR_exact: bool = False,
                  JCR_npoints: int = 200,
                  JCR_rtol: float = 1e-2
                  ) -> CopoFitResult:
    r"""Fit copolymer composition data and estimate reactivity ratios.

    This function employs rigorous nonlinear algorithms to estimate the
    reactivity ratios from experimental copolymer composition data of type
    $F(f)$, $f(x;f_0)$, and $F(x,f_0)$. 

    The fitting is performed using one of two methods: nonlinear least squares
    (NLLS) or orthogonal distance regression (ODR). NLLS considers only
    observational errors in the dependent variable, whereas ODR takes into
    account observational errors in both the dependent and independent
    variables. Although the ODR method is statistically more general, it is
    also more complex and can (at present) only be used for fitting $F(f)$
    data. Whenever composition drift data is provided, NLLS must be utilized.

    The joint confidence region (JCR) of the reactivity ratios is generated
    using approximate (linear) and/or exact methods. In most cases, the linear
    method should be sufficiently accurate. Nonetheless, for these types of
    fits, the exact method is computationally inexpensive, making it perhaps a
    preferable choice.

    **Reference**

    *   Van Herk, A.M. and Dröge, T. (1997), Nonlinear least squares fitting
        applied to copolymerization modeling. Macromol. Theory Simul.,
        6: 1263-1276.
    *   Boggs, Paul T., et al. "Algorithm 676: ODRPACK: software for weighted
        orthogonal distance regression." ACM Transactions on Mathematical
        Software (TOMS) 15.4 (1989): 348-364.

    Parameters
    ----------
    data_Ff : list[CopoDataset_Ff]
        F(f) instantaneous composition datasets.
    data_fx : list[CopoDataset_fx]
        f(x) composition drift datasets.
    data_Fx : list[CopoDataset_Fx]
        F(x) composition drift datasets
    r_guess : tuple[float, float]
        Initial guess for the reactivity ratios.
    method : Literal['NLLS', 'ODR']
        Optimization method. `NLLS` for nonlinear least squares or `ODR` for
        orthogonal distance regression. The `ODR` method is only available for
        F(f) data.
    alpha : float
        Significance level.
    plot_data : bool
        If `True`, comparisons between experimental and fitted data will be
        plotted.
    JCR_linear : bool, optional
        If `True`, the JCR will be computed using the linear method.
    JCR_exact : bool, optional
        If `True`, the JCR will be computed using the exact method.
    JCR_npoints : int
        Number of points where the JCR is evaluated. The computational effort
        increases linearly with `npoints`.
    JCR_rtol : float
        Relative tolerance for the determination of the JCR. The default value
        (1e-2) should be adequate in most cases, as it implies a 1% accuracy in
        the JCR coordinates. 

    Returns
    -------
    CopoFitResult
        Dataclass with all fit results.

    See also
    --------
    * [`confidence_ellipse`](../math/confidence_ellipse.md): linear method
      used to calculate the joint confidence region.
    * [`confidence_region`](../math/confidence_region.md): exact method
      used to calculate the joint confidence region.
    * [`fit_Finemann_Ross`](fit_Finemann_Ross.md): alternative method.  

    """

    # Calculate and check 'ndata'
    npar = 2
    ndata = 0
    for dataset in data_Ff:
        ndata += len(dataset.f1)
    for dataset in data_fx:
        ndata += len(dataset.x)
    for dataset in data_Fx:
        ndata += len(dataset.x)
    if ndata < npar:
        raise FitError("Not enough data to estimate reactivity ratios.")

    # Check method choice
    if method == 'ODR' and (len(data_fx) > 0 or len(data_Fx) > 0):
        raise FitError("ODR method not implemented for drift data.")

    # Fit data
    if method == 'NLLS':
        r_opt, cov, sse = _fit_copo_NLLS(data_Ff, data_fx, data_Fx, r_guess,
                                         ndata)
    elif method == 'ODR':
        r_opt, cov, sse = _fit_copo_ODR(data_Ff, r_guess)
    else:
        raise ValueError(f"Method {method} is not valid.")

    # Standard parameter errors and confidence intervals
    if ndata > npar and cov is not None:
        se_r = np.sqrt(np.diag(cov))
        ci_r = se_r*tdist.ppf(1. - alpha/2, ndata - npar)
    else:
        se_r = [np.nan, np.nan]
        ci_r = [np.nan, np.nan]

    # Plot data vs model
    plots = {}
    if plot_data:
        xmax = 1.
        npoints = 500
        # Plot F(f) data
        if data_Ff:
            plots['Ff'] = plt.subplots()
            ax = plots['Ff'][1]
            ax.set_xlabel(r"$f_1$")
            ax.set_ylabel(r"$F_1$")
            ax.set_xlim(0., 1.)
            ax.set_ylim(0., 1.)
            for dataset in data_Ff:
                ax.scatter(dataset.f1, dataset.F1, label=dataset.name)
            f1 = np.linspace(0., 1., npoints)
            F1_est = inst_copolymer_binary(f1, *r_opt)
            ax.plot(f1, F1_est)
            ax.legend(loc="best")

        # Plot f(x) data
        if data_fx:
            plots['fx'] = plt.subplots()
            ax = plots['fx'][1]
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$f_1$")
            ax.set_xlim(0., 1.)
            ax.set_ylim(0., 1.)
            x = np.linspace(0., xmax, npoints)
            for dataset in data_fx:
                ax.scatter(dataset.x, dataset.f1, label=dataset.name)
                f1_est = monomer_drift_binary(dataset.f10, x, *r_opt)
                ax.plot(x, f1_est)
            ax.legend(loc="best")

        # Plot F(x) data
        if data_Fx:
            plots['Fx'] = plt.subplots()
            ax = plots['Fx'][1]
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$F_1$")
            ax.set_xlim(0., 1.)
            ax.set_ylim(0., 1.)
            x = np.linspace(0., xmax, npoints)
            for dataset in data_Fx:
                f10 = dataset.f10
                ax.scatter(dataset.x, dataset.F1, label=dataset.name)
                f1_est = monomer_drift_binary(f10, x, *r_opt)
                F1_est = f1_est + (f10 - f1_est)/(x + 1e-10)
                F1_est[0] = inst_copolymer_binary(f10, *r_opt)
                ax.plot(x, F1_est)
            ax.legend(loc="best")

        # Parity plot
        plots['parity'] = plt.subplots()
        ax = plots['parity'][1]
        ax.set_xlabel("Observed value")
        ax.set_ylabel("Predicted value")
        ax.set_xlim(0., 1.)
        ax.set_ylim(0., 1.)
        ax.plot([0., 1.], [0., 1.], color='black', linewidth=0.5)
        for dataset in data_Ff:
            F1_est = inst_copolymer_binary(dataset.f1, *r_opt)
            ax.scatter(dataset.F1, F1_est, label=dataset.name)
        for dataset in data_fx:
            f1_est = monomer_drift_binary(dataset.f10, dataset.x, *r_opt)
            ax.scatter(dataset.f1, f1_est, label=dataset.name)
        for dataset in data_Fx:
            f1_est = monomer_drift_binary(dataset.f10, dataset.x, *r_opt)
            F1_est = f1_est + (dataset.f10 - f1_est)/(dataset.x + 1e-10)
            ax.scatter(dataset.F1, F1_est, label=dataset.name)
        ax.legend(loc="best")

    # Joint confidence region
    if (JCR_linear or JCR_exact) and cov is not None:

        plots['JCR'] = plt.subplots()
        ax = plots['JCR'][1]
        ax.set_xlabel(r"$r_1$")
        ax.set_ylabel(r"$r_2$")
        colors = ['tab:blue', 'tab:orange']
        idx = 0

        if JCR_linear:
            confidence_ellipse(ax,
                               center=r_opt,
                               cov=cov,
                               ndata=ndata,
                               alpha=alpha,
                               color=colors[idx], label='linear JCR')
            idx += 1

        if JCR_exact:
            jcr = confidence_region(center=r_opt,
                                    sse=sse,
                                    ndata=ndata,
                                    alpha=alpha,
                                    width=2*ci_r[0],
                                    rtol=JCR_rtol,
                                    npoints=JCR_npoints)

            # ax.scatter(r1, r2, c=colors[idx], s=5)
            ax.plot(*jcr, color=colors[idx], label='exact JCR')

        ax.legend(loc="best")

    result = CopoFitResult(method=method,
                           r1=r_opt[0], r2=r_opt[1],
                           alpha=alpha,
                           ci_r1=ci_r[0], ci_r2=ci_r[1],
                           se_r1=se_r[0], se_r2=se_r[1],
                           cov=cov,
                           plots=plots)

    return result


def _fit_copo_NLLS(data_Ff: list[CopoDataset_Ff],
                   data_fx: list[CopoDataset_fx],
                   data_Fx: list[CopoDataset_Fx],
                   r_guess: tuple[float, float],
                   ndata: int):
    """Fit copolymerization data using NLLS."""

    def sse(r: tuple[float, float], atol=1e-4, rtol=1e-4) -> float:
        "Total sum of squared errors, for optimizer and exact JCR."
        result = 0.
        # F1(f1) datasets
        for dataset in data_Ff:
            F1_est = inst_copolymer_binary(dataset.f1, *r)
            rx = (dataset.F1 - F1_est)/dataset.scale_F1
            result += dataset.weight*dot(rx, rx)

        # f1(x, f10) datasets
        for dataset in data_fx:
            f1_est = monomer_drift_binary(dataset.f10, dataset.x, *r,
                                          atol=atol, rtol=rtol)
            rx = (dataset.f1 - f1_est)/dataset.scale_f1
            result += dataset.weight*dot(rx, rx)

        # F1(x, f10) datasets
        for dataset in data_Fx:
            f10 = dataset.f10
            x = dataset.x
            f1_est = monomer_drift_binary(f10, x, *r,
                                          atol=atol, rtol=rtol)
            F1_est = f1_est + (f10 - f1_est)/(x + 1e-10)
            rx = (dataset.F1 - F1_est)/dataset.scale_F1
            result += dataset.weight*dot(rx, rx)
        return result

    # Parameter estimation
    sol = minimize(sse,
                   x0=r_guess,
                   bounds=((1e-3, 1e2), (1e-3, 1e2)),
                   method='L-BFGS-B',  # most efficient
                   options={'maxiter': 200})
    if not sol.success:
        raise FitError(sol.message)
    r_opt = sol.x
    sse_opt = sol.fun

    # Covarance matrix
    npar = 2
    if ndata > npar:
        s_sq = sse_opt/(ndata - npar)
        H = hessian2(sse, r_opt, h=1e-4)
        Hinv = np.linalg.inv(H)
        cov = 2*Hinv*s_sq
    else:
        cov = None

    def sse_NLLS(r):
        return sse(r, rtol=1e-4, atol=1e-5)

    return (r_opt, cov, sse_NLLS)


def _fit_copo_ODR(data_Ff: list[CopoDataset_Ff],
                  r_guess: tuple[float, float]):
    """Fit copolymerization data using ODR."""

    # Concatenate all F(f) datasets
    f1, F1, sf1, sF1 = [], [], [], []
    for dataset in data_Ff:
        f1.extend(dataset.f1)
        F1.extend(dataset.F1)
        _sf1 = dataset.scale_f1/(dataset.weight + 1e-10)
        if isinstance(_sf1, (int, float)):
            _sf1 = [_sf1]*len(dataset.f1)
        sf1.extend(_sf1)
        _sF1 = dataset.scale_F1/(dataset.weight + 1e-10)
        if isinstance(_sF1, (int, float)):
            _sF1 = [_sF1]*len(dataset.f1)
        sF1.extend(_sF1)
    f1 = np.asarray(f1)
    F1 = np.asarray(F1)
    sf1 = np.asarray(sf1)
    sF1 = np.asarray(sF1)

    # Parameter estimation
    odr_Model = odr.Model(lambda beta, x: inst_copolymer_binary(x, *beta))
    odr_Data = odr.RealData(x=f1, y=F1, sx=sf1, sy=sF1)
    odr_ODR = odr.ODR(odr_Data, odr_Model, beta0=r_guess)
    solution = odr_ODR.run()

    if solution.info > 4:  # type: ignore
        raise FitError(solution.stopreason)

    r_opt = solution.beta
    f1plus = solution.xplus  # type: ignore
    if f1.size > 2:
        # cov_beta is absolute, so rescaling is required
        cov = solution.cov_beta*solution.res_var  # type: ignore
    else:
        cov = None

    def sse_ODR(r):
        "Total sum of squared errors, for exact JCR."
        ry = (F1 - inst_copolymer_binary(f1plus, *r))/sF1
        rx = (f1 - f1plus)/sf1
        return dot(ry, ry) + dot(rx, rx)

    return (r_opt, cov, sse_ODR)
