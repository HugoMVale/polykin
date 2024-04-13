# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from scipy import odr
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.stats.distributions import t as tdist

from polykin.copolymerization.binary import inst_copolymer_binary
from polykin.math import confidence_ellipse, confidence_region
from polykin.utils.exceptions import FitError
from polykin.utils.math import (convert_FloatOrVectorLike_to_FloatOrVector,
                                convert_FloatOrVectorLike_to_FloatVector)
from polykin.utils.tools import check_bounds, check_shapes
from polykin.utils.types import (Float2x2Matrix, FloatVectorLike)

__all__ = ['fit_Finemann_Ross',
           'fit_reactivity_ratios']


# %% CopoFitResult


@dataclass
class CopoFitResult():
    r"""Dataclass for copolymerization fit results.

    Parameters
    ----------
    r1 : float
        Reactivity ratio of M1.
    r2: float
        Reactivity ratio of M2
    cov : Float2x2Matrix
        Scaled variance-covariance matrix.
    se_r1 : float
        Standard error of r1.
    se_r2 : float
        Standard error of r2.
    ci_r1 : float
        Confidence interval of r1.
    ci_r2: float
        Confidence interval of r2.
    alpha : float
        Significance level.
    method : str
        Name of the fit method.
    M1 : str
        Name of M1.
    M2 : str
        Name of M2.
    Mayo : tuple[Figure, Axes] | None
        Mayo-Lewis plot with experimental data and fitted curve.
    JCR : tuple[Figure, Axes] | None
        Joint confidence region of reactivity ratios.
    """
    r1: float
    r2: float
    cov: Float2x2Matrix
    se_r1: float
    se_r2: float
    ci_r1: float
    ci_r2: float
    alpha: float
    method: str
    M1: str = 'M1'
    M2: str = 'M2'
    Mayo: Optional[tuple[Figure, Axes]] = None
    JCR: Optional[tuple[Figure, Axes]] = None

    def __repr__(self):
        s1 = \
            f"method:     {self.method}\n" \
            f"M1:         {self.M1}\n" \
            f"M2:         {self.M2}\n" \
            f"r1:         {self.r1:.2E}\n" \
            f"r2:         {self.r2:.2E}\n"
        if self.se_r1 is not None:
            s2 = \
                f"alpha:   {self.alpha:.2f}\n" \
                f"se_r1:   {self.se_r1:.2E}\n" \
                f"se_r2:   {self.se_r2:.2E}\n" \
                f"ci_r1:   {self.ci_r1:.2E}\n" \
                f"ci_r2:   {self.ci_r2:.2E}\n" \
                f"cov:     {self.cov}\n"
        else:
            s2 = ""
        return s1 + s2

# %% Fit functions


def fit_reactivity_ratios(
        f1: FloatVectorLike,
        F1: FloatVectorLike,
        scale_f: Union[float, FloatVectorLike] = 1.,
        scale_F: Union[float, FloatVectorLike] = 1.,
        method: Literal['NLLS', 'ODR'] = 'NLLS',
        alpha: float = 0.05,
        Mayo_plot: bool = True,
        JCR_method: list[Literal['linear', 'exact']] = ['linear'],
        rtol: float = 1e-5
) -> CopoFitResult:
    r"""Fit the reactivity ratios of the terminal model from instantaneous
    copolymer composition data.

    This function employs rigorous nonlinear algorithms to estimate the
    reactivity ratios from experimental $F(f)$ data obtained at low monomer
    conversion. The parameters are estimated by minimizing the sum of squared
    errors: 

    $$ SSE = \sum_i \left[ \left(\frac{f_i - \hat{f_i}}{s_{f_i}}\right)^2 +
             \left(\frac{F_i - \hat{F_i}}{s_{F_i}}\right)^2 \right]   $$

    where $s_{f_i}$ and $s_{F_i}$ are the scale factors for the monomer and the
    copolymer composition, respectively.

    The optimization is done using one of two methods: NLLS or ODR. The
    nonlinear least squares (NLLS) method neglects the first term of the
    summation, i.e. it only considers the observational errors in $F$. In
    contrast, the orthogonal distance regression (ODR) method takes the errors
    in both variables into account. 

    In well-designed experiments, the uncertainty in $f \ll F$, and so the NLLS
    method should suffice. However, if this condition is not met, the ODR
    method can be utilized to consider the uncertainty on both $f$ and $F$ in
    a statistically correct manner.

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
    f1 : FloatVectorLike (N)
        Molar fraction of M1.
    F1 : FloatVectorLike (N)
        Instantaneous copolymer composition of M1.
    scale_f : float | FloatVectorLike (N)
        Absolute scale for f1.
    scale_F : float | FloatVectorLike (N)
        Absolute scale for F1.
    method : Literal['NLLS', 'ODR']
        Optimization method. `NLLS` for nonlinear least squares or `ODR` for
        orthogonal distance regression. 
    alpha : float
        Significance level.
    Mayo_plot : bool
        If `True` a Mayo-Lewis plot will be generated.
    JCR_method : list[Literal['linear', 'exact']
        Method used to compute the joint confidence region of the reactivity
        ratios.
    rtol : float
        Relative tolerance of the ODE solver for the exact JCR method. The
        default value may be decreased by one or more orders of magnitude if
        the resolution of the JCR appears insufficient.

    Returns
    -------
    CopoFitResult
        Dataclass with all fit results.

    !!! note annotate "See also"

        * [`confidence_ellipse`](../math/confidence_ellipse.md): linear method
        used to calculate the joint confidence region.
        * [`confidence_region`](../math/confidence_region.md): exact method
        used to calculate the joint confidence region.
        * [`fit_Finemann_Ross`](fit_Finemann_Ross.md): alternative method.  

    Examples
    --------
    >>> from polykin.copolymerization.fitting import fit_reactivity_ratios
    >>>
    >>> f1 = [0.186, 0.299, 0.527, 0.600, 0.700, 0.798]
    >>> F1 = [0.196, 0.279, 0.415, 0.473, 0.542, 0.634]
    >>>
    >>> res = fit_reactivity_ratios(f1, F1)
    >>> print(
    ... f"r1={res.r1:.2f}±{res.ci_r1:.2f}, r2={res.r2:.2f}±{res.ci_r2:.2f}")
    r1=0.26±0.04, r2=0.81±0.10

    """

    f1, F1 = convert_FloatOrVectorLike_to_FloatVector([f1, F1])
    scale_f, scale_F = convert_FloatOrVectorLike_to_FloatOrVector(
        [scale_f, scale_F])

    # Check inputs
    check_shapes([f1, F1], [scale_f, scale_F])
    check_bounds(f1, 0., 1., 'f1')
    check_bounds(F1, 0., 1., 'F1')

    # Convert scale to vectors
    ndata = f1.size
    if not isinstance(scale_f, Sequence):
        scale_f = np.full(ndata, scale_f)
    if not isinstance(scale_F, Sequence):
        scale_F = np.full(ndata, scale_F)

    r1, r2, cov, = None, None, None
    error_message = ''
    if method == 'NLLS':

        solution = curve_fit(inst_copolymer_binary,
                             xdata=f1,
                             ydata=F1,
                             p0=(1., 1.),
                             sigma=scale_F,
                             absolute_sigma=False,  # for scaled cov
                             bounds=(0., np.inf),
                             full_output=True)
        if solution[4]:
            r1, r2 = solution[0]
            cov = solution[1]
        else:
            error_message = solution[3]

    elif method == 'ODR':

        odr_Model = odr.Model(lambda beta, x: inst_copolymer_binary(x, *beta))
        odr_Data = odr.RealData(x=f1, y=F1, sx=scale_f, sy=scale_F)
        odr_ODR = odr.ODR(odr_Data, odr_Model, beta0=(1., 1.))
        solution = odr_ODR.run()
        if (solution.info < 5):  # type: ignore
            r1, r2 = solution.beta
            # cov_beta is absolute, so rescaling is required
            cov = solution.cov_beta*solution.res_var  # type: ignore
            # estimated f1, needed for sse
            f1plus = solution.xplus  # type: ignore
        else:
            error_message = solution.stopreason

    else:
        raise ValueError(f"Invalid method `{method}`.")

    if r1 is None or r2 is None or cov is None:
        raise FitError(error_message)

    # Standard parameter errors and confidence intervals
    se_r = np.sqrt(np.diag(cov))
    ci_r = se_r*tdist.ppf(1. - alpha/2, ndata - 2)

    # Mayo plot
    Mayo = None
    if Mayo_plot:
        Mayo = plt.subplots()
        ax = Mayo[1]
        ax.set_xlabel(r"$f_1$")
        ax.set_ylabel(r"$F_1$")
        ax.scatter(f1, F1)
        ax.set_xlim(0., 1.)
        ax.set_ylim(0., 1.)
        x = np.linspace(0., 1., 200)
        y = inst_copolymer_binary(x, r1, r2)
        ax.plot(x, y)

    # Joint confidence region
    JCR = None
    if JCR_method:
        JCR = plt.subplots()
        ax = JCR[1]
        ax.set_xlabel(r"$r_1$")
        ax.set_ylabel(r"$r_2$")
        colors = ['tab:blue', 'tab:orange']
        idx = 0

        if 'linear' in JCR_method:
            confidence_ellipse(ax, (r1, r2), cov, ndata, alpha=alpha,
                               color=colors[idx], label='linear JCR')
            idx += 1

        if 'exact' in JCR_method:

            def sse_NLLS(r1, r2):
                ey = (F1 - inst_copolymer_binary(f1, r1, r2))/scale_F
                return np.dot(ey, ey)

            def sse_ODR(r1, r2):
                ey = (F1 - inst_copolymer_binary(f1plus, r1, r2))/scale_F
                ex = (f1 - f1plus)/scale_f
                return np.dot(ey, ey) + np.dot(ex, ex)

            sse = sse_NLLS if method == 'NLLS' else sse_ODR

            jcr = confidence_region((r1, r2), sse, ndata, alpha=alpha,
                                    width=ci_r[0], rtol=rtol)

            ax.scatter(r1, r2, c=colors[idx], s=5)
            ax.plot(jcr[0], jcr[1], color=colors[idx], label='exact JCR')

        ax.legend(loc="best")

    result = CopoFitResult(r1=r1, r2=r2,
                           cov=cov,
                           se_r1=se_r[0], se_r2=se_r[1],
                           ci_r1=ci_r[0], ci_r2=ci_r[1],
                           alpha=alpha,
                           method=method,
                           Mayo=Mayo, JCR=JCR)

    return result


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

    !!! note

        The Finemann-Ross method relies on a linearization procedure that can
        lead to significant statistical bias. The method is provided for its
        historical significance, but should no longer be used for fitting
        reactivity ratios.

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

    !!! note annotate "See also"

        * [`fit_reactivity_ratios`](fit_reactivity_ratios.md): alternative
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
