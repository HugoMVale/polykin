# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from scipy import odr
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.stats.distributions import t as tdist

from polykin.copolymerization.binary import inst_copolymer_binary
from polykin.math import confidence_ellipse
from polykin.utils.exceptions import FitError
from polykin.utils.math import (convert_FloatOrVectorLike_to_FloatOrVector,
                                convert_FloatOrVectorLike_to_FloatVector)
from polykin.utils.tools import check_bounds, check_shapes
from polykin.utils.types import (Float2x2Matrix, FloatOrVectorLike,
                                 FloatVectorLike)

__all__ = ['fit_Finemann_Ross',
           'fit_reactivity_ratios']


# %% CopoFitResult


@dataclass
class CopoFitResult():
    """Something"""
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
                f"se_r1:   {self.se_r1:.2E}\n" \
                f"se_r2:   {self.se_r2:.2E}\n" \
                f"ci_r1:   {self.ci_r1:.2E}\n" \
                f"ci_r2:   {self.ci_r2:.2E}\n" \
                f"cov:     {self.cov}\n"
        else:
            s2 = ""
        return s1 + s2

# %%


def fit_reactivity_ratios(
        f1: FloatVectorLike,
        F1: FloatVectorLike,
        sigma_f: FloatOrVectorLike,
        sigma_F: FloatOrVectorLike,
        method: Literal['NLLS', 'ODR'] = 'NLLS',
        alpha: float = 0.05,
        plot_Mayo: bool = True,
        plot_JCR: bool = True) -> CopoFitResult:

    f1, F1 = convert_FloatOrVectorLike_to_FloatVector([f1, F1])
    sigma_f, sigma_F = convert_FloatOrVectorLike_to_FloatOrVector(
        [sigma_f, sigma_F])

    # Check inputs
    check_shapes([f1, F1], [sigma_f, sigma_F])
    check_bounds(f1, 0., 1., 'f1')
    check_bounds(F1, 0., 1., 'F1')

    # Convert sigma to vectors
    ndata = f1.size
    if isinstance(sigma_f, float):
        sigma_f = np.full(ndata, sigma_f)
    if isinstance(sigma_F, float):
        sigma_F = np.full(ndata, sigma_F)

    r1, r2, cov, = None, None, None
    error_message = ''
    if method == 'NLLS':

        solution = curve_fit(inst_copolymer_binary,
                             xdata=f1,
                             ydata=F1,
                             p0=(1., 1.),
                             sigma=sigma_F,
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
        odr_Data = odr.RealData(x=f1, y=F1, sx=sigma_f, sy=sigma_F)
        odr_ODR = odr.ODR(odr_Data, odr_Model, beta0=(1., 1.))
        solution = odr_ODR.run()
        if (solution.info < 5):  # type: ignore
            r1, r2 = solution.beta
            # cov_beta is absolute, so rescaling is required
            cov = solution.cov_beta*solution.res_var  # type: ignore
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
    if plot_Mayo:
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
    if plot_JCR:
        JCR = plt.subplots()
        ax = JCR[1]
        ax.set_xlabel(r"$r_1$")
        ax.set_ylabel(r"$r_2$")
        confidence_ellipse(ax, (r1, r2), cov, ndata, alpha, 'tab:blue')
        ax.legend(bbox_to_anchor=(1.05, 1.), loc="upper left")

    result = CopoFitResult(r1=r1, r2=r2,
                           cov=cov,
                           se_r1=se_r[0], se_r2=se_r[1],
                           ci_r1=ci_r[0], ci_r2=ci_r[1],
                           alpha=alpha,
                           method=method,
                           Mayo=Mayo, JCR=JCR)

    return result

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

    Examples
    --------
    >>> from polykin.copolymerization.fitting import fit_Finemann_Ross
    >>> import numpy as np
    >>>
    >>> f1 = [0.186, 0.299, 0.527, 0.600, 0.700, 0.798]
    >>> F1 = [0.196, 0.279, 0.415, 0.473, 0.542, 0.634]
    >>>
    >>> r1, r2 = fit_Finemann_Ross(f1, F1)
    >>> print(f"r1 = {r1:.3f}, r2 = {r2:.3f}")
    r1 = 0.226, r2 = 0.762

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
