# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from typing import Optional

import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.patches import Ellipse
from scipy.stats.distributions import f as Fdist
from scipy.stats.distributions import t as tdist

from polykin.utils.exceptions import ShapeError
from polykin.utils.tools import check_bounds
from polykin.utils.types import Float2x2Matrix

__all__ = ['confidence_ellipse']


def confidence_ellipse(ax: Axes,
                       center: tuple[float, float],
                       cov: Float2x2Matrix,
                       ndata: int,
                       alpha: float = 0.05,
                       color: str = 'black',
                       label: Optional[str] = None,
                       confint: bool = True
                       ) -> None:
    r"""Generate a confidence ellipse for 2 jointly estimated parameters
    using a linear approximation method.

    The joint $100(1-\alpha)\%$ confidence region for the parameters
    $\beta=(\beta_1, \beta_2)$ is represented by the domain of values that
    satisfy the following condition:

    $$ (\beta -\hat\beta)^T V^{-1}(\beta -\hat\beta)
       \leq p F_{p,n-p,1-\alpha}$$

    where $\hat\beta$ is the point estimate of $\beta$ (obtained by
    least-squares fitting), $V$ is the _scaled_ variance-covariance
    matrix, $p=2$ is the number of parameters, $n>p$ is the number of data
    points considered in the regression, $\alpha$ is the significance level,
    and $F$ is the Fisher-Snedecor distribution.

    The method is exact for models that are linear in the parameters. For
    models that are non-linear in the parameters, the size and shape of the
    ellipse is only an approximation to the true joint confidence region.

    Additionally, the confidence intervals for the individual coefficients are
    given by:

    $$ \hat{\beta}_i \pm \hat{V}_{ii}^{1/2} t_{n-p,1-\alpha/2} $$

    where $t$ is Student's $t$ distribution.

    **Reference**

    *   Vugrin, K. W., L. P. Swiler, R. M. Roberts, N. J. Stucky-Mack, and
    S. P. Sullivan (2007), Confidence region estimation techniques for
    nonlinear regression in groundwater flow: Three case studies, Water
    Resour. Res., 43, W03423, doi:10.1029/2005WR004804.

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes object to which ellipse will be patched.
    center : tuple[float, float]
        Point estimate of the model parameters, $\hat{\beta}$.
    cov : Float2x2Matrix
        Scaled variance-covariance matrix (2x2), $V$.
    ndata : int
        Number of data points.
    alpha : float
        Significance level, $\alpha$.
    color : str
        Color of ellipse contour and center.
    label : str
        Ellipse label.
    confint : bool
        If `True` the _individual_ confidence intervals of the parameters are
        represented as lines.
    """

    # method implementation is specific for 2D
    npar = 2
    if len(center) != npar:
        raise ShapeError(f"`center` must be a vector of length {npar}.")

    if cov.shape != (npar, npar):
        raise ShapeError(f"`cov` must be a {npar}x{npar} matrix.")

    # check inputs
    check_bounds(alpha, 0.001, 0.50, 'alpha')
    check_bounds(ndata, npar + 1, np.inf, 'ndata')

    # eigenvalues and (orthogonal) eigenvectors of cov
    eigen = np.linalg.eigh(cov)
    eigenvector = eigen.eigenvectors[0]
    eigenvalues = eigen.eigenvalues

    # angle of ellipse wrt to x-axis
    angle = np.degrees(np.arctan(eigenvector[1]/eigenvector[0]))

    # "scale" of ellipse
    scale = npar*Fdist.ppf(1. - alpha, npar, ndata - npar)

    # lengths of ellipse axes
    width, height = 2*np.sqrt(scale*eigenvalues)

    ellipse = Ellipse(center,
                      width=width,
                      height=height,
                      angle=angle,
                      facecolor='none',
                      edgecolor=color,
                      label=label)

    ax.add_patch(ellipse)
    ax.scatter(*center, c=color, s=5)

    if confint:
        ci = np.sqrt(np.diag(cov))*tdist.ppf(1. - alpha/2, ndata - npar)
        ax.errorbar(*center, xerr=ci[0], yerr=ci[1], color=color)

    return None
