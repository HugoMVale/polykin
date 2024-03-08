# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from typing import Callable, Optional

import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.patches import Ellipse
from numpy import arctan, cos, sin, sqrt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.stats.distributions import f as Fdist
from scipy.stats.distributions import t as tdist

from polykin.utils.exceptions import (ODESolverError, RootSolverError,
                                      ShapeError)
from polykin.utils.math import eps
from polykin.utils.tools import check_bounds
from polykin.utils.types import Float2x2Matrix

__all__ = ['confidence_ellipse',
           'confidence_region']


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

    The joint $100(1-\alpha)\%$ confidence ellipse for the parameters
    $\beta=(\beta_1, \beta_2)$ is represented by the domain of values that
    satisfy the following condition:

    $$ \left \{\beta: (\beta -\hat{\beta})^T \hat{V}^{-1}(\beta -\hat{\beta})
       \leq 2 F_{2,n-2,1-\alpha} \right \} $$

    where $\hat{\beta}$ is the point estimate of $\beta$ (obtained by
    least-squares fitting), $\hat{V}$ is the _scaled_ variance-covariance
    matrix, $n>2$ is the number of data points considered in the regression,
    $\alpha$ is the significance level, and $F$ is the Fisher-Snedecor
    distribution.

    !!! note

        This method is only exact for models that are linear in the parameters.
        For models that are non-linear in the parameters, the size and shape of
        the ellipse is only an approximation to the true joint confidence
        region.

    Additionally, the confidence intervals for the individual coefficients are
    given by:

    $$ \hat{\beta}_i \pm \hat{V}_{ii}^{1/2} t_{n-2,1-\alpha/2} $$

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
        Scaled variance-covariance matrix (2x2), $\hat{V}$.
    ndata : int
        Number of data points.
    alpha : float
        Significance level, $\alpha$.
    color : str
        Color of ellipse contour and center.
    label : str | None
        Ellipse label.
    confint : bool
        If `True` the _individual_ confidence intervals of the parameters are
        represented as lines.

    !!! note annotate "See also"

        * [`confidence_region`](confidence_region.md): alternative method
        based on rigorous approach for non-linear models.
    """

    # Method implementation is specific for 2D
    npar = 2
    if len(center) != npar:
        raise ShapeError(f"`center` must be a vector of length {npar}.")

    if cov.shape != (npar, npar):
        raise ShapeError(f"`cov` must be a {npar}x{npar} matrix.")

    # Check inputs
    check_bounds(alpha, 0.001, 0.90, 'alpha')
    check_bounds(ndata, npar + 1, np.inf, 'ndata')

    # Eigenvalues and (orthogonal) eigenvectors of cov
    eigen = np.linalg.eigh(cov)
    eigenvector = eigen.eigenvectors[0]
    eigenvalues = eigen.eigenvalues

    # Angle of ellipse wrt to x-axis
    angle = np.degrees(arctan(eigenvector[1]/eigenvector[0]))

    # "Scale" of ellipse
    scale = npar*Fdist.ppf(1. - alpha, npar, ndata - npar)

    # Lengths of ellipse axes
    width, height = 2*sqrt(scale*eigenvalues)

    # Ellipse
    ellipse = Ellipse(center,
                      width=width,
                      height=height,
                      angle=angle,
                      facecolor='none',
                      edgecolor=color,
                      label=label)

    ax.add_patch(ellipse)
    ax.scatter(*center, c=color, s=5)

    # Individual confidence intervals
    if confint:
        ci = sqrt(np.diag(cov))*tdist.ppf(1. - alpha/2, ndata - npar)
        ax.errorbar(*center, xerr=ci[0], yerr=ci[1], color=color)

    return None


def confidence_region(ax: Axes,
                      center: tuple[float, float],
                      sse: Callable[[tuple[float, float]], float],
                      ndata: int,
                      alpha: float = 0.05,
                      width: Optional[float] = None,
                      rtol: float = 1e-4,
                      color: str = 'black',
                      label: Optional[str] = None,
                      npoints: int = 200
                      ) -> None:
    r"""Generate a confidence region for 2 jointly estimated parameters
    using a rigorous method.

    The joint $100(1-\alpha)\%$ confidence region (JCR) for the parameters
    $\beta=(\beta_1, \beta_2)$ is represented by the domain of values that
    satisfy the following condition:

    $$ \left \{\beta: S(\beta) \leq S(\hat{\beta})
        [1 + \frac{2}{n-2} F_{2,n-2,1-\alpha}] \right \} $$

    where $\hat\beta$ is the point estimate of $\beta$ (obtained by
    least-squares fitting), $S(\beta)$ is the error sum of squares (SSE)
    function, $n>2$ is the number of data points considered in the regression,
    $\alpha$ is the significance level, and $F$ is the Fisher-Snedecor
    distribution.

    !!! note

        This method is suitable for arbitrary models (linear or non-linear in
        the parameters), without making assumptions about the shape of the JCR.
        The algorithm used to compute the JCR is very efficient in comparison
        to naive 2D mesh screening approaches, but the number of $S(\beta)$
        evaluations remains large (typically several hundreds). Therefore, the
        applicability of this method depends on the cost of evaluating
        $S(\beta)$.

    **Reference**

    *   Vugrin, K. W., L. P. Swiler, R. M. Roberts, N. J. Stucky-Mack, and
    S. P. Sullivan (2007), Confidence region estimation techniques for
    nonlinear regression in groundwater flow: Three case studies, Water
    Resour. Res., 43, W03423, doi:10.1029/2005WR004804.

    * Arutjunjan, R., Schaefer, B. M., Kreutz, C., Constructing Exact
    Confidence Regions on Parameter Manifolds of Non-Linear Models, 2022,
    https://doi.org/10.48550/arXiv.2211.03421.

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes object to which the joint confidence region will be
        added.
    center : tuple[float, float]
        Point estimate of the model parameters, $\hat{\beta}$.
    sse : Callable[[tuple[float, float]], float]
        Error sum of squares function, $S(\beta_1, \beta_2)$.
    ndata : int
        Number of data points.
    alpha : float
        Significance level, $\alpha$.
    width : float | None
        Initial guess of the width of the joint confidence region at its
        center. If `None`, it is assumed that `width=0.5*center[0]`.
    rtol : float
        Relative tolerance of ODE solver.
    color : str
        Color of confidence region boundary and center.
    label : str | None
        Ellipse label.
    npoints : int
        Number of points used to draw the confidence region.

    !!! note annotate "See also"

        * [`confidence_ellipse`](confidence_ellipse.md): alternative method
        based on linear approximation.

    """

    # Method implementation is specific for 2D
    npar = 2
    if len(center) != npar:
        raise ShapeError(f"`center` must be a vector of length {npar}.")

    # Check inputs
    check_bounds(alpha, 0.001, 0.90, 'alpha')
    check_bounds(ndata, npar + 1, np.inf, 'ndata')

    # Boundary
    Fval = Fdist.ppf(1. - alpha, npar, ndata - npar)
    sse_boundary = sse(center)*(1. + npar/(ndata - npar)*Fval)

    # Find boundary radius at 3 o'clock
    sol = root_scalar(
        f=lambda r: sse((center[0] + r, center[1])) - sse_boundary,
        method='secant',
        x0=0,
        x1=width/2 if width is not None else 0.25*center[0])
    if not sol.converged:
        raise RootSolverError(sol.flag)
    else:
        r0 = sol.root

    # Move along boundary using radial coordinates
    def rdot(theta: float, r: float) -> float:
        "Calculate dr/dtheta along a path of constant 'sse'."
        a = sin(theta)
        b = cos(theta)
        x = center[0] + r*b
        y = center[1] + r*a
        sse_0 = sse((x, y))
        h = 2*sqrt(eps)
        hx = max(h, abs(x)*h)
        hy = max(h, abs(y)*h)
        sse_x = (sse((x + hx, y)) - sse_0)/hx
        sse_y = (sse((x, y + hy)) - sse_0)/hy
        return r*(a*sse_x - b*sse_y)/(a*sse_y + b*sse_x)

    theta_span = (0, 2*np.pi)
    sol = solve_ivp(rdot, theta_span, [r0],
                    t_eval=np.linspace(*theta_span, npoints),
                    method='RK45',
                    atol=0., rtol=rtol)
    if not sol.success:
        raise ODESolverError(sol.message)
    else:
        theta = sol.t
        r = sol.y[0, :]
        if not np.isclose(r[0], r[-1], atol=0, rtol=10*rtol):
            print("Warning: offset between start and end positions of JCR > 10*rtol.")

    # Convert to cartesian coordinates and plot
    x = center[0] + r*cos(theta)
    y = center[1] + r*sin(theta)
    ax.plot(x, y, color=color, label=label)
    ax.scatter(*center, c=color, s=5)

    return None
