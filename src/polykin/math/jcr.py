# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

from typing import Callable, Optional

import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.patches import Ellipse
from numpy import arctan, cos, exp, log, sin, sqrt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.stats.distributions import f as Fdist
from scipy.stats.distributions import t as tdist

from polykin.utils.exceptions import (ODESolverError, RootSolverError,
                                      ShapeError)
from polykin.utils.math import eps
from polykin.utils.tools import check_bounds
from polykin.utils.types import Float2x2Matrix, FloatVector

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
    Resour. Res., 43.

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes object to which ellipse will be patched.
    center : tuple[float, float]
        Point estimate of the model parameters, $\hat{\beta}$.
    cov : Float2x2Matrix
        Scaled variance-covariance matrix, $\hat{V}$.
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


def confidence_region(center: tuple[float, float],
                      sse: Callable[[float, float], float],
                      ndata: int,
                      alpha: float = 0.05,
                      width: Optional[float] = None,
                      rtol: float = 1e-4
                      ) -> tuple[FloatVector, FloatVector]:
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

    * Arutjunjan, R., Schaefer, B. M., Kreutz, C., Constructing Exact
    Confidence Regions on Parameter Manifolds of Non-Linear Models, 2022.

    *   Vugrin, K. W., L. P. Swiler, R. M. Roberts, N. J. Stucky-Mack, and
    S. P. Sullivan (2007), Confidence region estimation techniques for
    nonlinear regression in groundwater flow: Three case studies, Water
    Resour. Res., 43.

    Parameters
    ----------
    center : tuple[float, float]
        Point estimate of the model parameters, $\hat{\beta}$.
    sse : Callable[[float, float], float]
        Error sum of squares function, $S(\beta_1, \beta_2)$.
    ndata : int
        Number of data points.
    alpha : float
        Significance level, $\alpha$.
    width : float | None
        Initial guess of the width of the joint confidence region at its
        center. If `None`, it is assumed that `width=0.5*center[0]`.
    rtol : float
        Relative tolerance of ODE solver. The default value may be decreased
        by one or more orders of magnitude if the resolution of the JCR appears
        insufficient.

    Returns
    -------
    tuple[FloatVector, FloatVector]
        Coordinates (x, y) of the confidence region.

    !!! note annotate "See also"

        * [`confidence_ellipse`](confidence_ellipse.md): alternative method
        based on linear approximation.

    """

    # Method implementation is specific for 2D
    npar = 2
    if len(center) != npar:
        raise ShapeError(f"`center` must be a vector of length {npar}.")

    # Check inputs
    check_bounds(alpha, 0.001, 0.99, 'alpha')
    check_bounds(ndata, npar + 1, np.inf, 'ndata')

    # Boundary
    Fval = Fdist.ppf(1. - alpha, npar, ndata - npar)
    sse_boundary = sse(*center)*(1. + npar/(ndata - npar)*Fval)

    # Find boundary radius at 3 o'clock
    sol = root_scalar(
        f=lambda r: sse(center[0] + r, center[1]) - sse_boundary,
        method='secant',
        x0=0,
        x1=width/2 if width is not None else 0.25*center[0])
    if not sol.converged:
        raise RootSolverError(sol.flag)
    else:
        r0 = sol.root

    # Transform 'sse' to log-radial coordinates
    def f(s: float, q: float) -> float:
        r = exp(s)
        x = center[0] + r*cos(q)
        y = center[1] + r*sin(q)
        return sse(x, y)

    # Move along boundary using radial coordinates
    # Imagine a particle transported by a velocity field orthogonal to 'sse'
    def dydt(t: float, y: np.ndarray) -> np.ndarray:
        s = y[0]
        q = y[1]
        h = 2*sqrt(eps)
        f_s = (f(s + h, q) - f(s - h, q))/h
        f_q = (f(s, q + h) - f(s, q - h))/h
        ydot = np.empty_like(y)
        ydot[0] = -f_q
        ydot[1] = f_s
        return ydot

    # Stop the trajectory after one revolution
    def event(t: float, y: np.ndarray) -> float:
        return y[1] - 2*np.pi
    event.terminal = True

    sol = solve_ivp(dydt, (0., 1e10), (log(r0), 0.),
                    method='LSODA',
                    events=event,
                    atol=[0, 1*rtol], rtol=rtol)
    if not sol.success:
        raise ODESolverError(sol.message)
    else:
        r = exp(sol.y[0, :])
        theta = sol.y[1, :]
        if not np.isclose(r[0], r[-1], atol=min(*center), rtol=10*rtol):
            print("Warning: offset between start and end positions of JCR > 10*rtol.")

    # Convert to cartesian coordinates
    x = center[0] + r*cos(theta)
    y = center[1] + r*sin(theta)

    return (x, y)
