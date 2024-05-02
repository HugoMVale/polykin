# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2024

import functools
from typing import Callable, Optional

import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.patches import Ellipse
from numpy import arctan, cos, exp, log, sin, sqrt
from scipy.stats.distributions import f as Fdist
from scipy.stats.distributions import t as tdist

from polykin.math.solvers import RootResult, root_secant
from polykin.utils.exceptions import ShapeError
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

    See also
    --------
    * [`confidence_region`](confidence_region.md): alternative method based on
    a rigorous approach for non-linear models.

    Examples
    --------
    >>> from polykin.math.jcr import confidence_ellipse
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig, ax = plt.subplots()
    >>> confidence_ellipse(ax=ax, center=(0.3,0.8),
    ...     cov=np.array([[2e-4, 3e-4], [3e-4, 2e-3]]), ndata=9)
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
                      sse: Callable[[tuple[float, float]], float],
                      ndata: int,
                      alpha: float = 0.05,
                      width: Optional[float] = None,
                      npoints: int = 200,
                      rtol: float = 1e-2
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
        The algorithm used to compute the JCR is efficient in comparison
        to naive 2D mesh screening approaches, but the number of $S(\beta)$
        evaluations remains large (typically several hundreds). Therefore, the
        applicability of this method depends on the cost of evaluating
        $S(\beta)$.

    **References**

    *   Vugrin, K. W., L. P. Swiler, R. M. Roberts, N. J. Stucky-Mack, and
    S. P. Sullivan (2007), Confidence region estimation techniques for
    nonlinear regression in groundwater flow: Three case studies, Water
    Resour. Res., 43.

    Parameters
    ----------
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
    npoints : int
        Number of points where the JCR is evaluated. The computational effort
        increases linearly with `npoints`.
    rtol : float
        Relative tolerance for the determination of the JCR. The default value
        (1e-2) should be adequate in most cases, as it implies a 1% accuracy in
        the JCR coordinates. 

    Returns
    -------
    tuple[FloatVector, FloatVector]
        Coordinates (x, y) of the confidence region.

    See also
    --------
    * [`confidence_ellipse`](confidence_ellipse.md): alternative method based
      on a linear approximation.

    Examples
    --------
    Let's generate a confidence region for a non-quadratic sse function.
    >>> from polykin.math import confidence_region
    >>> import matplotlib.pyplot as plt
    >>> def sse(x):
    ...     return 1. + ((x[0]-10)**2 + (x[1]-20)**2 + (x[0]-10)*np.sin((x[1]-20)**2))
    >>> x, y = confidence_region(center=(10, 20.), sse=sse, ndata=10, alpha=0.10)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x,y)
    """

    # Method implementation is specific for 2D
    npar = 2
    if len(center) != npar:
        raise ShapeError(f"`center` must be a vector of length {npar}.")

    # Check inputs
    check_bounds(alpha, 0.001, 0.99, 'alpha')
    check_bounds(ndata, npar + 1, np.inf, 'ndata')
    if width is not None:
        check_bounds(width, eps, np.inf, 'width')
    check_bounds(npoints, 1, 500, 'npoints')
    check_bounds(rtol, 1e-4, 1e-1, 'rtol')

    # Get 'sse' at center
    sse_center = sse(center)
    if abs(sse_center) < eps:
        raise ValueError(
            "`sse(center)` is close to zero. Without residual error, there is no JCR.")

    @functools.cache
    def nsse(lnr: float, θ: float) -> float:
        "Normalized 'sse' in log-radial coordinates"
        r = exp(lnr)
        x = center[0] + r*cos(θ)
        y = center[1] + r*sin(θ)
        return sse((x, y))/sse_center

    # Boundary value
    Fval = float(Fdist.ppf(1. - alpha, npar, ndata - npar))
    nsse_boundary = (1. + npar/(ndata - npar)*Fval)

    def find_radius(θ: float, r0: float) -> RootResult:
        "Find boundary ln(radius) at given angle θ."
        solution = root_secant(
            f=lambda x: nsse(x, θ)/nsse_boundary - 1.0,
            x0=log(r0*1.02),
            x1=log(r0),
            xtol=abs(log(1 + rtol)),
            ftol=1e-4,
            maxiter=100)
        return solution

    # Find radius at each angle using previous solution as initial guess
    angles = np.linspace(0., 2*np.pi, npoints)
    r = np.zeros_like(angles)
    r_guess_0 = abs(width/2) if width is not None else 0.25*center[0]
    r_guess = r_guess_0
    for i, θ in enumerate(angles):
        sol = find_radius(θ, r_guess)
        if sol.success:
            r[i] = exp(sol.x)
            r_guess = r[i]
        else:
            r[i] = np.nan
            r_guess = r_guess_0

    # Convert to cartesian coordinates
    x = center[0] + r*cos(angles)
    y = center[1] + r*sin(angles)

    return (x, y)
