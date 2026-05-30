# Optimization Solvers (polykin.math.optimization)

## Univariate Optimization

These solvers cover the typical trade-off between local speed and global robustness for minimization of scalar objective functions $f(x)$. They differ in the use of derivatives and whether a bounded interval is required.

[fmin_secant] is a fast local method for smooth problems when two reasonable initial guesses are available. It uses a secant approximation of the first derivative, evaluated via centered finite differences or complex-step differentiation. It typically converges quickly near a well-behaved minimum but is not robust to poor initial guesses, weak curvature, or nonsmooth objectives.

[fmin_brent] is a robust bounded method for cases where an interval containing a minimum is known. It combines golden-section search with inverse parabolic interpolation, achieving reliable convergence while exploiting smoothness when present. It does not require derivatives and is the preferred choice for safe one-dimensional minimization within a bracketed interval.

## Multivariate Optimization

These solvers provide different trade-offs between robustness, efficiency, and derivative requirements for unconstrained minimization of multivariate objective functions, $f(\mathbf{x})$.

[fmin_nelder_mead] is a derivative-free method for unconstrained minimization of multivariate functions. It evolves a simplex of trial points through reflection, expansion, contraction, and shrink operations, making it useful when derivatives are unavailable, unreliable, or too expensive to compute. In exchange for this flexibility, convergence is usually slower than with gradient-based methods, and performance may degrade for poorly scaled or high-dimensional problems.

[fmin_qnewton] is a fast local method for smooth unconstrained problems when gradient information is available analytically or can be estimated accurately. It combines Newton-type curvature information with optional BFGS Hessian updates and supports either a line-search or dogleg globalization strategy to improve robustness from remote initial guesses. This is generally the preferred solver when the objective is differentiable and a reasonably good starting point is available. As with other quasi-Newton methods, however, convergence remains fundamentally local.

[fmin_secant]: fmin_secant.md
[fmin_brent]: fmin_brent.md
[fmin_nelder_mead]: fmin_nelder_mead.md
[fmin_qnewton]: fmin_qnewton.md
