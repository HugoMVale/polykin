# Optimization Solvers (polykin.math.optimization)

## Univariate Optimization

These solvers cover the typical trade-off between local speed and global robustness for minimization of scalar objective functions $f(x)$. They differ in the use of derivatives and whether a bounded interval is required.

[fmin_secant] is a fast local method for smooth problems when two reasonable initial guesses are available. It uses a secant approximation of the first derivative, evaluated via centered finite differences or complex-step differentiation. It typically converges quickly near a well-behaved minimum but is not robust to poor initial guesses, weak curvature, or nonsmooth objectives.

[fmin_brent] is a robust bounded method for cases where an interval containing a minimum is known. It combines golden-section search with inverse parabolic interpolation, achieving reliable convergence while exploiting smoothness when present. It does not require derivatives and is the preferred choice for safe one-dimensional minimization within a bracketed interval.

## Multivariate Optimization

At present, `polykin.math.optimization` does not include solvers for multidimensional optimization.

[fmin_secant]: fmin_secant.md
[fmin_brent]: fmin_brent.md
