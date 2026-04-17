# Optimization Solvers (polykin.math.optimization)

## Scalar Optimization

The optimization module currently provides a single scalar minimization method, [fmin_brent]. It implements Brent's derivative-free algorithm, combining golden-section search with inverse parabolic interpolation to minimize a real-valued function over a bounded interval.

This method is intended for one-dimensional local minimization when a bracketing interval is available. In smooth regions it can accelerate with parabolic steps; when that is not reliable, it falls back to the more robust golden-section strategy.

## Multidimensional Optimization

There are currently no solvers for multidimensional optimization in `polykin.math.optimization`.

[fmin_brent]: fmin_brent.md
