# Optimization Solvers (polykin.math.optimization)

## Scalar Optimization

The optimization module currently provides a single scalar minimization routine, [fmin_brent]. This function implements Brent’s derivative-free algorithm, which combines golden-section search with inverse parabolic interpolation to efficiently minimize a real-valued function over a bounded interval.

## Multidimensional Optimization

At present, `polykin.math.optimization` does not include solvers for multidimensional optimization.

[fmin_brent]: fmin_brent.md
