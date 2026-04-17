# Optimization Solvers (polykin.math.optimize)

## Scalar Root Solvers

These three solvers span the usual tradeoff between local speed and global robustness for real-valued equations $f(x)=0$. They differ mainly in how much information they require and how sensitive they are to the initial guess.

[root_newton] is the fastest option for smooth problems with a good initial guess. It uses a Newton–Raphson update and estimates derivatives via complex-step differentiation, so no user-supplied derivative is needed. Convergence is typically very rapid, but the function must accept complex inputs, and—as with any Newton method—it may fail or converge to the wrong root if initialized poorly.

[root_secant] removes the need for derivatives by approximating the slope from two initial guesses. It retains fast local convergence on well-behaved problems and works when complex-step differentiation is not applicable. It is generally less robust than Brent’s method and less decisive than Newton near difficult roots, but provides a useful middle ground between simplicity and speed.

[root_brent] is the safest choice when a bracketing interval is available. By combining bisection, secant steps, and inverse quadratic interpolation, it maintains the reliability of bracketed methods while recovering much of the speed of open methods. The only requirement is an interval where the function changes sign; when this is available, Brent’s method is typically the best choice for robust production use.

## Vector Root Solver

[rootvec_qnewton] targets systems of nonlinear equations $\boldsymbol{F}(\boldsymbol{x})=\boldsymbol{0}$, which are inherently more challenging than scalar problems. There is no multidimensional analogue of bracketing, and thus no method combining global guarantees with high efficiency comparable to Brent’s method.

Instead, such solvers rely on local linearization. Classical Newton methods use the full Jacobian, but forming and factorizing it can be expensive for large systems. `rootvec_qnewton` supports two approaches: recomputing the full Jacobian at each iteration (either user-supplied or via finite differences), or updating an approximate Jacobian using Broyden’s rank-1 formula. In this sense, Broyden’s method plays a role analogous to the secant method in the scalar case.

Full Jacobian updates improve robustness at higher cost, while Broyden updates reduce per-iteration cost at the expense of reliability, especially with poor initial guesses or strong nonlinearity. As with other quasi-Newton methods, convergence is local, so good initial guesses remain important.

[root_newton]: root_newton.md
[root_secant]: root_secant.md
[root_brent]: root_brent.md
[rootvec_qnewton]: rootvec_qnewton.md
