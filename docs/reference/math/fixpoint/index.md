# Fixed-Point Solvers (polykin.math.fixpoint)

## Scalar Problems

[fixpoint_steffensen] is the only solver in this group specialized for scalar fixed-point problems $g(x)=x$. It accelerates direct substitution through Aitken's delta-squared process, often converging significantly faster than plain fixed-point iteration when the fixed-point sequence is smooth and locally contractive.

## Multivariable Problems

[fixpoint_damped] is the most basic option for systems $\mathbf{g}(\mathbf{x})$. It applies direct substitution with optional damping, making each iteration inexpensive and easy to interpret. This is a reasonable first choice when the mapping is already contractive or only mildly unstable, especially if function evaluations are cheap.

[fixpoint_wegstein] improves on direct substitution by estimating a componentwise secant slope and converting it into a bounded acceleration factor. In practice, it often offers a favorable balance between robustness and speed for weakly to moderately coupled systems. It is usually more effective than simple damping, but because the acceleration is applied componentwise, performance can degrade when the variables are strongly coupled.

[fixpoint_dem] targets problems whose slow convergence is dominated by a single eigenvalue of the fixed-point iteration. By estimating that dominant mode and periodically applying a promotion step, it can reduce iteration counts for recycle-type problems and other slowly convergent systems. It is more specialized than Wegstein's method and is most useful when there is a clear dominant convergence mode.

[fixpoint_anderson] is generally the most powerful solver in this family for difficult coupled systems. It combines several previous iterates through a least-squares acceleration step, often delivering substantially better convergence than direct substitution-based methods. The additional linear algebra introduces some overhead, but this cost is usually justified when evaluating $\mathbf{g}$ is expensive or when simpler accelerators stall.

[fixpoint_steffensen]: fixpoint_steffensen.md
[fixpoint_damped]: fixpoint_damped.md
[fixpoint_wegstein]: fixpoint_wegstein.md
[fixpoint_dem]: fixpoint_dem.md
[fixpoint_anderson]: fixpoint_anderson.md
