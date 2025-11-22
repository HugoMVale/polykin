# Flash Solvers (polykin.thermo.flash)

## Overview

This module implements methods for solving single-stage, two-phase flash problems.

Currently, methods are available for problems specified by $P$-$T$, $P$-$V$, and $T$-$V$ conditions. Enthalpy- and entropy-based flashes are not yet supported.

In all cases, the following system of equations is solved:

$$\begin{aligned}
F & = L + V \\
F z_i &= L x_i + V y_i \\
y_i &= K_i(T,P,x,y) x_i
\end{aligned}$$

where $F$, $L$, and $V$ are the inlet feed, outlet liquid, and outlet vapor molar flow rates. The quantities $z_i$, $x_i$, and $y_i$ are the corresponding feed, liquid, and vapor mole fractions, and $K_i(T,P,x,y)$ are the K-values. These K-values may be obtained from an equation of state or from an activity-coefficient model.
