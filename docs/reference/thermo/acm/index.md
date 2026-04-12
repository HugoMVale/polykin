# Activity Coeff. Models (polykin.thermo.acm)

[:simple-jupyter: Tutorial](../../../tutorials/activity_coefficient_models){ .md-button }

## Overview

This module implements activity coefficient models (ACMs) used in polymer reactor modeling. All models are formulated in terms of the molar excess Gibbs energy,

$$ g^{E} = g^{E}(T, x_i) $$

with all relations evaluated at constant pressure. Only multicomponent models are considered.

From $g^E$, “all other thermodynamic properties can be obtained. For instance, the excess entropy and enthalpy are given by:

\begin{aligned}
s^{E} &= -\left(\frac{\partial g^{E}}{\partial T}\right)_{P,x_i} \\
h^{E} &= -R T^2 \left(\frac{\partial (g^{E}/RT)}{\partial T} \right)_{P,x_i}
\end{aligned}

A _regular_ solution satisfies $s^{E}=0$, while an _atermal_ solution satisfies $h^{E}=0$.

The activity coefficients are given by:

$$ \ln \gamma_i = \frac{1}{RT}  \left( \frac{\partial (n g^E)}{\partial n_i} \right)_{T,P,n_j} $$

with:

$$ g^{E} = RT \sum_i x_i \ln \gamma_i $$
