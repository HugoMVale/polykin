# Heat & Mass Transfer (polykin.transport.hmt)

## Overview

This module implements heat transfer correlations for geometries commonly encountered in
chemical processes. Thanks to the heat and mass transfer analogy, convection correlations of
the form $ Nu = Nu(Re, Pr) $ can also be used for mass transfer involving the same geometry by
interpreting them as $ Sh = Sh(Re, Sc) $.

| Geometry        | Forced convection      | Free convection     |
|-----------------|------------------------|---------------------|
| Cylinder        | [Nu_cylinder]          | [Nu_cylinder_free]  |
| Cylinder, bank  | [Nu_cylinder_bank]     | -                   |
| Flat plate      | [Nu_plate]             | [Nu_plate_free]     |
| Sphere          | [Nu_sphere], [Nu_drop] | [Nu_sphere_free]    |
| Stirred tank    | [Nu_tank]              | -                   |
| Tube (internal) | [Nu_tube]              | -                   |

[Nu_cylinder]: Nu_cylinder.md
[Nu_cylinder_bank]: Nu_cylinder_bank.md
[Nu_cylinder_free]: Nu_cylinder_free.md
[Nu_plate]: Nu_plate.md
[Nu_plate_free]: Nu_plate_free.md
[Nu_sphere]: Nu_sphere.md
[Nu_sphere_free]: Nu_sphere_free.md
[Nu_drop]: Nu_drop.md
[Nu_tube]: Nu_tube.md
[Nu_tank]: Nu_tank.md

## Dimensionless Numbers

| Dimensionless number | Definition                                               |
|----------------------|----------------------------------------------------------|
| Grashof, $Gr$        | $$ \frac{g \beta (T_s - T_b) L^3}{\nu ^ 2} $$            |
| Nusselt, $Nu$        | $$ \frac{h L}{k} $$                                      |
| Rayleigh, $Ra$       | $$ Gr Pr = \frac{g \beta (T_s - T_b) L^3}{\nu \alpha} $$ |
| Reynolds, $Re$       | $$ \frac{\rho v L}{\mu}=\frac{v L}{\nu} $$               |
| Prandtl, $Pr$        | $$ \frac{c_P \mu}{k} = \frac{\nu}{\alpha} $$             |
| Schmidt, $Sc$        | $$ \frac{\nu}{\mathscr{D}} $$                            |
| Sherwood, $Sh$       | $$ \frac{k_c L}{\mathscr{D}} $$                          |
