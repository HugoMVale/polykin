# Thermal Conductivity (polykin.properties.thermal_conductivity)

This module implements methods to calculate the thermal conductivity of pure gases, gas
mixtures, pure liquids, and liquid mixtures.

|                     |                     Gas                    |      Liquid     |
|---------------------|:------------------------------------------:|:---------------:|
| DIPPR equations     | [DIPPR100], [DIPPR102]                     | [DIPPR100]      |
| Estimation methods  |                      —                     |      —          |
| Mixing rules        | [KVMX2_Wassilijewa]                        | [KLMX2_Li]      |
| Pressure correction | [KVPC_Stiel_Thodos], [KVMXPC_Stiel_Thodos] |      —          |

!!! note

    * Estimation methods for gases are not included, as they are rather complicated to apply,
    typically requiring structural information, heat capacity data, and viscosity data. If all
    that information is available, it is quite likely that experimental thermal conductivity
    data are also available — in which case, those should be used instead.

    * Estimation methods for liquids are also not included, as none of those reported in the
    literature are considered sufficiently reliable. Nonetheless, below the normal boiling point,
    most organic liquids have thermal conductivities between 0.1 and 0.2 W /(m·K), which can be
    used as a first approximation.

[:simple-jupyter: Tutorial](../../../tutorials/thermal_conductivity){ .md-button }

[DIPPR100]: ../equations/index.md#polykin.properties.equations.dippr.DIPPR100
[DIPPR102]: ../equations/index.md#polykin.properties.equations.dippr.DIPPR102

[KLMX2_Li]: KLMX2_Li.md
[KVMX2_Wassilijewa]: KVMX2_Wassilijewa.md
[KVPC_Stiel_Thodos]: KVPC_Stiel_Thodos.md
[KVMXPC_Stiel_Thodos]: KVMXPC_Stiel_Thodos.md