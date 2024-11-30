# Thermal Conductivity (polykin.properties.thermal_conductivity)

This module implements methods to calculate the thermal conductivity of pure gases, gas
mixtures, pure liquids, and liquid mixtures.

|                     |                     Gas                    |      Liquid     |
|---------------------|:------------------------------------------:|:---------------:|
| DIPPR equations     | [DIPPR100], [DIPPR102]                     | [DIPPR100]      |
| Estimation methods  |                      —                     |      —          |
| Mixing rules        | [KVMX2_Wassilijewa]                        | [KLMX2_Li]      |
| Pressure correction | [KVPC_Stiel_Thodos], [KVMXPC_Stiel_Thodos] |      —          |

[:simple-jupyter: Tutorial](../../../tutorials/thermal_conductivity){ .md-button }

[DIPPR100]: ../equations/index.md#polykin.properties.equations.dippr.DIPPR100
[DIPPR102]: ../equations/index.md#polykin.properties.equations.dippr.DIPPR102

[KLMX2_Li]: KLMX2_Li.md
[KVMX2_Wassilijewa]: KVMX2_Wassilijewa.md
[KVPC_Stiel_Thodos]: KVPC_Stiel_Thodos.md
[KVMXPC_Stiel_Thodos]: KVMXPC_Stiel_Thodos.md