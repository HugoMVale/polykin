# Pressure Relief Valves (polykin.flow.prv)

## Overview

This module implements preliminary sizing methods for pressure relief devices — valves or
rupture disks — according to the API standard 520.

| Inlet Mixture           |   Single-phase Flow  |                     Two-phase Flow                    |
|-------------------------|:--------------------:|:-----------------------------------------------------:|
| Subcooled liquid        | [area_relief_liquid] |             [area_relief_2phase_subcooled]            |
| Saturated liquid        |           —          | [area_relief_2phase_subcooled], [area_relief_2phase]  |
| Liquid + Vapor          |           —          |                  [area_relief_2phase]                 |
| Saturated vapor         |           —          |                  [area_relief_2phase]                 |
| Superheated vapor / gas |   [area_relief_gas]  |                  [area_relief_2phase]                 |

[area_relief_gas]: area_relief_gas.md
[area_relief_liquid]: area_relief_liquid.md
[area_relief_2phase]: area_relief_2phase.md
[area_relief_2phase_subcooled]: area_relief_2phase_subcooled.md