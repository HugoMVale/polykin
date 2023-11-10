# polykin.properties.diffusion

::: polykin.properties.diffusion.estimation_methods
    options:
        members:
            - DL_hayduk_minhas

## Examples

Estimate the diffusion coefficient of vinyl chloride through liquid water.

```python exec="on" source="console"
from polykin.properties.diffusion import DL_hayduk_minhas

D = DL_hayduk_minhas(
    T=298.,           # temperature
    method='aqueous', # equation for aqueous solutions
    MA=62.5e-3,       # molar mass of vinyl chloride
    rhoA=910.,        # density of vinyl chloride at the normal boiling point
    viscB=0.89e-3     # viscosity of water at solution temperature
    )

print(D)
```
