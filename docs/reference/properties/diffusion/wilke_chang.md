
# polykin.properties.diffusion

::: polykin.properties.diffusion.estimation_methods
    options:
        members:
            - wilke_chang

## Examples

Estimate the diffusion coefficient of vinyl chloride through liquid water.

```python exec="on" source="console"
from polykin.properties.diffusion import wilke_chang

D = wilke_chang(
    T=298.,         # temperature
    MA=62.5e-3,     # molar mass of vinyl chloride
    MB=18.0e-3,     # molar mass of water
    rhoA=910.,      # density of vinyl chloride at the normal boiling point
    viscB=0.89e-3,  # viscosity of water at solution temperature
    phi=2.6         # association factor for water (see docstring)
    )

print(D)
```
