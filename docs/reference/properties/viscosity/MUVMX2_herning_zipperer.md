# polykin.properties.viscosity

::: polykin.properties.viscosity
    options:
        members:
            - MUVMX2_herning_zipperer

## Examples

Estimate the viscosity of a 50 mol% ethylene/1-butene gas mixture at 120°C and 1 bar.

```python exec="on" source="material-block"
from polykin.properties.viscosity import MUVMX2_herning_zipperer
import numpy as np

y = np.array([0.5, 0.5])
mu = np.array([130e-7, 100e-7]) # Pa.s, from literature
M = np.array([28.e-3, 56.e-3])  # kg/mol

mu_mix = MUVMX2_herning_zipperer(y, mu, M)

print(f"{mu_mix:.2e} Pa·s")
```
