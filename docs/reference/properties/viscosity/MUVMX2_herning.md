# polykin.properties.viscosity

::: polykin.properties.viscosity
    options:
        members:
            - MUVMX2_herning

## Examples

Estimate the viscosity of a 50 mol% ethylene/1-butene mixture at 120°C and 1 bar.

```python exec="on" source="console"
from polykin.properties.viscosity import MUVMX2_herning
import numpy as np

y = np.array([0.5, 0.5])
M = np.array([28.e-3, 56.e-3])
mu = np.array([130e-7, 100e-7]) # from literature

mu_mix = MUVMX2_herning(y, mu, M)

print(mu_mix, "Pa.s")
```
