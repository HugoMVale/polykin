# polykin.properties.viscosity

::: polykin.properties.viscosity
    options:
        members:
            - MULMX2_perry

## Examples

Estimate the viscosity of a 50 mol% styrene/toluene liquid mixture at 20°C.

```python exec="on" source="material-block"
from polykin.properties.viscosity import MULMX2_perry
import numpy as np

x = np.array([0.5, 0.5])
mu = np.array([0.76, 0.59]) # cP, from literature

mu_mix = MULMX2_perry(x, mu)

print(f"{mu_mix:.2f} cP")
```
