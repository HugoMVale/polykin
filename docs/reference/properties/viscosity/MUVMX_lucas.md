# polykin.properties.viscosity

::: polykin.properties.viscosity
    options:
        members:
            - MUVMX_lucas

## Examples

Estimate the viscosity of a 60 mol% ethylene/nitrogen gas mixture at 350 K and 10 bar.

```python exec="on" source="material-block"
from polykin.properties.viscosity import MUVMX_lucas
import numpy as np

y = np.array([0.6, 0.4])
M = np.array([28.e-3, 28.e-3])   # kg/mol
Tc = np.array([282.4, 126.2])    # K
Pc = np.array([50.4e5, 33.9e5])  # Pa
Zc = np.array([0.280, 0.290])
dm = np.array([0., 0.])

mu_mix = MUVMX_lucas(T=350., P=10e5, y=y, M=M, Tc=Tc, Pc=Pc, Zc=Zc, dm=dm)

print(f"{mu_mix:.2e} Pa·s")
```
