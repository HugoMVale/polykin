# polykin.properties.viscosity

::: polykin.properties.viscosity
    options:
        members:
            - MUVMXPC_Dean_Stiel

## Examples

Estimate the residual viscosity of a 50 mol% ethylene/propylene mixture
at 350 K and 100 bar.

```python exec="on" source="material-block"
from polykin.properties.viscosity import MUVMXPC_Dean_Stiel
import numpy as np

v = 1.12e-4  # m³/mol, with Peng-Robinson

y = np.array([0.5, 0.5])
M = np.array([28.05e-3, 42.08e-3]) # kg/mol
Tc = np.array([282.4, 364.9])      # K
Pc = np.array([50.4e5, 46.0e5])    # Pa
Zc = np.array([0.280, 0.274])

mu_residual = MUVMXPC_Dean_Stiel(v, y, M, Tc, Pc, Zc)

print(f"{mu_residual:.2e} Pa·s")
```
