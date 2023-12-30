# polykin.properties.eos

::: polykin.properties.eos
    options:
        members:
            - Virial
            - B_pure
            - B_mixture

## Examples

Estimate the molar volume of a 50 mol% ethylene/nitrogen at 350 K and 10 bar.

```python exec="on" source="material-block"
from polykin.properties.eos import Virial
import numpy as np

Tc = np.array([282.4, 126.2])    # K
Pc = np.array([50.4e5, 33.9e5])  # Pa
Zc = np.array([0.280, 0.290])
w = np.array([0.089, 0.039])

eos = Virial(Tc, Pc, Zc, w)
v = eos.v(T=350., P=10e5, y=np.array([0.5, 0.5]))

print(f"{v:.2e} m³/mol")
```
