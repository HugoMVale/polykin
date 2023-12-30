# polykin.properties.eos

::: polykin.properties.eos
    options:
        members:
            - Soave

## Examples

Estimate the compressibility factor of a 50 mol% ethylene/nitrogen gas mixture at 300 K and
100 bar.

```python exec="on" source="material-block"
from polykin.properties.eos import Soave
import numpy as np

Tc = np.array([282.4, 126.2])    # K
Pc = np.array([50.4e5, 33.9e5])  # Pa
w = np.array([0.089, 0.039])

eos = Soave(Tc, Pc, w)
Z = eos.Z(T=300., P=100e5, y=np.array([0.5, 0.5]))

print(f"{Z[0]:.2f}")
```
