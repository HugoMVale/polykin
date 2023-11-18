# polykin.properties.viscosity

::: polykin.properties.viscosity
    options:
        members:
            - MULMX2_perry

## Examples

Estimate the viscosity of a 50 mol% styrene/toluene mixture at 20°C.

```python exec="on" source="console"
from polykin.properties.viscosity import MULMX2_perry
import numpy as np

x = np.array([0.5, 0.5])
visc = np.array([0.76, 0.59]) # cP, from literature
MU = MULMX2_perry(x, visc)

print(MU, "cP")
```
