# polykin.copolymerization

::: polykin.copolymerization
    options:
        members:
            - monomer_drift_multi

## Examples

```python exec="on" source="material-block"
from polykin.copolymerization import monomer_drift_multi
import numpy as np

r = np.ones((3, 3))
r[0, 1] = 0.5
r[1, 0] = 2.3
r[0, 2] = 3.0
r[2, 0] = 0.9
r[1, 2] = 0.4
r[2, 1] = 1.5

f = monomer_drift_multi(f0=np.array([0.5, 0.3, 0.2]), r=r, x=[0.99])

print(f"f1 = {f[0,0]:.2f}; f2 = {f[0,1]:.2f}; f3 = {f[0,2]:.2f}")
```
