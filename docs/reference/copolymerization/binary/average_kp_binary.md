# polykin.copolymerization

::: polykin.copolymerization.binary
    options:
        members:
            - average_kp_binary

## Examples

```python exec="on" source="material-block"
from polykin.copolymerization import average_kp_binary
import numpy as np

kp = average_kp_binary(f1=0.5, r1=0.16, r2=0.70, k11=100., k22=1000.)

print(f"{kp:.0f} L/(mol·s)")
```
