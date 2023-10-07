# polykin.physprops

::: polykin.physprops.tait
    options:
        members:
            - Tait

## Examples

Estimate the PVT properties of MMA.

```python exec="on" source="console"
from polykin.physprops import Tait

# Parameters from Handbook Polymer Solution Thermodynamics, p.39 
m = Tait(
    A0=8.2396e-4,
    A1=3.0490e-7,
    A2=7.0201e-10,
    B0=2.9803e8,
    B1=4.3789e-3,
    Tmin=387.15,
    Tmax=432.15,
    Pmin=0.1e6,
    Pmax=200e6,
    name="PMMA"
    )

print(m(159., 2000, Tunit='C', Punit='bar'))
print(m.alpha(432.15, 2e8))
print(m.beta(432.15, 2e8))
```
