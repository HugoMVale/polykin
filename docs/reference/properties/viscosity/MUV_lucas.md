# polykin.properties.viscosity

::: polykin.properties.viscosity
    options:
        members:
            - MUV_lucas

## Examples

Estimate the viscosity of ethylene at 350 K and 10 bar.

```python exec="on" source="console"
from polykin.properties.viscosity import MUV_lucas

MU = MUV_lucas(T=350., P=10e5, M=28.05e-3,
                Tc=282.4, Pc=50.4e5, Zc=0.280, dm=0.) 

print(MU, "Pa.s")
```
