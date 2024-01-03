# polykin.properties.viscosity

::: polykin.properties.viscosity
    options:
        members:
            - MUV_Lucas

## Examples

Estimate the viscosity of ethylene at 350 K and 10 bar.

```python exec="on" source="material-block"
from polykin.properties.viscosity import MUV_Lucas

mu = MUV_Lucas(T=350., P=10e5, M=28.05e-3,
                Tc=282.4, Pc=50.4e5, Zc=0.280, dm=0.) 

print(f"{mu:.2e} Pa·s")
```
