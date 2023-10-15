# polykin.properties.pvt_polymer

::: polykin.properties.pvt_polymer
    options:
        members:
            - Flory
            - HartmannHaque
            - SanchezLacombe
            - Tait

## Examples

Estimate the PVT properties of PMMA.

```python exec="on" source="console"
from polykin.properties.pvt_polymer import Flory

# Parameters from Handbook of Diffusion and Thermal Properties of Polymers
# and Polymer Solutions, p.72. 
m = Flory(
    V0=0.7204e-3,
    T0=7717.,
    P0=568.8e6,
    Tmin=387.15,
    Tmax=432.15,
    Pmin=0.1e6,
    Pmax=200e6,
    name="PMMA"
    )

print(m.V(159., 2000, Tunit='C', Punit='bar'))
print(m.alpha(432.15, 2e8))
print(m.beta(432.15, 2e8))
```

```python exec="on" source="console"
from polykin.properties.pvt_polymer import Tait

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

print(m.V(159., 2000, Tunit='C', Punit='bar'))
print(m.alpha(432.15, 2e8))
print(m.beta(432.15, 2e8))
```
