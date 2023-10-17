# polykin.properties.pvt_polymer

::: polykin.properties.pvt_polymer
    options:
        members:
            - Tait

## Parameter databank

{{ read_csv('src/polykin/properties/pvt_polymer/Tait_parameters.tsv', delim_whitespace=True) }}

## Examples

Estimate the PVT properties of PMMA.

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

print(m.V(127., 1500, Tunit='C', Punit='bar'))
print(m.alpha(400., 1.5e8))
print(m.beta(400., 1.5e8))
```

```python exec="on" source="console"
from polykin.properties.pvt_polymer import Tait

# Parameters retrieved from internal databank 
m = Tait.from_database("PMMA")

print(m.V(127., 1500, Tunit='C', Punit='bar'))
print(m.alpha(400., 1.5e8))
print(m.beta(400., 1.5e8))
```
