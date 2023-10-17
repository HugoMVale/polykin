# polykin.properties.pvt_polymer

::: polykin.properties.pvt_polymer
    options:
        members:
            - HartmannHaque

## Parameter databank

{{ read_csv('src/polykin/properties/pvt_polymer/HartmannHaque_parameters.tsv', delim_whitespace=True) }}

## Examples

Estimate the PVT properties of PMMA.

```python exec="on" source="console"
from polykin.properties.pvt_polymer import HartmannHaque

# Parameters from Handbook of Diffusion and Thermal Properties of Polymers
# and Polymer Solutions, p.85. 
m = HartmannHaque(
    V0=0.7582e-3,
    T0=1467.,
    P0=3819e6,
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
from polykin.properties.pvt_polymer import HartmannHaque

# Parameters retrieved from internal databank 
m = HartmannHaque.from_database("PMMA")

print(m.V(127., 1500, Tunit='C', Punit='bar'))
print(m.alpha(400., 1.5e8))
print(m.beta(400., 1.5e8))
```
