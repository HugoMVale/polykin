# polykin.properties.vaporization_enthalpy

::: polykin.properties.vaporization_enthalpy
    options:
        members:
            - DHVL_pitzer

## Examples

Estimate the vaporization enthalpy of vinyl chloride at 50°C.

```python exec="on" source="console"
from polykin.properties.vaporization_enthalpy import DHVL_pitzer

DHVL = DHVL_pitzer(T=273.15+50, Tc=425., w=0.122)

print(f"{DHVL/1e3:.1f} kJ/mol")
```
