# polykin.properties.vaporization_enthalpy

::: polykin.properties.vaporization_enthalpy
    options:
        members:
            - DHVL_vetere

## Examples

Estimate the vaporization enthalpy of vinyl chloride at the normal boiling temperature.

```python exec="on" source="console"
from polykin.properties.vaporization_enthalpy import DHVL_vetere

DHVL = DHVL_vetere(Tb=259.8, Tc=425., Pc=51.5e5)

print(DHVL/1e3, "kJ/mol")
```
