# polykin.properties.vaporization_enthalpy

::: polykin.properties.vaporization_enthalpy
    options:
        members:
            - hvapb_vetere

## Examples

Estimate the vaporization enthalpy of vinyl chloride at the normal boiling temperature.

```python exec="on" source="console"
from polykin.properties.vaporization_enthalpy import hvapb_vetere

hvap = hvapb_vetere(Tb=259.8, Tc=425., Pc=51.5e5)

print(hvap/1e3, "kJ/mol")
```
