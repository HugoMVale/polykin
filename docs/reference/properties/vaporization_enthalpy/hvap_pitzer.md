# polykin.properties.vaporization_enthalpy

::: polykin.properties.vaporization_enthalpy
    options:
        members:
            - hvap_pitzer

## Examples

Estimate the vaporization enthalpy of vinyl chloride at 50°C.

```python exec="on" source="console"
from polykin.properties.vaporization_enthalpy import hvap_pitzer

hvap = hvap_pitzer(T=273.15+50, Tc=425., w=0.122)

print(hvap/1e3, "kJ/mol")
```
