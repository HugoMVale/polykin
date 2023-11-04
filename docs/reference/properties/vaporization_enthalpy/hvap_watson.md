# polykin.properties.vaporization_enthalpy

::: polykin.properties.vaporization_enthalpy
    options:
        members:
            - hvap_watson

## Examples

Estimate the vaporization enthalpy of vinyl chloride at 50°C from the known value at the normal
boiling temperature.

```python exec="on" source="console"
from polykin.properties.vaporization_enthalpy import hvap_watson

hvap = hvap_watson(hvap1=22.9, T1=258., T2=273.15+50, Tc=425.)

print(hvap, "kJ/mol")
```
