# polykin.thermo.eos

::: polykin.thermo.eos.idealgas
    options:
        show_bases: true
        members:
            - IdealGas

## Examples

Estimate the molar volume of a gas at 0°C and 1 atm.

```python exec="on" source="material-block"
from polykin.thermo.eos import IdealGas

eos = IdealGas(1)
v = eos.v(T=273.15, P=1.01325e5, y=None)

print(f"{v:.2e} m³/mol")
```
