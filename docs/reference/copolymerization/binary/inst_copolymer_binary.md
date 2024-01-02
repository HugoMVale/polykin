# polykin.copolymerization

::: polykin.copolymerization.binary
    options:
        members:
            - inst_copolymer_binary

## Examples

```python exec="on" source="material-block"
from polykin.copolymerization import inst_copolymer_binary

F1 = inst_copolymer_binary(f1=0.5, r1=0.16, r2=0.70)

print(f"F1 = {F1:.2f}")
```
