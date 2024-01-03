# polykin.copolymerization

::: polykin.copolymerization
    options:
        members:
            - inst_copolymer_ternary

## Examples

```python exec="on" source="material-block"
from polykin.copolymerization import inst_copolymer_ternary

F1, F2, F3 = inst_copolymer_ternary(f1=0.5, f2=0.3, r12=0.2, r21=2.3,
                                    r13=3.0, r31=0.9, r23=0.4, r32=1.5)

print(f"F1 = {F1:.2f}; F2 = {F2:.2f}; F3 = {F3:.2f}")
```
