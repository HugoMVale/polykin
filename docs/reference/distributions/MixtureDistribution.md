# polykin.distributions.MixtureDistribution

::: polykin.distributions.baseclasses
    options:
        members:
            - MixtureDistribution

## Example

```python exec="on" source="console"
from polykin.distributions import Flory, SchulzZimm

a = Flory(100, M0=0.050, name='A')
b = SchulzZimm(100, PDI=3., M0=0.10, name='B')

c = 0.3*a + 0.7*b # c is now a MixtureDistribution instance

print(c.Mz)
print(c.pdf(c.DPn))
print(c.cdf([c.DPn, c.DPw, c.DPz]))
```
