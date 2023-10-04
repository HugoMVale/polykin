# polykin.distributions.SchulzZimm

::: polykin.distributions.analyticaldistributions
    options:
        members:
            - SchulzZimm

## Example

Define a SchulzZimm distribution and evaluate the corresponding probability density function and
cumulative distribution function for representative chain lengths.

```python exec="on" source="console"
from polykin.distributions import SchulzZimm

a = SchulzZimm(100, PDI=3., M0=0.050, name='A')

print(a.Mz)
print(a.pdf(a.DPn))
print(a.cdf([a.DPn, a.DPw, a.DPz]))
```
