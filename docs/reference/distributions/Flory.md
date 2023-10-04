# polykin.distributions.Flory

::: polykin.distributions.analyticaldistributions
    options:
        members:
            - Flory

## Example

Define a Flory distribution and evaluate the corresponding probability density function and
cumulative distribution function for representative chain lengths.

```python exec="on" source="console"
from polykin.distributions import Flory

a = Flory(100, M0=0.050, name='A')

print(a.Mz)
print(a.pdf(a.DPn))
print(a.cdf([a.DPn, a.DPw, a.DPz]))
```

