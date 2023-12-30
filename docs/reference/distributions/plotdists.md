# Distributions (polykin.distributions)

::: polykin.distributions.base
    options:
        members:
            - plotdists

## Examples

Define a LogNormal distribution and evaluate the corresponding probability density function and
cumulative distribution function for representative chain lengths.

```python exec="on" source="console"
from polykin.distributions import Flory, LogNormal, plotdists

a = Flory(100, M0=0.050, name='A')
b = LogNormal(100, PDI=3., M0=0.050, name='B')

fig = plotdists([a, b], kind='gpc', xrange=(1, 1e4), cdf=2)
```