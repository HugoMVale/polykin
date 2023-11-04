# polykin.distributions

::: polykin.distributions.datadistribution
    options:
        members:
            - DataDistribution

## Examples

```python exec="on" source="console"
from polykin.distributions import DataDistribution
from polykin.distributions import sample_mmd

a = DataDistribution(
    sample_mmd['size_data'], sample_mmd['pdf_data'],
    kind=sample_mmd['kind'], name='sample-X')

print(a.Mz)
print(a.pdf(a.DPn))
print(a.cdf([a.DPn, a.DPw, a.DPz]))
```
