# polykin.kinetics.emulsion

::: polykin.kinetics.emulsion.smithewart
    options:
        members:
            - nbar_Li_Brooks

### Benchmark

Illustration of the typical error obtained with the Li-Brooks approximation.

```python exec="on" source="above" html="on"
import matplotlib.pyplot as plt
import numpy as np
from polykin.kinetics.emulsion import nbar_Li_Brooks, nbar_Stockmayer_OToole
from polykin.utils.docs import to_html

fig, ax = plt.subplots()
alpha = np.logspace(-4, 4, 100)
error = np.zeros_like(alpha)
for m in [0, 0.1, 1, 10]:
    for i in range(len(alpha)):
        nbar_SO = nbar_Stockmayer_OToole(alpha[i], m)
        nbar_LB = nbar_Li_Brooks(alpha[i], m)
        error[i] = 1e2*(nbar_LB - nbar_SO)/nbar_SO
    ax.plot(alpha, error,  label=rf"$m={m}$")
ax.set_xscale("log")
ax.grid(True)
ax.legend(loc="best")
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Relative Error (%)");

print(to_html(fig))
```
