# Installation

`PolyKin` requires Python>=3.9, because it makes use of recent type hint syntax. It further
relies on a number of mature and well-maintained mathematical/scientific libraries:
`matplotlib`, `mpmath`, `numba`, `numpy`, `pydantic`, `scipy`, etc.

## From PyPI

With `pip` do:

```
pip install polykin
```

With `poetry` do:
```
poetry add polykin
```

## From the source code repository

The very latest code may be installed directly from the source code repository:
```
git clone https://github.com/HugoMVale/polykin.git
cd polykin
pip install . 
# poetry install
```