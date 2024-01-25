# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023


# %% Exceptions


class RangeWarning(Warning):
    pass


class RangeError(ValueError):
    pass


class ShapeError(ValueError):
    pass


class FitError(ValueError):
    pass


class RootSolverError(ValueError):
    pass


class ODESolverError(ValueError):
    pass
