# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023


# %% Exceptions

class ConvergenceError(ValueError):
    pass


class FitError(ValueError):
    pass


class ODESolverError(ValueError):
    pass


class RangeWarning(Warning):
    pass


class RangeError(ValueError):
    pass


class RootSolverError(ValueError):
    pass


class ShapeError(ValueError):
    pass
