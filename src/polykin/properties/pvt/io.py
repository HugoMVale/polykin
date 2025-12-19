# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2025

from pathlib import Path

import pandas as pd

table_parameters: dict[str, pd.DataFrame | None] = {}


def load_PVT_parameters(method: str) -> pd.DataFrame:
    """Load table with PVT parameters for a given equation."""
    global table_parameters
    table = table_parameters.get(method, None)
    if table is None:
        filepath = (Path(__file__).parent).joinpath(method + "_parameters.tsv")
        table = pd.read_csv(filepath, sep=r"\s+")
        table.set_index("Polymer", inplace=True)
        table_parameters[method] = table
    return table
