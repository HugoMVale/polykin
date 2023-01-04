"""A polymerization kinetics framework"""

from pathlib import Path

from .distributions import Flory, Poisson

__version__ = (Path(__file__).resolve().parent / "_version.txt").read_text(
    encoding="utf-8")
