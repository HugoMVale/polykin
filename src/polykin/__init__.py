"""A polymerization kinetics framework"""

from pathlib import Path

__version__ = (Path(__file__).resolve().parent / "_version.txt").read_text(
    encoding="utf-8")
