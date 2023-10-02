# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2023

from polykin.utils import check_type


class Base():
    """Abstract base class."""

    @property
    def name(self) -> str:
        """Name of the object."""
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = check_type(name, str, "name")
        # if name != "":
        #     # full_name = f"{name} ({self.__class__.__name__})"
        #     full_name = name
        # else:
        #     full_name = self.__class__.__name__
        # self.__name = full_name
