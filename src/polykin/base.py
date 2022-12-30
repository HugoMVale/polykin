from utils import check_type


class Base():
    """Abstract base class."""

    @property
    def name(self) -> str:
        """Name of the object."""
        return self.__name

    @name.setter
    def name(self, name: str):
        check_type(name, str, "name")
        if name != "":
            full_name = f"{name} ({self.__class__.__name__})"
        else:
            full_name = self.__class__.__name__
        self.__name = full_name
