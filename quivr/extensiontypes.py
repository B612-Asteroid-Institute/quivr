import enum
import pickle
from typing import Type

import pyarrow


class EnumType(pyarrow.ExtensionType):
    """A PyArrow extension type for storing enum values.

    Enum values are stored as a dictionary, mapping the enum values to
    a compressed set.

    Examples
    --------

    >>> import pyarrow as pa
    >>> import quivr.extensiontypes as qet
    >>> import enum
    >>> class Color(enum.Enum):
    ...     RED = "red"
    ...     GREEN = "green"
    ...     BLUE = "blue"
    ...
    >>> array = qet.EnumArray.from_enum_list(Color, [Color.RED, Color.GREEN, Color.BLUE, Color.RED])
    >>> print(array)
    -- dictionary:
      [
        "red",
        "green",
        "blue"
      ]
    -- indices:
      [
        0,
        1,
        2,
        0
      ]
    >>> array.to_pylist()
    [<Color.RED: 'red'>, <Color.GREEN: 'green'>, <Color.BLUE: 'blue'>, <Color.RED: 'red'>]
    >>> array.to_pandas()
    0      red
    1    green
    2     blue
    3      red
    dtype: category
    Categories (3, object): ['red', 'green', 'blue']

    """

    def __init__(self, enum_class: Type[enum.Enum]):
        self.enum_class = enum_class

        # Ensure enum_class is pickleable
        try:
            pickle.dumps(enum_class)
        except Exception as e:
            print(self.__repr__())
            raise ValueError("Enum class must be pickleable") from e

        # Leave a bit of extra room in each case to allow for
        # expansion of the enum
        if len(enum_class) < 2**6:
            index_type = pyarrow.int8()
        elif len(enum_class) < 2**14:
            index_type = pyarrow.int16()
        elif len(enum_class) < 2**30:
            index_type = pyarrow.int32()
        elif len(enum_class) < 2**62:
            index_type = pyarrow.int64()
        else:
            raise ValueError("Too many enum values")

        if all(isinstance(value.value, str) for value in enum_class):
            value_type = pyarrow.string()
        elif all(isinstance(value.value, int) for value in enum_class):
            value_type = pyarrow.int64()
        else:
            raise ValueError("Enum values must be all strings or all integers")

        storage_type = pyarrow.dictionary(index_type, value_type)

        pyarrow.ExtensionType.__init__(self, storage_type, "quivr.enum")

    def __arrow_ext_serialize__(self):
        data = {
            "enum_class": self.enum_class,
        }
        return pickle.dumps(data)

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        data = pickle.loads(serialized)
        assert isinstance(data, dict)
        assert "enum_class" in data
        return cls(data["enum_class"])

    def __arrow_ext_class__(self):
        return EnumArray

    def __arrow_ext_scalar_class__(self):
        return EnumScalar


class EnumArray(pyarrow.ExtensionArray):
    @classmethod
    def from_pylist(cls, enum_class: Type[enum.Enum], values: list[enum.Enum]):
        ext_type = EnumType(enum_class)
        # Peek at the first value to determine if they are actual enums
        if isinstance(values[0], enum.Enum):
            values = [value.value for value in values]
        storage = pyarrow.array(values, ext_type.storage_type)
        return cls.from_storage(ext_type, storage)


class EnumScalar(pyarrow.ExtensionScalar):
    def as_py(self):
        ext_type = self.type
        return ext_type.enum_class(self.value.as_py())

    @classmethod
    def from_enum(cls, enum_class: Type[enum.Enum], value: enum.Enum):
        ext_type = EnumType(enum_class)
        storage = pyarrow.scalar(value.value, ext_type.storage_type)
        return cls.from_storage(ext_type, storage)
