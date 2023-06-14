import enum

import numpy.testing as npt
import pyarrow
import pytest

from quivr import extensiontypes


# Example of an enum with string values
class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


# Example of an enum with int values
class IPVersion(enum.Enum):
    IPV4 = 4
    IPV6 = 6


# Example of an enum with mixed values
class MixedEnum(enum.Enum):
    ONE = 1
    TWO = "two"
    THREE = 3


def test_enum_extension_type_creation():
    et = extensiontypes.EnumType(Color)
    assert pyarrow.types.is_string(
        et.storage_type.value_type
    ), "An enum with string values should have a string storage type"
    assert pyarrow.types.is_dictionary(et.storage_type), "An enum should have a dictionary storage type"
    assert pyarrow.types.is_integer(et.storage_type.index_type), "An enum should have an integer index type"

    et = extensiontypes.EnumType(IPVersion)
    assert pyarrow.types.is_int64(
        et.storage_type.value_type
    ), "An enum with int values should have an int storage type"
    assert pyarrow.types.is_dictionary(et.storage_type), "An enum should have a dictionary storage type"
    assert pyarrow.types.is_integer(et.storage_type.index_type), "An enum should have an integer index type"

    with pytest.raises(ValueError, match="Enum values must be all strings or all integers"):
        et = extensiontypes.EnumType(MixedEnum)


class TestEnumArray:
    def test_from_pylist_string_enum_values(self):
        values = [Color.RED, Color.GREEN, Color.BLUE, Color.RED, Color.GREEN, Color.BLUE]
        arr = extensiontypes.EnumArray.from_pylist(Color, values)
        assert arr.type == extensiontypes.EnumType(Color)
        assert arr.to_pylist() == values
        npt.assert_array_equal(
            arr.to_numpy(zero_copy_only=False), ["red", "green", "blue", "red", "green", "blue"]
        )

    def test_from_pylist_string_scalar_values(self):
        values = ["red", "green", "blue", "red", "green", "blue"]
        arr = extensiontypes.EnumArray.from_pylist(Color, values)
        assert arr.type == extensiontypes.EnumType(Color)
        assert arr.to_pylist() == [Color.RED, Color.GREEN, Color.BLUE, Color.RED, Color.GREEN, Color.BLUE]
        npt.assert_array_equal(
            arr.to_numpy(zero_copy_only=False), ["red", "green", "blue", "red", "green", "blue"]
        )

    def test_from_pylist_int_enum_values(self):
        values = [IPVersion.IPV4, IPVersion.IPV6, IPVersion.IPV4, IPVersion.IPV6]
        arr = extensiontypes.EnumArray.from_pylist(IPVersion, values)
        assert arr.type == extensiontypes.EnumType(IPVersion)
        assert arr.to_pylist() == values
        npt.assert_array_equal(arr.to_numpy(zero_copy_only=False), [4, 6, 4, 6])

    def test_from_pylist_int_scalar_values(self):
        values = [4, 6, 4, 6]
        arr = extensiontypes.EnumArray.from_pylist(IPVersion, values)
        assert arr.type == extensiontypes.EnumType(IPVersion)
        assert arr.to_pylist() == [IPVersion.IPV4, IPVersion.IPV6, IPVersion.IPV4, IPVersion.IPV6]
        npt.assert_array_equal(arr.to_numpy(zero_copy_only=False), [4, 6, 4, 6])
