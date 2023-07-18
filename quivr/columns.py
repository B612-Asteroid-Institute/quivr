from __future__ import annotations

import sys

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from typing import TYPE_CHECKING, Generic, Optional, TypeVar, Union, overload

import pyarrow as pa

from . import validators

if TYPE_CHECKING:
    from .tables import Table

Byteslike: TypeAlias = Union[bytes, bytearray, memoryview, str]
MetadataDict: TypeAlias = dict[Byteslike, Byteslike]


class Column:
    """
    A Column is an accessor for data in a Table, and also a descriptor for the Table's structure.
    """

    def __init__(
        self,
        dtype: pa.DataType,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        self.dtype = dtype
        self.nullable = nullable
        self.metadata = metadata
        self.validator = validator

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: Table, objtype: type) -> pa.Array:
        ...

    def __get__(self, obj: Union[Table, None], objtype: type) -> Union[Self, pa.Array]:
        if obj is None:
            return self
        return obj.table.column(self.name)

    def __set__(self, obj: Table, value: pa.Array) -> None:
        idx = obj.table.schema.get_field_index(self.name)
        obj.table = obj.table.set_column(idx, self.pyarrow_field(), [value])

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def pyarrow_field(self) -> pa.Field:
        return pa.field(self.name, self.dtype, self.nullable, self.metadata)


T = TypeVar("T", bound="Table")


class SubTableColumn(Column, Generic[T]):
    """
    A column which represents an embedded Quivr table.
    """

    def __init__(self, table_type: type[T], nullable: bool = True, metadata: Optional[MetadataDict] = None):
        self.table_type = table_type
        self.schema = table_type.schema
        dtype = pa.struct(table_type.schema)
        super().__init__(dtype, nullable=nullable, metadata=metadata)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: Table, objtype: type) -> T:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, T]:
        if obj is None:
            return self
        array = obj.table.column(self.name)

        metadata = self.metadata
        if metadata is None:
            metadata = {}
        metadata.update(obj._metadata_for_column(self.name))  # type: ignore

        schema = self.schema.with_metadata(metadata)

        subtable = pa.Table.from_arrays(array.flatten(), schema=schema)
        return self.table_type(subtable)


class Int8Column(Column):
    """
    A column for storing 8-bit integers.
    """

    def __init__(
        self,
        nullable: bool = True,
        validator: Optional[validators.Validator] = None,
        metadata: Optional[MetadataDict] = None,
    ):
        super().__init__(pa.int8(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.Int8Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.Int8Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class Int16Column(Column):
    """
    A column for storing 16-bit integers.
    """

    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.int16(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.Int16Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.Int16Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class Int32Column(Column):
    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.int32(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.Int32Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.Int32Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class Int64Column(Column):
    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.int64(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.Int64Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.Int64Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class UInt8Column(Column):
    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.uint8(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.UInt8Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.UInt8Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class UInt16Column(Column):
    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.uint16(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.UInt16Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.UInt16Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class UInt32Column(Column):
    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.uint32(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.UInt32Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.UInt32Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class UInt64Column(Column):
    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.uint64(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.UInt64Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.UInt64Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class Float16Column(Column):
    """
    A column for storing 16-bit floating point numbers.
    """

    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.float16(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.lib.HalfFloatArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.lib.HalfFloatArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class Float32Column(Column):
    """
    A column for storing 32-bit floating point numbers.
    """

    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.float32(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.lib.FloatArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.lib.FloatArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class Float64Column(Column):
    """
    A column for storing 64-bit floating point numbers.
    """

    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.float64(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.lib.DoubleArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.lib.DoubleArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class StringColumn(Column):
    """A column for storing strings.

    This can be used to store strings of any length, but it is not
    recommended for storing very long strings (over 2GB, for
    example). For long strings, use LargeStringColumn instead.

    """

    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.string(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: Table, objtype: type) -> pa.StringArray:
        ...

    def __get__(self, obj: Optional[Table], objtype: type) -> Union[pa.StringArray, Self]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class LargeBinaryColumn(Column):
    """
    A column for storing large binary objects. Large binary data is stored in
    variable-length chunks.
    """

    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.large_binary(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.LargeBinaryArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.LargeBinaryArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class LargeStringColumn(Column):
    """
    A column for storing large strings. Large string data is stored in
    variable-length chunks.
    """

    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.large_string(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.LargeStringArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.LargeStringArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class Date32Column(Column):
    """A column for storing dates.

    Internally, this column stores dates as 32-bit integers which
    represent time since the UNIX epoch.

    """

    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.date32(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.Date32Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.Date32Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class Date64Column(Column):
    """A column for storing dates.

    Internally, this column stores dates as 64-bit integers which
    represent time since the UNIX epoch in milliseconds, where the
    values are evenly divisible by 86,400,000.
    """

    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.date64(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.Date64Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.Date64Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class TimestampColumn(Column):
    """A column for storing timestamps.

    Timestamp data can be stored in one of four units:
      - seconds
      - milliseconds
      - microseconds
      - nanoseconds

    Internally, a timestamp is a 64-bit integer which represents time
    since 1970-01-01 00:00:00 UTC (the UNIX epoch), and is optionally
    annotated with a timezone.

    Timestamp values do not include any leap seconds (in other words,
    all days are considered 86,400 seconds long).
    """

    def __init__(
        self,
        unit: str,
        tz: Optional[str] = None,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.timestamp(unit, tz), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.TimestampArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.TimestampArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class Time32Column(Column):
    """
    A column for storing time values.

    Time data can be stored in one of two units for this 32-bit type:
        - seconds
        - milliseconds

    Internally, a time32 value is a 32-bit integer which an elapsed time since midnight.
    """

    def __init__(
        self,
        unit: str,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.time32(unit), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.Time32Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.Time32Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class Time64Column(Column):
    """
    A column for storing time values with high precision.

    Time data can be stored in one of two units for this 64-bit type:
        - microseconds
        - nanoseconds

    Internally, a time64 value is a 64-bit integer which an elapsed time since midnight.
    """

    def __init__(
        self,
        unit: str,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.time64(unit), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.Time64Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.Time64Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class DurationColumn(Column):
    """
    An absolute length of time unrelated to any calendar artifacts.

    The resolution defaults to millisecond, but can be any of the other
    supported time unit values (seconds, milliseconds, microseconds,
    nanoseconds).

    Internall, a duration value is always represented as an 8-byte integer.
    """

    def __init__(
        self,
        unit: str,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.duration(unit), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.DurationArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.DurationArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class MonthDayNanoIntervalColumn(Column):
    """A column for storing calendar intervals (an elapsed number of months,
    days, and nanoseconds).

    Internally, a month_day_nano_interval value is a 96-bit
    integer. 32 bits are devoted to months and days, and 64 bits are
    devoted to nanoseconds.

    Leap seconds are ignored.

    """

    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(
            pa.month_day_nano_interval(), nullable=nullable, metadata=metadata, validator=validator
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.MonthDayNanoIntervalArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.MonthDayNanoIntervalArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class BinaryColumn(Column):
    """A column for storing opaque binary data.

    The data can be either variable-length or fixed-length, depending
    on the 'length' parameter passed in the initializer.

    If length is -1 (the default) then the data is variable-length. If
    length is greater than or equal to 0, then the data is
    fixed-length.

    """

    def __init__(
        self,
        length: int = -1,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.binary(length), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.BinaryArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.BinaryArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class Decimal128Column(Column):
    """A column for storing arbitrary-precision decimal numbers.

    Arrow decimals are fixed-point decimal numbers encoded as a scaled
    integer. The precision is the number of significant digits that the
    decimal type can represent; the scale is the number of digits after
    the decimal point (note the scale can be negative).

    As an example, Decimal128Column(7, 3) can exactly represent the numbers
    1234.567 and -1234.567 (encoded internally as the 128-bit integers
    1234567 and -1234567, respectively), but neither 12345.67 nor
    123.4567.

    DecimalColumn(5, -3) can exactly represent the number 12345000
    (encoded internally as the 128-bit integer 12345), but neither
    123450000 nor 1234500.

    If you need a precision higher than 38 significant digits,
    consider using Decimal256Column.

    """

    def __init__(
        self, precision: int, scale: int, nullable: bool = True, metadata: Optional[MetadataDict] = None
    ):
        super().__init__(pa.decimal128(precision, scale), nullable=nullable, metadata=metadata)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.Decimal128Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.Decimal128Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class Decimal256Column(Column):
    """A column for storing arbitrary-precision decimal numbers.

    Arrow decimals are fixed-point decimal numbers encoded as a scaled
    integer. The precision is the number of significant digits that the
    decimal type can represent; the scale is the number of digits after
    the decimal point (note the scale can be negative).

    Values are stored as 256-bit integers.
    """

    def __init__(
        self, precision: int, scale: int, nullable: bool = True, metadata: Optional[MetadataDict] = None
    ):
        super().__init__(pa.decimal256(precision, scale), nullable=nullable, metadata=metadata)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.Decimal256Array:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.Decimal256Array]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class NullColumn(Column):
    """A column for storing null values.

    Nulls are represented as a single bit, and do not take up any
    memory space.

    """

    def __init__(
        self,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.null(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.NullArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.NullArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


# Complex types follow


class ListColumn(Column):
    """A column for storing lists of values.

    The values in the list can be of any type.

    Note that all quivr Tables are storing lists of values, so this
    column type is only useful for storing lists of lists.



    """

    def __init__(
        self,
        value_type: Union[pa.DataType, pa.Field, Column],
        list_size: int = -1,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        """
        Parameters
        ----------
        value_type : Union[pa.DataType, pa.Field, Column]
            The type of the values in the list.

        list_size : int
            The size of the list. If -1, then the list is variable-length.

        nullable : bool
            Whether the list can contain null values.

        metadata : Optional[MetadataDict]
            A dictionary of metadata to attach to the column.

        validator: Optional[validators.Validator]
            A validator to run against the column's values.
        """
        if isinstance(value_type, Column):
            value_type = value_type.dtype
        super().__init__(
            pa.list_(value_type, list_size), nullable=nullable, metadata=metadata, validator=validator
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.ListArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.ListArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class LargeListColumn(Column):
    """A column for storing large lists of values.

    Unless you need to represent data with more than 2**31 elements,
    prefer ListColumn.

    The values in the list can be of any type.

    Note that all quivr Tables are storing lists of values, so this
    column type is only useful for storing lists of lists.
    """

    def __init__(
        self,
        value_type: Union[pa.DataType, pa.Field, Column],
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        """
        Parameters
        ----------
        value_type : Union[pa.DataType, pa.Field, Column]
            The type of the values in the list.

        nullable : bool
            Whether the list can contain null values.

        metadata : Optional[MetadataDict]
            A dictionary of metadata to attach to the column.

        validator: Optional[validators.Validator]
            A validator to run against the column's values.
        """
        if isinstance(value_type, Column):
            value_type = value_type.dtype
        super().__init__(pa.large_list(value_type), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.LargeListArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.LargeListArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class MapColumn(Column):
    """A column for storing maps of key-value pairs.

    The keys and values can be of any type, as long as the keys are
    hashable and unique.
    """

    def __init__(
        self,
        key_type: Union[pa.DataType, pa.Field, Column],
        item_type: Union[pa.DataType, pa.Field, Column],
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        """
        Parameters
        ----------
        key_type : Union[pa.DataType, pa.Field, Column]
            The type of the keys in the map.

        item_type : Union[pa.DataType, pa.Field, Column]
            The type of the values in the map.

        nullable : bool
            Whether the map can contain null values.

        metadata : Optional[MetadataDict]
            A dictionary of metadata to attach to the column.

        validator: Optional[validators.Validator]
            A validator to run against the column's values.
        """
        if isinstance(key_type, Column):
            key_type = key_type.dtype
        if isinstance(item_type, Column):
            item_type = item_type.dtype
        super().__init__(
            pa.map_(key_type, item_type), nullable=nullable, metadata=metadata, validator=validator
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.MapArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.MapArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class DictionaryColumn(Column):
    """A column for storing dictionary-encoded values.

    This is intended for use with categorical data. See MapColumn for a
    more general mapping type.
    """

    def __init__(
        self,
        index_type: pa.DataType,
        value_type: Union[pa.DataType, pa.Field, Column],
        ordered: bool = False,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        """
        Parameters
        ----------
        index_type : IntegerDataType
            The type of the dictionary indices. Must be an integer type.

        value_type : Union[pa.DataType, pa.Field, Column]
            The type of the values in the dictionary.

        ordered : bool
            Whether the dictionary is ordered.

        nullable : bool
            Whether the dictionary can contain null values.

        metadata : Optional[MetadataDict]
            A dictionary of metadata to attach to the column.

        validator: Optional[validators.Validator]
            A validator to run against the column's values.
        """
        if isinstance(index_type, Column):
            index_type = index_type.dtype
        if isinstance(value_type, Column):
            value_type = value_type.dtype
        super().__init__(
            pa.dictionary(index_type, value_type, ordered=ordered),
            nullable=nullable,
            metadata=metadata,
            validator=validator,
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> pa.DictionaryArray:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[Self, pa.DictionaryArray]:
        if obj is None:
            return self
        return obj.table[self.name].combine_chunks()


class StructColumn(Column):
    """A column for storing structured data.

    In general, prefer to define Tables and use their as_column method
    instead of using StructColumn.

    """

    def __init__(
        self,
        fields: list[pa.Field],
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        """
        Parameters
        ----------
        fields : List[pa.Field]
            The fields in the struct.

        nullable : bool
            Whether the struct can contain null values.

        metadata : Optional[MetadataDict]
            A dictionary of metadata to attach to the column.

        validator: Optional[validators.Validator]
            A validator to run against the column's values.
        """
        super().__init__(pa.struct(fields), nullable=nullable, metadata=metadata, validator=validator)


class RunEndEncodedColumn(Column):
    """A column for storing run-end encoded data.

    This is a special column type that is used to efficiently store
    highly ordered data. Internally, the data is stored as two
    buffers:

    - An array of values, with all consecutive runs of the same value
      reduced to a single element
    - An array of run lengths, with the length of each run

    This is more compact than storing the redundant values and also
    allow for very efficient computations like aggregations upon the
    data.

    """

    def __init__(
        self,
        run_end_type: pa.DataType,
        value_type: pa.DataType,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        """
        Parameters
        ----------
        run_end_type : pa.DataType
            The type of the run-end encoded values. Must be a 16-, 32-, or 64-bit integer type.

        value_type : pa.DataType
            The type of the values in the run-end encoded data.

        nullable : bool
            Whether the data can contain null values.

        metadata : Optional[MetadataDict]
            A dictionary of metadata to attach to the column.

        validator: Optional[validators.Validator]
            A validator to run against the column's values.
        """
        super().__init__(
            pa.run_end_encoded(run_end_type, value_type),
            nullable=nullable,
            metadata=metadata,
            validator=validator,
        )


class ForeignKeyColumn(Column, Generic[T]):
    def __init__(self, table_type: type[T], on: Column):
        self.on = on
        self.table_type = table_type
        super().__init__(
            pa.list_(on.dtype),
            nullable=False,
            metadata=None,
            validator=None,
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> BoundForeignKeyColumn[T]:
        ...

    @overload
    def __get__(self, obj: "Table", objtype: type) -> T:
        ...

    def __get__(self, obj: Optional["Table"], objtype: type) -> Union[BoundForeignKeyColumn[T], pa.ListArray]:
        if obj is None:
            return self.bound_to(objtype)

        # Resolve the foreign key relationship
        fk = obj._foreign_keys[self.name]
        return ForeignKeyReference(fk.other_table, self.on, obj.table[self.name].combine_chunks())

    def bound_to(self, table_type: type[T]) -> BoundForeignKeyColumn[T]:
        return BoundForeignKeyColumn(self, table_type)
            


class BoundForeignKeyColumn(Column, Generic[T]):
    def __init__(self, foreign_key_col: ForeignKeyColumn[T], table_type: type[T]):
        self.foreign_key_col = foreign_key_col
        self.table_type = table_type

    def reference(self, other: T, data: Any) -> pa.ListArray:
        return ForeignKeyReference(other, self.foreign_key_col, data)

class ForeignKeyReference(Generic[T]):
    def __init__(self, other_table: T, other_table_column: Column, data: Any):
        self.other_table = other_table
        self.other_table_column = other_table_column
        self.data = data

    def __iter__(self):
        for indices in self.data:
            # ludicrously inefficient, oh well
            table = self.other_table.table.filter(
                pa.compute.field(self.other_table_column.name).isin(indices.as_py()),
            )
            yield self.other_table.__class__(table)

            
