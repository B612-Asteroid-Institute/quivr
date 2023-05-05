from typing import TYPE_CHECKING, Generic, Optional, TypeAlias, TypeVar, Union

import pyarrow as pa

if TYPE_CHECKING:
    from .tables import Table

Byteslike: TypeAlias = Union[bytes, bytearray, memoryview, str]
MetadataDict: TypeAlias = dict[Byteslike, Byteslike]


class Field:
    """
    A Field is an accessor for data in a Table, and also a descriptor for the Table's structure.
    """

    def __init__(self, dtype: pa.DataType, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        self.dtype = dtype
        self.nullable = nullable
        self.metadata = metadata

    def __get__(self, obj: "Table", objtype: type):
        return obj.table.column(self.name)

    def __set__(self, obj: "Table", value):
        idx = obj.table.schema.get_field_index(self.name)
        obj.table = obj.table.set_column(idx, self.pyarrow_field(), [value])

    def __set_name__(self, owner: type, name: str):
        self.name = name

    def pyarrow_field(self):
        return pa.field(self.name, self.dtype, self.nullable, self.metadata)


T = TypeVar("T", bound="Table")


class SubTableField(Field, Generic[T]):
    """
    A field which represents an embedded Quivr table.
    """

    def __init__(self, table_type: type[T], nullable: bool = True, metadata: Optional[MetadataDict] = None):
        self.table_type = table_type
        self.schema = table_type.schema
        dtype = pa.struct(table_type.schema)
        super().__init__(dtype, nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> T:
        array = obj.table.column(self.name)
        subtable = pa.Table.from_arrays(array.flatten(), schema=self.schema)
        return self.table_type(subtable)


class Int8Field(Field):
    """
    A field for storing 8-bit integers.
    """

    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.int8(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.Int8Array:
        return obj.table[self.name].combine_chunks()


class Int16Field(Field):
    """
    A field for storing 16-bit integers.
    """

    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.int16(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.Int16Array:
        return obj.table[self.name].combine_chunks()


class Int32Field(Field):
    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.int32(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.Int32Array:
        return obj.table[self.name].combine_chunks()


class Int64Field(Field):
    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.int64(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.Int64Array:
        return obj.table[self.name].combine_chunks()


class UInt8Field(Field):
    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.uint8(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.UInt8Array:
        return obj.table[self.name].combine_chunks()


class UInt16Field(Field):
    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.uint16(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.UInt16Array:
        return obj.table[self.name].combine_chunks()


class UInt32Field(Field):
    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.uint32(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.UInt32Array:
        return obj.table[self.name].combine_chunks()


class UInt64Field(Field):
    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.uint64(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.UInt64Array:
        return obj.table[self.name].combine_chunks()


class Float16Field(Field):
    """
    A field for storing 16-bit floating point numbers.
    """

    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.float16(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.lib.HalfFloatArray:
        return obj.table[self.name].combine_chunks()


class Float32Field(Field):
    """
    A field for storing 32-bit floating point numbers.
    """

    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.float32(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.lib.FloatArray:
        return obj.table[self.name].combine_chunks()


class Float64Field(Field):
    """
    A class representing a 64-bit floating-point field in a table.

    Attributes
    ----------
    name : str
        The name of the field.
    type : pyarrow.float64
        The data type of the field, which is a 64-bit floating-point number.
    nullable : bool, optional, default True
        Whether the field allows null values.
    metadata : MetadataDict, optional, default None
        Additional metadata associated with the field.

    Methods
    -------
    __init__(nullable: bool = True, metadata: Optional[MetadataDict] = None)
        Initialize the Float64Field instance.
    __get__(obj: "Table", objtype: type) -> pa.DoubleArray
        Get the field's data as a pyarrow.DoubleArray object.

    Examples
    --------
    >>> from pyarrow import Table
    >>> import pandas as pd
    >>> data = {'float_field': [1.0, 2.0, 3.0]}
    >>> df = pd.DataFrame(data)
    >>> table = Table.from_pandas(df)
    >>> float_field = Float64Field(name='float_field')
    >>> float_array = float_field.__get__(table, Table)
    >>> float_array
    <pyarrow.lib.DoubleArray object at 0x7f8e6f1e19a0>
    [1.0, 2.0, 3.0]
    """

    """
    A field for storing 64-bit floating point numbers.
    """

    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.float64(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.lib.DoubleArray:
        return obj.table[self.name].combine_chunks()


class StringField(Field):
    """A field for storing strings.

    This can be used to store strings of any length, but it is not
    recommended for storing very long strings (over 2GB, for
    example). For long strings, use LargeStringField instead.

    """

    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.string(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.StringArray:
        return obj.table[self.name].combine_chunks()


class LargeBinaryField(Field):
    """
    A field for storing large binary objects. Large binary data is stored in
    variable-length chunks.
    """

    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.large_binary(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.LargeBinaryArray:
        return obj.table[self.name].combine_chunks()


class LargeStringField(Field):
    """
    A field for storing large strings. Large string data is stored in
    variable-length chunks.
    """

    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.large_string(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.LargeStringArray:
        return obj.table[self.name].combine_chunks()


class Date32Field(Field):
    """A field for storing dates.

    Internally, this field stores dates as 32-bit integers which
    represent time since the UNIX epoch.

    """

    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.date32(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.Date32Array:
        return obj.table[self.name].combine_chunks()


class Date64Field(Field):
    """A field for storing dates.

    Internally, this field stores dates as 64-bit integers which
    represent time since the UNIX epoch in milliseconds, where the
    values are evenly divisible by 86,400,000.
    """

    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.date64(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.Date64Array:
        return obj.table[self.name].combine_chunks()


class TimestampField(Field):
    """A field for storing timestamps.

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
    ):
        super().__init__(pa.timestamp(unit, tz), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.TimestampArray:
        return obj.table[self.name].combine_chunks()


class Time32Field(Field):
    """
    A field for storing time values.

    Time data can be stored in one of two units for this 32-bit type:
        - seconds
        - milliseconds

    Internally, a time32 value is a 32-bit integer which an elapsed time since midnight.
    """

    def __init__(self, unit: str, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.time32(unit), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.Time32Array:
        return obj.table[self.name].combine_chunks()


class Time64Field(Field):
    """
    A field for storing time values with high precision.

    Time data can be stored in one of two units for this 64-bit type:
        - microseconds
        - nanoseconds

    Internally, a time64 value is a 64-bit integer which an elapsed time since midnight.
    """

    def __init__(self, unit: str, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.time64(unit), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.Time64Array:
        return obj.table[self.name].combine_chunks()


class DurationField(Field):
    """
    An absolute length of time unrelated to any calendar artifacts.

    The resolution defaults to millisecond, but can be any of the other
    supported time unit values (seconds, milliseconds, microseconds,
    nanoseconds).

    Internall, a duration value is always represented as an 8-byte integer.
    """

    def __init__(self, unit: str, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.duration(unit), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.DurationArray:
        return obj.table[self.name].combine_chunks()


class MonthDayNanoIntervalField(Field):
    """A field for storing calendar intervals (an elapsed number of months,
    days, and nanoseconds).

    Internally, a month_day_nano_interval value is a 96-bit
    integer. 32 bits are devoted to months and days, and 64 bits are
    devoted to nanoseconds.

    Leap seconds are ignored.

    """

    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.month_day_nano_interval(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.MonthDayNanoIntervalArray:
        return obj.table[self.name].combine_chunks()


class BinaryField(Field):
    """A field for storing opaque binary data.

    The data can be either variable-length or fixed-length, depending
    on the 'length' parameter passed in the initializer.

    If length is -1 (the default) then the data is variable-length. If
    length is greater than or equal to 0, then the data is
    fixed-length.

    """

    def __init__(self, length: int = -1, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.binary(length), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.BinaryArray:
        return obj.table[self.name].combine_chunks()


class Decimal128Field(Field):
    """A field for storing arbitrary-precision decimal numbers.

    Arrow decimals are fixed-point decimal numbers encoded as a scaled
    integer. The precision is the number of significant digits that the
    decimal type can represent; the scale is the number of digits after
    the decimal point (note the scale can be negative).

    As an example, Decimal128Field(7, 3) can exactly represent the numbers
    1234.567 and -1234.567 (encoded internally as the 128-bit integers
    1234567 and -1234567, respectively), but neither 12345.67 nor
    123.4567.

    DecimalField(5, -3) can exactly represent the number 12345000
    (encoded internally as the 128-bit integer 12345), but neither
    123450000 nor 1234500.

    If you need a precision higher than 38 significant digits,
    consider using Decimal256Field.

    """

    def __init__(
        self, precision: int, scale: int, nullable: bool = True, metadata: Optional[MetadataDict] = None
    ):
        super().__init__(pa.decimal128(precision, scale), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.Decimal128Array:
        return obj.table[self.name].combine_chunks()


class Decimal256Field(Field):
    """A field for storing arbitrary-precision decimal numbers.

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

    def __get__(self, obj: "Table", objtype: type) -> pa.Decimal256Array:
        return obj.table[self.name].combine_chunks()


class NullField(Field):
    """A field for storing null values.

    Nulls are represented as a single bit, and do not take up any
    memory space.

    """

    def __init__(self, nullable: bool = True, metadata: Optional[MetadataDict] = None):
        super().__init__(pa.null(), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.NullArray:
        return obj.table[self.name].combine_chunks()


# Complex types follow


class ListField(Field):
    """A field for storing lists of values.

    The values in the list can be of any type.

    Note that all quivr Tables are storing lists of values, so this
    field type is only useful for storing lists of lists.



    """

    def __init__(
        self,
        value_type: Union[pa.DataType, pa.Field, Field],
        list_size: int = -1,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
    ):
        """
        Parameters
        ----------
        value_type : Union[pa.DataType, pa.Field, Field]
            The type of the values in the list.

        list_size : int
            The size of the list. If -1, then the list is variable-length.

        nullable : bool
            Whether the list can contain null values.

        metadata : Optional[MetadataDict]
            A dictionary of metadata to attach to the field.
        """
        if isinstance(value_type, Field):
            value_type = value_type.dtype
        super().__init__(pa.list_(value_type, list_size), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.ListArray:
        return obj.table[self.name].combine_chunks()


class LargeListField(Field):
    """A field for storing large lists of values.

    Unless you need to represent data with more than 2**31 elements,
    prefer ListField.

    The values in the list can be of any type.

    Note that all quivr Tables are storing lists of values, so this
    field type is only useful for storing lists of lists.
    """

    def __init__(
        self,
        value_type: Union[pa.DataType, pa.Field, Field],
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
    ):
        """
        Parameters
        ----------
        value_type : Union[pa.DataType, pa.Field, Field]
            The type of the values in the list.

        nullable : bool
            Whether the list can contain null values.

        metadata : Optional[MetadataDict]
            A dictionary of metadata to attach to the field.
        """
        if isinstance(value_type, Field):
            value_type = value_type.dtype
        super().__init__(pa.large_list(value_type), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.LargeListArray:
        return obj.table[self.name].combine_chunks()


class MapField(Field):
    """A field for storing maps of key-value pairs.

    The keys and values can be of any type, as long as the keys are
    hashable and unique.
    """

    def __init__(
        self,
        key_type: Union[pa.DataType, pa.Field, Field],
        item_type: Union[pa.DataType, pa.Field, Field],
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
    ):
        """
        Parameters
        ----------
        key_type : Union[pa.DataType, pa.Field, Field]
            The type of the keys in the map.

        item_type : Union[pa.DataType, pa.Field, Field]
            The type of the values in the map.

        nullable : bool
            Whether the map can contain null values.

        metadata : Optional[MetadataDict]
            A dictionary of metadata to attach to the field.
        """
        if isinstance(key_type, Field):
            key_type = key_type.dtype
        if isinstance(item_type, Field):
            item_type = item_type.dtype
        super().__init__(pa.map_(key_type, item_type), nullable=nullable, metadata=metadata)

    def __get__(self, obj: "Table", objtype: type) -> pa.MapArray:
        return obj.table[self.name].combine_chunks()


class DictionaryField(Field):
    """A field for storing dictionary-encoded values.

    This is intended for use with categorical data. See MapField for a
    more general mapping type.
    """

    def __init__(
        self,
        index_type: pa.DataType,
        value_type: Union[pa.DataType, pa.Field, Field],
        ordered: bool = False,
        nullable: bool = True,
        metadata: Optional[MetadataDict] = None,
    ):
        """
        Parameters
        ----------
        index_type : IntegerDataType
            The type of the dictionary indices. Must be an integer type.

        value_type : Union[pa.DataType, pa.Field, Field]
            The type of the values in the dictionary.

        ordered : bool
            Whether the dictionary is ordered.

        nullable : bool
            Whether the dictionary can contain null values.

        metadata : Optional[MetadataDict]
            A dictionary of metadata to attach to the field.
        """
        if isinstance(index_type, Field):
            index_type = index_type.dtype
        if isinstance(value_type, Field):
            value_type = value_type.dtype
        super().__init__(
            pa.dictionary(index_type, value_type, ordered=ordered), nullable=nullable, metadata=metadata
        )

    def __get__(self, obj: "Table", objtype: type) -> pa.DictionaryArray:
        return obj.table[self.name].combine_chunks()


class StructField(Field):
    """A field for storing structured data.

    In general, prefer to define Tables and use their as_field method
    instead of using StructField.

    """

    def __init__(
        self, fields: list[pa.Field], nullable: bool = True, metadata: Optional[MetadataDict] = None
    ):
        """
        Parameters
        ----------
        fields : List[pa.Field]
            The fields in the struct.

        nullable : bool
            Whether the struct can contain null values.

        metadata : Optional[MetadataDict]
            A dictionary of metadata to attach to the field.
        """
        super().__init__(pa.struct(fields), nullable=nullable, metadata=metadata)


class RunEndEncodedField(Field):
    """A field for storing run-end encoded data.

    This is a special field type that is used to efficiently store
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
            A dictionary of metadata to attach to the field.
        """
        super().__init__(pa.run_end_encoded(run_end_type, value_type), nullable=nullable, metadata=metadata)
