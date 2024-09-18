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

import datetime
import decimal
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union, overload

import pyarrow as pa

from . import errors, tables, validators

Byteslike: TypeAlias = Union[bytes, str]
MetadataDict: TypeAlias = Dict[Byteslike, Byteslike]


class Column:
    """A Column is an accessor for data in a Table, and also a
    descriptor for the Table's structure.

    This is a base class for all column types. It is not intended to
    be used directly; instead, use one of its subclasses.

    Columns implement the descriptor protocol, so they should be used
    as class attributes on a Table subclass.

    :param dtype: The pyarrow data type of the column.
    :param nullable: Whether the column can contain null values.
    :param metadata: A dictionary of metadata to attach to the column.
    :param validator: A validator to use when setting the column.
    :param default: A default value to use when setting the column.

    :ivar dtype: The pyarrow data type of the column.
    :ivar nullable: Whether the column can contain null values.
    :ivar metadata: A dictionary of metadata to attach to the column.
    :ivar validator: A validator to use when setting the column.
    :ivar default: A default value to use when setting the column.
    """

    def __init__(
        self,
        dtype: pa.DataType,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, Callable[[], Any], Any] = None,
    ):
        self.dtype = dtype
        self.nullable = nullable
        self.metadata = metadata
        self.validator = validator

        self.default = default
        if self.default is not None:
            self._default_is_callable = callable(self.default)
            if self._default_is_callable:
                self._default_fn = self.default
            else:
                try:
                    self._default_scalar = pa.scalar(self.default, self.dtype)
                except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError) as e:
                    raise errors.InvalidColumnDefault(self.default, self.dtype) from e

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.Array: ...

    def __get__(self, obj: Union[tables.Table, None], objtype: type) -> Union[Self, pa.Array]:
        """

        Gets the Column object from a Table class, or the associated data from a Table instance.

        This method is part of the `descriptor protocol <https://docs.python.org/3/howto/descriptor.html>`_.

        """
        if obj is None:
            return self
        return obj.table.column(self.name)

    def __set_name__(self, owner: type, name: str) -> None:
        """
        Sets the name of the column.

        This method is part of the `descriptor protocol <https://docs.python.org/3/howto/descriptor.html>`_.
        """
        self.name = name

    def fill_default(self, array: pa.Array) -> pa.Array:
        """
        Fills null values in the array with the Column's default value.
        """
        if self.default is None or array.null_count == 0:
            return array
        if self._default_is_callable:
            null_count = array.null_count
            try:
                fill_values = pa.array([self._default_fn() for _ in range(null_count)], self.dtype)
            except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError) as e:
                raise errors.InvalidColumnDefault(self._default_fn, self.dtype, self.name) from e
            null_mask = pa.compute.is_null(array)
            return pa.compute.replace_with_mask(array, null_mask, fill_values)
        else:
            return array.fill_null(self._default_scalar)

    def pyarrow_field(self) -> pa.Field:
        """
        Returns a pyarrow Field object for this column.
        """
        return pa.field(self.name, self.dtype, self.nullable, self.metadata)

    def _load(
        self,
        data: Union[tables.DataSourceType, None],
        size_hint: Optional[int],
    ) -> Optional[pa.Array]:
        """Load a data source as a pyarrow Array of data which
        matches the column's schema. Defaults are not poulated.

        size_hint is an optional expected size of the input data. If provided:
         - For input of None, an array of nulls is returned.
         - For all other inputs, the input's length is checked.

        If the column is nullable and the input value is None, then
        this returns None.  Otherwise, it will either return a pyarrow
        Array, or raise an error.
        """
        if data is None:
            if not self.nullable and self.default is None:
                raise errors.InvalidColumnDataError(
                    self, "received no data, but field is not nullable and has no default"
                )

            # Either it's nullable, or has a default; either way, we can proceed.
            if size_hint is None:
                # We can't even make a null array of the appropriate
                # size, so we're forced to return None.
                return None

            return self._nulls(size_hint)
        if isinstance(data, pa.ChunkedArray) or isinstance(data, pa.Array):
            return data.cast(self.dtype)
        return pa.array(data, type=self.dtype)

    def _nulls(self, n: int) -> pa.Array:
        """Return an array of nulls of the appropriate size."""
        return pa.nulls(n, type=self.dtype)

    def _set_on_pyarrow_table(self, table: pa.Table, data: pa.Array) -> pa.Table:
        """Set the data for this column on a pyarrow Table."""
        idx = table.schema.get_field_index(self.name)
        value = self.fill_default(data)
        return table.set_column(idx, self.pyarrow_field(), [value])


T = TypeVar("T", bound=tables.Table)


class SubTableColumn(Column, Generic[T]):
    """
    A column which represents an embedded quivr table.

    :param table_type: The type of the table to embed.
    :param nullable: Whether the column can contain null values.
    :param metadata: A dictionary of metadata to attach to the column.
    """

    def __init__(self, table_type: type[T], nullable: bool = False, metadata: Optional[MetadataDict] = None):
        self.table_type = table_type

        self.schema = table_type.schema
        if nullable:
            # permit nulls in fields. This is necessary for the
            # case where the field is not-nullable, but the table
            # is nullable.
            fields = []
            for field in self.schema:
                if field.nullable:
                    fields.append(field)
                else:
                    fields.append(field.with_nullable(True))
            self.schema = pa.schema(fields)

        dtype = pa.struct(self.schema)
        super().__init__(dtype, nullable=nullable, metadata=metadata)

    def _set_on_pyarrow_table(self, table: pa.Table, value: T) -> pa.Table:
        """Set the data for this column on a pyarrow Table. This
        includes setting metadata which encodes the subtable's
        attributes.
        """
        # Propagate the value's metadata through to the parent
        metadata = table.schema.metadata
        value_meta = value.table.schema.metadata
        if value_meta is not None:
            for key, val in value_meta.items():
                key = (self.name + "." + key.decode("utf-8")).encode("utf-8")
                metadata[key] = val

        idx = table.schema.get_field_index(self.name)
        table = table.replace_schema_metadata(metadata)
        data = value.to_structarray()
        if self.nullable:
            # rewrite the structarray's type to make all fields
            # nullable. This is necessary for the case where the
            # field is not-nullable, but the table is nullable.
            fields = []
            for field in data.type:
                fields.append(field.with_nullable(True))
            data = pa.StructArray.from_arrays(data.flatten(), fields=fields)

        table = table.set_column(idx, self.pyarrow_field(), data)
        return table

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> T: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, T]:
        if obj is None:
            return self
        array = obj.table.column(self.name)

        metadata = self.metadata
        if metadata is None:
            metadata = {}
        metadata.update(obj._metadata_for_column(self.name))  # type: ignore

        schema = self.schema.with_metadata(metadata)

        subtable = pa.Table.from_arrays(array.flatten(), schema=schema)
        # We don't validate the subtable. If the parent table was validated,
        # then the subtable would have been validated as part of that process.
        # If the parent table is intentionally not validated, then we don't
        # want to validate the subtable at this time as it will throw an error
        # when accessing the subtable as a column (e.g. during concatenation).
        return self.table_type.from_pyarrow(subtable, permit_nulls=self.nullable, validate=False)

    def _nulls(self, n: int) -> pa.Array:
        """Return an array of nulls of the appropriate size."""
        # Quite unfortunate: pyarrow doesn't support creating a struct
        # array of nulls; it incorrectly populates all the fields with
        # nulls (including non-nullable ones!) so we need to do this
        # manually.
        #
        # See: https://github.com/apache/arrow/issues/37072
        return pa.array([None] * n, type=self.dtype)


class Int8Column(Column):
    """
    A column for storing 8-bit integers.
    """

    primitive_dtype = pa.int8()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, Callable[[], int], int] = None,
    ):
        super().__init__(
            pa.int8(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.Int8Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.Int8Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class Int16Column(Column):
    """
    A column for storing 16-bit integers.
    """

    primitive_dtype = pa.int16()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, Callable[[], int], int] = None,
    ):
        super().__init__(
            pa.int16(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.Int16Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.Int16Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class Int32Column(Column):
    """
    A column for storing 32-bit integers.
    """

    primitive_dtype = pa.int32()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, Callable[[], int], int] = None,
    ):
        super().__init__(
            pa.int32(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.Int32Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.Int32Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class Int64Column(Column):
    """
    A column for storing 64-bit integers.
    """

    primitive_dtype = pa.int64()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, Callable[[], int], int] = None,
    ):
        super().__init__(
            pa.int64(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.Int64Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.Int64Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class UInt8Column(Column):
    """
    A column for storing 8-bit unsigned integers.
    """

    primitive_dtype = pa.uint8()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, Callable[[], int], int] = None,
    ):
        super().__init__(
            pa.uint8(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.UInt8Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.UInt8Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class UInt16Column(Column):
    """
    A column for storing 16-bit unsigned integers.
    """

    primitive_dtype = pa.uint16()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, Callable[[], int], int] = None,
    ):
        super().__init__(
            pa.uint16(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.UInt16Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.UInt16Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class UInt32Column(Column):
    """
    A column for storing 32-bit unsigned integers.
    """

    primitive_dtype = pa.uint32()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, Callable[[], int], int] = None,
    ):
        super().__init__(
            pa.uint32(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.UInt32Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.UInt32Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class UInt64Column(Column):
    """
    A column for storing 64-bit unsigned integers.
    """

    primitive_dtype = pa.uint64()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, Callable[[], int], int] = None,
    ):
        super().__init__(
            pa.uint64(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.UInt64Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.UInt64Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class Float16Column(Column):
    """A column for storing 16-bit floating point numbers.

    16-bit floating point arrays have limited support in Arrow, and
    thus limited support in quivr.

    In particular:
    - They cannot be written to or read from Parquet files.

    - They cannot be constructed from Python floats in a natural way
      (one must use numpy.float16).

    - They don't support quivr column default values (because they are
      not supported by Arrow's compute functions).
    """

    primitive_dtype = pa.float16()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.float16(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.lib.HalfFloatArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.lib.HalfFloatArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class Float32Column(Column):
    """
    A column for storing 32-bit floating point numbers.
    """

    primitive_dtype = pa.float32()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, Callable[[], float], float] = None,
    ):
        super().__init__(
            pa.float32(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.lib.FloatArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.lib.FloatArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class Float64Column(Column):
    """
    A column for storing 64-bit floating point numbers.
    """

    primitive_dtype = pa.float64()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, Callable[[], float], float] = None,
    ):
        super().__init__(
            pa.float64(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.lib.DoubleArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.lib.DoubleArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class BooleanColumn(Column):
    """A column for storing booleans."""

    primitive_dtype = pa.bool_()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, Callable[[], bool], bool] = None,
    ):
        super().__init__(
            pa.bool_(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.BooleanArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[pa.BooleanArray, Self]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class StringColumn(Column):
    """A column for storing strings.

    This can be used to store strings of any length, but it is not
    recommended for storing very long strings (over 2GB, for
    example). For long strings, use LargeStringColumn instead.

    """

    primitive_dtype = pa.string()

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, str, Callable[[], str]] = None,
    ):
        super().__init__(
            pa.string(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.StringArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[pa.StringArray, Self]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class LargeBinaryColumn(Column):
    r"""
    A column for storing large binary objects (over 2\ :sup:`31` bytes long). Large binary data is stored in
    variable-length chunks.
    """

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, bytes, Callable[[], bytes]] = None,
    ):
        super().__init__(
            pa.large_binary(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.LargeBinaryArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.LargeBinaryArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class LargeStringColumn(Column):
    r"""
    A column for storing large strings (over 2\ :sup:`31` bytes long). Large string data is stored in
    variable-length chunks.
    """

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, str, Callable[[], str]] = None,
    ):
        super().__init__(
            pa.large_string(), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.LargeStringArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.LargeStringArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class Date32Column(Column):
    """A column for storing dates.

    Internally, this column stores dates as 32-bit integers which
    represent time since the UNIX epoch.

    """

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.date32(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.Date32Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.Date32Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class Date64Column(Column):
    """A column for storing dates.

    Internally, this column stores dates as 64-bit integers which
    represent time since the UNIX epoch in milliseconds, where the
    values are evenly divisible by 86,400,000.
    """

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.date64(), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.Date64Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.Date64Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


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

    :param unit: The unit of time to use for the timestamp. Valid
        values are "s", "ms", "us", and "ns".
    :param tz: An optional timezone to associate with the timestamp.
    :param nullable: Whether the column can contain null values.
    :param metadata: Optional metadata to associate with the column.
    :param validator: An optional validator to apply to the column.
    :param default: An optional default value for the column. This
        can be a scalar value or a callable that takes no arguments.
    """

    def __init__(
        self,
        unit: str,
        tz: Optional[str] = None,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, datetime.datetime, Callable[[], datetime.datetime]] = None,
    ):
        super().__init__(
            pa.timestamp(unit, tz), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.TimestampArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.TimestampArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class Time32Column(Column):
    """
    A column for storing time values.

    Time data can be stored in one of two units for this 32-bit type:
        - seconds
        - milliseconds

    Internally, a time32 value is a 32-bit integer which an elapsed time since midnight.

    :param unit: The unit of time to use for the time. Valid values are "s" and "ms".
    :param nullable: Whether the column can contain null values.
    :param metadata: Optional metadata to associate with the column.
    :param validator: An optional validator to apply to the column.
    :param default: An optional default value for the column. This
        can be a scalar value or a callable that takes no arguments.
    """

    def __init__(
        self,
        unit: str,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, datetime.time, Callable[[], datetime.time]] = None,
    ):
        super().__init__(
            pa.time32(unit), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.Time32Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.Time32Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class Time64Column(Column):
    """
    A column for storing time values with high precision.

    Time data can be stored in one of two units for this 64-bit type:
        - microseconds
        - nanoseconds

    Internally, a time64 value is a 64-bit integer which an elapsed time since midnight.

    :param unit: The unit of time to use for the time. Valid values are "us" and "ns".
    :param nullable: Whether the column can contain null values.
    :param metadata: Optional metadata to associate with the column.
    :param validator: An optional validator to apply to the column.
    :param default: An optional default value for the column. This
        can be a scalar value or a callable that takes no arguments.
    """

    def __init__(
        self,
        unit: str,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, datetime.time, Callable[[], datetime.time]] = None,
    ):
        super().__init__(
            pa.time64(unit), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.Time64Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.Time64Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class DurationColumn(Column):
    """
    A column which stores an absolute length of time.

    Internally, a duration value is always represented as an 64-bit integer.
    :param unit: The unit of time to use for the duration. Valid values are "s", "ms", "us", and "ns".
    :param nullable: Whether the column can contain null values.
    :param metadata: Optional metadata to associate with the column.
    :param validator: An optional validator to apply to the column.
    """

    def __init__(
        self,
        unit: str,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.duration(unit), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.DurationArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.DurationArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


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
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(
            pa.month_day_nano_interval(), nullable=nullable, metadata=metadata, validator=validator
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.MonthDayNanoIntervalArray: ...

    def __get__(
        self, obj: Optional[tables.Table], objtype: type
    ) -> Union[Self, pa.MonthDayNanoIntervalArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class BinaryColumn(Column):
    """A column for storing opaque binary data."""

    def __init__(
        self,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, bytes, Callable[[], bytes]] = None,
    ):
        super().__init__(
            pa.binary(-1), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.BinaryArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.BinaryArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class FixedSizeBinaryColumn(Column):
    """A column for storing opaque fixed-size binary data.

    :param byte_width: The number of bytes per value.
    :param nullable: Whether the column can contain null values.
    :param metadata: Optional metadata to associate with the column.
    :param validator: An optional validator to apply to the column.
    :param default: An optional default value for the column. This
        can be a scalar value or a callable that takes no arguments.
    """

    def __init__(
        self,
        byte_width: int,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
        default: Union[None, bytes, Callable[[], bytes]] = None,
    ):
        self.byte_width = byte_width
        super().__init__(
            pa.binary(byte_width), nullable=nullable, metadata=metadata, validator=validator, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.FixedSizeBinaryArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.FixedSizeBinaryArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class Decimal128Column(Column):
    """A column for storing arbitrary-precision decimal numbers.

    Arrow decimals are fixed-point decimal numbers encoded as a scaled
    integer. The precision is the number of significant digits that the
    decimal type can represent; the scale is the number of digits after
    the decimal point (note the scale can be negative).

    As an example, ``Decimal128Column(7, 3)`` can exactly represent the numbers
    1234.567 and -1234.567 (encoded internally as the 128-bit integers
    1234567 and -1234567, respectively), but neither 12345.67 nor
    123.4567.

    ``DecimalColumn(5, -3)`` can exactly represent the number 12345000
    (encoded internally as the 128-bit integer 12345), but neither
    123450000 nor 1234500.

    If you need a precision higher than 38 significant digits,
    consider using :class:`Decimal256Column`.

    :param precision: The number of significant digits.
    :param scale: The number of digits after the decimal point.
    :param nullable: Whether the column can contain nulls.
    :param metadata: A dictionary of metadata to attach to the column.
    :param default: An optional default value for the column. This
        can be a scalar value or a callable that takes no arguments.
    """

    def __init__(
        self,
        precision: int,
        scale: int,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        default: Union[None, decimal.Decimal, Callable[[], decimal.Decimal]] = None,
    ):
        super().__init__(
            pa.decimal128(precision, scale), nullable=nullable, metadata=metadata, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.Decimal128Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.Decimal128Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class Decimal256Column(Column):
    """A column for storing arbitrary-precision decimal numbers.

    Arrow decimals are fixed-point decimal numbers encoded as a scaled
    integer. The precision is the number of significant digits that the
    decimal type can represent; the scale is the number of digits after
    the decimal point (note the scale can be negative).

    Values are stored as 256-bit integers.

    :param precision: The number of significant digits.
    :param scale: The number of digits after the decimal point.
    :param nullable: Whether the column can contain nulls.
    :param metadata: A dictionary of metadata to attach to the column.
    :param default: An optional default value for the column. This
        can be a scalar value or a callable that takes no arguments.
    """

    def __init__(
        self,
        precision: int,
        scale: int,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        default: Union[None, decimal.Decimal, Callable[[], decimal.Decimal]] = None,
    ):
        super().__init__(
            pa.decimal256(precision, scale), nullable=nullable, metadata=metadata, default=default
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.Decimal256Array: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.Decimal256Array]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class NullColumn(Column):
    """A column for storing null values.

    Nulls are represented as a single bit, and do not take up any
    memory space.

    """

    def __init__(
        self,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(pa.null(), nullable=True, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.NullArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.NullArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


# Complex types follow


class ListColumn(Column):
    """A column for storing variably-sized lists of values.

    The values in the list can be of any type.

    Note that all quivr Tables are storing lists of values, so this
    column type is only useful for storing lists of lists.

    :param value_type: The type of the values in the list.
    :param nullable: Whether the list can contain null values.
    :param metadata: A dictionary of metadata to attach to the column.
    :param validator: A validator to run against the column's values.

    """

    def __init__(
        self,
        value_type: Union[pa.DataType, pa.Field, Column],
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        if isinstance(value_type, Column):
            value_type = value_type.dtype
        super().__init__(pa.list_(value_type, -1), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.ListArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.ListArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class FixedSizeListColumn(Column):
    """A column for storing lists of values of a fixed size.

    The values in the list can be of any type.

    Note that all quivr Tables are storing lists of values, so this
    column type is only useful for storing lists of lists.

    :param value_type: The type of the values in the list.
    :param list_size: The size of the list. Must be greater than 0.
    :param nullable: Whether the list can contain null values.
    :param metadata: A dictionary of metadata to attach to the column.
    :param validator: A validator to run against the column's values.
    """

    def __init__(
        self,
        value_type: Union[pa.DataType, pa.Field, Column],
        list_size: int,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        if list_size <= 0:
            raise ValueError("list_size must be greater than 0")
        if isinstance(value_type, Column):
            value_type = value_type.dtype
        super().__init__(
            pa.list_(value_type, list_size), nullable=nullable, metadata=metadata, validator=validator
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.FixedSizeListArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.FixedSizeListArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class LargeListColumn(Column):
    r"""A column for storing large lists of values (over 2\ :sup:`31` objects).

    Unless you need to represent data with more than 2**31 elements,
    prefer ListColumn.

    The values in the list can be of any type.

    Note that all quivr Tables are storing lists of values, so this
    column type is only useful for storing lists of lists.

    :param value_type: The type of the values in the list.
    :param nullable: Whether the list can contain null values.
    :param metadata: A dictionary of metadata to attach to the column.
    :param validator: A validator to run against the column's values.
    """

    def __init__(
        self,
        value_type: Union[pa.DataType, pa.Field, Column],
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        if isinstance(value_type, Column):
            value_type = value_type.dtype
        super().__init__(pa.large_list(value_type), nullable=nullable, metadata=metadata, validator=validator)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.LargeListArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.LargeListArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class MapColumn(Column):
    """A column for storing maps of key-value pairs.

    The keys and values can be of any type, as long as the keys are
    hashable and unique.

    :param key_type: The type of the keys in the map.
    :param item_type: The type of the values in the map.
    :param nullable: Whether the map can contain null key-value pairs.
    :param metadata: A dictionary of metadata to attach to the column.
    :param validator: A validator to run against the column's values.
    """

    def __init__(
        self,
        key_type: Union[pa.DataType, pa.Field, Column],
        item_type: Union[pa.DataType, pa.Field, Column],
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        if isinstance(key_type, Column):
            key_type = key_type.dtype
        if isinstance(item_type, Column):
            item_type = item_type.dtype
        super().__init__(
            pa.map_(key_type, item_type), nullable=nullable, metadata=metadata, validator=validator
        )

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.MapArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.MapArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class DictionaryColumn(Column):
    """A column for storing dictionary-encoded values.

    This is intended for use with categorical data. See MapColumn for a
    more general mapping type.

    :param index_type: The type of the dictionary indices. Must be an integer type.
    :param value_type: The type of the values in the dictionary.
    :param ordered: Whether the dictionary is ordered.
    :param nullable: Whether the dictionary can contain null values.
    :param metadata: A dictionary of metadata to attach to the column.
    :param validator: A validator to run against the column's values.
    """

    def __init__(
        self,
        index_type: pa.DataType,
        value_type: Union[pa.DataType, pa.Field, Column],
        ordered: bool = False,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
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
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: tables.Table, objtype: type) -> pa.DictionaryArray: ...

    def __get__(self, obj: Optional[tables.Table], objtype: type) -> Union[Self, pa.DictionaryArray]:
        if obj is None:
            return self
        return _fast_combine_chunks(obj.table[self.name])


class StructColumn(Column):
    """A column for storing structured data.

    In general, prefer to define Tables and use their as_column method
    instead of using StructColumn.

    :param fields: The fields in the struct.
    :param nullable: Whether the struct can contain null values.
    :param metadata: A dictionary of metadata to attach to the column.
    :param validator: A validator to run against the column's values.

    """

    def __init__(
        self,
        fields: list[pa.Field],
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
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


    :param run_end_type: The type of the run-end encoded values. Must be a 16-, 32-, or 64-bit integer type.
    :param value_type: The type of the values in the run-end encoded data.
    :param nullable: Whether the data can contain null values.
    :param metadata: A dictionary of metadata to attach to the column.
    :param validator: A validator to run against the column's values.
    """

    def __init__(
        self,
        run_end_type: pa.DataType,
        value_type: pa.DataType,
        nullable: bool = False,
        metadata: Optional[MetadataDict] = None,
        validator: Optional[validators.Validator] = None,
    ):
        super().__init__(
            pa.run_end_encoded(run_end_type, value_type),
            nullable=nullable,
            metadata=metadata,
            validator=validator,
        )


def _fast_combine_chunks(chunked_array: pa.ChunkedArray) -> pa.Array:
    """Combine a chunked array into a single array."""
    if chunked_array.num_chunks == 1:
        return chunked_array.chunk(0)
    return chunked_array.combine_chunks()
