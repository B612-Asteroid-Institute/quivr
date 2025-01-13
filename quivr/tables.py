from __future__ import annotations

import os
import sys
from io import IOBase

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from typing import (
    Any,
    ClassVar,
    Hashable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.parquet

from . import _arrow_utils
from . import attributes as attr_module
from . import columns, errors, schemagraph, validators


class ArrowArrayProvider(Protocol):
    """
    A Protocol which describes objects that support the Arrow custom array extension protocol.
    """

    def __arrow_array__(self, type: Optional[pa.DataType] = None) -> pa.Array: ...


AttributeValueType: TypeAlias = Union[int, float, str]
DataSourceType: TypeAlias = Union[
    pa.Array, list[Any], "Table", "pd.Series[Any]", npt.NDArray[Any], "ArrowArrayProvider"
]
AnyTable = TypeVar("AnyTable", bound="Table")


# If a table uses any of the following names, it will break quivr
# internals entirely, so they must be rejected.
_FORBIDDEN_COLUMN_NAMES = {
    "table",
    "schema",
    "_quivr_subtables",
    "_quivr_attributes",
    "_column_validators",
}


class Table:
    """Table is the primary data structure in quivr.

    Tables are used to represent tabular data with a fixed schema. The
    schema is defined by subclassing Table, providing Column objects
    as class attributes. The Table class will then generate a pyarrow
    schema from those columns.

    Table instances can then be created from data, either by passing
    in a pyarrow Table, or by passing in data in a variety of other
    formats. The data will be validated against the schema, and
    converted to a pyarrow Table.

    Table instances are immutable, but can be sliced, filtered,
    sorted, or otherwise manipulated, resulting in new Table
    instances. In particular, see the :meth:`Table.set_column` method,
    which returns a copy of the Table with a single column replaced.

    :cvar schema: The pyarrow schema for this table.
    :vartype schema: pyarrow.Schema

    :ivar table: The underlying :class:`pyarrow.Table` for this Table instance.
    :vartype table: pyarrow.Table
    """

    schema: ClassVar[pa.Schema]

    table: pa.Table

    _quivr_subtables: ClassVar[dict[str, columns.SubTableColumn[Any]]]
    _quivr_attributes: ClassVar[dict[str, attr_module.Attribute[Any]]]
    _column_validators: ClassVar[dict[str, validators.Validator]]

    def __init_subclass__(cls: Type["Table"], **kwargs: Any):
        fields = []
        column_validators = {}
        subtables = {}
        attrs = {}
        for name, obj in cls.__dict__.items():
            if name in _FORBIDDEN_COLUMN_NAMES:
                raise AttributeError(
                    f"Invalid column name {name} in {cls.__name__}: {name} is a reserved name"
                )
            if isinstance(obj, columns.Column):
                fields.append(obj.pyarrow_field())
                if obj.validator is not None:
                    column_validators[name] = obj.validator
                if isinstance(obj, columns.SubTableColumn):
                    subtables[name] = obj
            elif isinstance(obj, attr_module.Attribute):
                attrs[name] = obj

        # Generate a pyarrow schema
        schema = pa.schema(fields)
        cls.schema = schema

        # Keep track of subtables
        cls._quivr_subtables = subtables

        # Add attributes
        cls._quivr_attributes = attrs

        # Add validators
        cls._column_validators = column_validators

        super().__init_subclass__(**kwargs)

    def __init__(self, table: pa.Table, **kwargs: AttributeValueType):
        self.table = table
        for name, value in kwargs.items():
            if name in self._quivr_attributes:
                setattr(self, name, value)
            else:
                raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
        for name in self._quivr_attributes:
            # Ensure all attributes are set or have a default
            getattr(self, name)

    @classmethod
    def from_pyarrow(
        cls,
        table: pa.Table,
        validate: bool = True,
        permit_nulls: bool = False,
        **kwargs: AttributeValueType,
    ) -> Self:
        """Create a new table from a pyarrow Table.

        This is a convenience method which can be used to create a
        Table from a pyarrow Table. It can also accept keyword-style
        arguments to set attributes on the table.

        When serializing to pyarrow, the table's schema metadata
        encodes the value of attributes. This method will use that
        metadata to set the attributes on the table if it is
        available. If any attributes are provided as keyword
        arguments, they override any values in the metadata.

        :param table: The pyarrow Table to create the table from.
        :param validate: Whether to validate the table against the
            schema, and run any column validators.
        :param permit_nulls: Whether to permit null values in the
            table. If True, nulls will be permitted, even in non-nullable
            fields. This is used when a Table is used as a nullable subtable.
        :param \\**kwargs: Keyword arguments to set attributes on the
            table.
        :type \\**kwargs: :obj:`AttributeValueType`
        :return: A new Table instance.
        """
        schema = cls.schema
        if permit_nulls:
            # Rewrite the schema to permit nulls
            fields = []
            for field in schema:
                if field.nullable:
                    fields.append(field)
                else:
                    fields.append(field.with_nullable(True))
            schema = pa.schema(fields, metadata=schema.metadata)

        # Absorb metadata from the table
        schema = schema.with_metadata(table.schema.metadata)

        table = table.cast(schema)
        instance = cls(table, **kwargs)
        if validate:
            instance.validate()
        return instance

    @classmethod
    def _attribute_kwargs_from_kwargs(
        cls, kwargs: dict[str, Union[AttributeValueType, DataSourceType]]
    ) -> dict[str, AttributeValueType]:
        attrib_kwargs: dict[str, AttributeValueType] = {}
        for k, v in kwargs.items():
            if k not in cls._quivr_attributes:
                raise TypeError(f"Unexpected keyword argument: {k}")
            if not isinstance(v, cls._quivr_attributes[k]._type):
                raise TypeError(f"Expected {cls._quivr_attributes[k]._type}, got {type(v)} for {k}")
            attrib_kwargs[k] = v
        return attrib_kwargs

    @classmethod
    def as_column(
        cls, nullable: bool = True, metadata: Optional[columns.MetadataDict] = None
    ) -> columns.SubTableColumn[Self]:
        """Embed the Table as a column in another Table.

        This method is the primary way to achieve composition of Tables with quivr.

        :param nullable: Whether the column can contain nulls. Note
            that this refers to whether an entire row - all columns -
            can be null, not whether a single value in one column of
            this table can be null. That is controlled entirely by
            this Table class's columns.

        :param metadata: Metadata to attach to the column.

        """
        return columns.SubTableColumn(cls, nullable=nullable, metadata=metadata)

    @classmethod
    def from_kwargs(
        cls,
        validate: bool = True,
        permit_nulls: bool = False,
        **kwargs: Union[DataSourceType, AttributeValueType],
    ) -> Self:
        """Create a Table instance from keyword arguments.

        Each keyword argument corresponds to a column in the Table.

        The keys should correspond to column names or attribute names.

        For columns, the values should be arrays, lists, or pyarrow Arrays.

        For attributes, the values should be the appropriate type for that attribute.

        :param validate: If (the default), run column validators on all input data.
        :param \\**kwargs: The data to populate the table with.
        :type \\**kwargs: Union[:obj:`AttributeValueType`, :obj:`DataSourceType`]

        """
        arrays: list[Union[None, pa.Array]] = []
        size = None
        size_col = ""

        # We don't know the size of the table until we've found a
        # field in the schema which corresponds to a kwarg with data.
        # Therefore, we need to keep track of which columns are empty
        # *before* we've discovered the size of the table so we can
        # populate them with nulls later.
        empty_columns = []

        metadata: dict[bytes, bytes] = {}
        for i, field in enumerate(cls.schema):
            column_name = field.name
            column = getattr(cls, column_name)

            value = kwargs.pop(column_name, None)
            array = column._load(value, size)
            if array is None:
                # We'll have to wait until we get to a non-None column
                # to figure out the size.
                empty_columns.append(i)
                arrays.append(None)
                continue

            if size is None:
                size = len(array)
                size_col = column_name
            elif len(array) != size:
                raise errors.InvalidColumnDataError(
                    column,
                    f"wrong length {len(value)} (expected {size} based on column {size_col})",
                )

            arrays.append(array)

            if isinstance(value, Table):
                field_meta = value.table.schema.metadata
                if field_meta is not None:
                    for key, val in field_meta.items():
                        key = (field.name + "." + key.decode("utf-8")).encode("utf-8")
                        metadata[key] = val

        if size is None:
            raise ValueError("No data provided")

        for idx in empty_columns:
            column = getattr(cls, cls.schema[idx].name)
            arrays[idx] = column._nulls(size)

        # Inform the type checker that we've filled all Nones
        arrays = cast(list[pa.Array], arrays)

        for i, array in enumerate(arrays):
            if array.null_count > 0:
                column = getattr(cls, cls.schema[i].name)
                arrays[i] = column.fill_default(array)

        pyarrow_table = cls._build_arrow_table(arrays, metadata)
        attrib_kwargs = cls._attribute_kwargs_from_kwargs(kwargs)
        return cls.from_pyarrow(
            table=pyarrow_table,
            validate=validate,
            permit_nulls=permit_nulls,
            **attrib_kwargs,
        )

    @classmethod
    def _build_arrow_table(cls, arrays: List[pa.Array], metadata: dict[bytes, bytes]) -> pa.Table:
        """
        Construct a pyarrow Table which will back cls from a list of arrays. The
        Table's schema comes from cls.schema.
        """
        schema = cls.schema.with_metadata(metadata)
        table = pa.Table.from_arrays(arrays, schema=schema)
        return table

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, validate: bool = True, **kwargs: AttributeValueType) -> Self:
        """Load a DataFrame into the Table.

        If the DataFrame is missing any of the Table's columns, an
        error is raised. If the DataFrame has extra columns, they are
        ignored.

        This function cannot load "flattened" dataframes. This only
        matters for nested Tables which contain other Table
        definitions as columns. For that use case, either load an
        unflattened DataFrame, or use from_flat_dataframe.

        :param df: A pandas DataFrame containing the data to load.
        :param validate: Whether to validate the data after loading.
        :param \\**kwargs: Additional keyword arguments for any Table attributes.
        :type \\**kwargs: :obj:`AttributeValueType`
        """

        table = pa.Table.from_pandas(df, schema=cls.schema)
        # If attributes are present on the dataframe, use them.
        if df.attrs:
            for k, v in df.attrs.items():
                if k not in kwargs and k in cls._quivr_attributes:
                    kwargs[str(k)] = v

        for attr_name, descriptor in cls._quivr_attributes.items():
            if attr_name not in kwargs:
                # Maybe there is a column that has the value?
                if attr_name in df.columns:
                    column = df[attr_name]
                    if len(column) > 0:
                        # We need to take care with types here, since the
                        # pandas series might be a non-python-native type.
                        kwargs[attr_name] = descriptor._type(column[0])
        return cls.from_pyarrow(table=table, validate=validate, permit_nulls=False, **kwargs)

    @classmethod
    def _unflatten_table(cls, table: pa.Table) -> pa.Table:
        """Unflatten a Table.

        This is used when loading a flattened CSV into a nested
        Table. It takes a Table with a flat schema, and returns a
        Table with a nested schema.
        """
        struct_fields = []

        for field in cls.schema:
            if pa.types.is_struct(field.type):
                struct_fields.append(field)

        # Walk the schema, and build a StructArray for each embedded
        # type.

        def struct_array_for(field: pa.Field, ancestors: list[pa.Field]) -> pa.StructArray:
            prefix = ".".join([f.name for f in ancestors if f.name] + [field.name])

            child_arrays = []
            for subfield in field.type:
                if pa.types.is_struct(subfield.type):
                    child_arrays.append(struct_array_for(subfield, ancestors + [field]))
                else:
                    path = prefix + "." + subfield.name
                    child_arrays.append(table.column(path).combine_chunks())
            return pa.StructArray.from_arrays(child_arrays, fields=list(field.type))

        child_arrays = []
        for field in cls.schema:
            if pa.types.is_struct(field.type):
                child_arrays.append(struct_array_for(field, []))
            else:
                child_arrays.append(table.column(field.name).combine_chunks())

        return pa.Table.from_arrays(child_arrays, schema=cls.schema)

    @classmethod
    def from_flat_dataframe(
        cls, df: pd.DataFrame, validate: bool = True, **kwargs: AttributeValueType
    ) -> Self:
        """Load a flattened DataFrame into the Table.

        .. caution::
          Known bug: Doesn't correctly interpret fixed-length lists.

        :param df: A pandas DataFrame containing the data to load.
        :param validate: Whether to validate the data after loading.
        :param \\**kwargs: Additional keyword arguments for any Table attributes.
        :type \\**kwargs: :obj:`AttributeValueType`

        """
        struct_fields = []
        for field in cls.schema:
            if pa.types.is_struct(field.type):
                struct_fields.append(field)

        if len(struct_fields) == 0:
            return cls.from_dataframe(df, validate=validate, **kwargs)

        root = pa.field("", pa.struct(cls.schema))

        # Walk the schema, and build a StructArray for each embedded
        # type. These are stored in a dictionary, keyed by their
        # dot-separated path. For example, if the schema is:
        #
        #   pa.field("foo", pa.struct([
        #       pa.field("bar", pa.struct([
        #           pa.field("baz", pa.int64())
        #       ]))
        #   ]))
        #
        # Then the struct array for the inner struct will be stored in
        # the dictionary at the key "foo.bar".
        struct_arrays: dict[str, pa.StructArray] = {}

        # Columns might also store scalar attribute data. Gather that as we go, if it's present.
        attr_keys = cls._attribute_metadata_keys()
        attr_dict = {}

        def visitor(field: pa.Field, ancestors: list[pa.Field]) -> None:
            # Modify the dataframe in place, trimming out the columns we
            # have already processed in depth-first order.
            nonlocal df
            if len(ancestors) == 0:
                # Root - gets special behavior.
                df_key = ""
            else:
                lineage = [f.name for f in ancestors if f.name] + [field.name]
                df_key = ".".join(lineage)

            # Pull out just the columns relevant to this field
            field_columns = df.columns[df.columns.str.startswith(df_key)]
            field_df = df[field_columns]

            # Grab onto any scalar metadata stored in columns.
            for col in field_columns:
                if col in attr_keys:
                    attr_dict[col] = field_df[col].iloc[0]

            # Replace column names like "foo.bar.baz" with "baz", the
            # last component.
            if len(ancestors) == 0:
                # If we're at the root, use the original column names.
                names = field_df.columns
            else:
                names = field_df.columns.str.slice(len(df_key) + 1)
            field_df.columns = names

            # Build a StructArray of all of the subfields.
            arrays = []
            for subfield in field.type:
                sa_key = df_key + "." + subfield.name if df_key else subfield.name
                if sa_key in struct_arrays:
                    # We've already built this array, so just use it.
                    arrays.append(struct_arrays[sa_key])
                else:
                    arrays.append(field_df[subfield.name])
            sa = _arrow_utils.build_struct_array(arrays, list(field.type))
            struct_arrays[df_key] = sa

            # Clean the fields out
            df = df.drop(field_columns, axis="columns")

        schemagraph._walk_schema(root, visitor, None)
        # Now build a table back up. Grab the root-level struct array.
        sa = struct_arrays[""]

        table_arrays = []
        for subfield in cls.schema:
            # Pull out the fields of that root-level struct array.
            table_arrays.append(sa.field(subfield.name))

        # If attributes are present on the dataframe, use them.
        if df.attrs:
            attr_dict.update(cls._schema_attrs_from_pandas(df.attrs))

        table = pa.Table.from_arrays(table_arrays, schema=cls.schema)

        # Absorb attr metadata from columnar values
        attr_metadata = cls._encode_attr_dict(attr_dict)
        schema = table.schema.with_metadata(attr_metadata)
        table = table.replace_schema_metadata(schema.metadata)
        return cls.from_pyarrow(table, validate=validate, permit_nulls=False, **kwargs)

    def flattened_table(self) -> pa.Table:
        """Completely flatten the Table's underlying Arrow table,
        taking into account any nested structure, and return the data
        table itself.
        """

        table = self.table
        while any(isinstance(field.type, pa.StructType) for field in table.schema):
            table = table.flatten()
        return table

    def select(self, column_name: str, value: Any) -> Self:
        """Select from the table by exact match, returning a new
        Table which only contains rows for which the value in
        column_name equals value. Column_name can be a nested column,
        in which case, this function will recursively search for
        the column through the nested subtables using dot-delimited notation.

        :param column_name: The name of the column to select on.
            Use dot-delimited notation for nested columns.
        :param value: The value to match.

        Examples:
            >>> import quivr as qv
            >>> import pyarrow.compute as pc
            >>> class MySubTable(qv.Table):
            ...     x = qv.Int64Column()
            ...     y = qv.Int64Column()
            >>> class MyWrapperTable(qv.Table):
            ...     child = MySubTable.as_column()
            >>> c = MySubTable.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
            >>> p = MyWrapperTable.from_kwargs(child=c)
            >>> p_select = p.select("child.x", 2)
            >>> print(p_select.child.x.to_pylist())
            [2]
        """
        return self.apply_mask(pc.equal(self.column(column_name), value))

    def sort_by(self, by: Union[str, list[str], list[tuple[str, str]]]) -> Self:
        """Sorts the Table by the given column name (or multiple
        columns). This operation requires a copy, and returns a new
        Table using the copied data. Supports dot-delimited nested columns.

        by should be a column name to sort by, a list of column names, or a list (column,
        order) tuples, where order can be "ascending" or "descending". If no order is
        specified, "ascending" is used.

        :param by: The column name or list of names, or list of (column, order) tuples to sort by.

        Examples:
            >>> import quivr as qv
            >>> class MySubTable(qv.Table):
            ...     x = qv.Int64Column()
            ...     y = qv.Int64Column()
            >>> class MyWrapperTable(qv.Table):
            ...     child = MySubTable.as_column()
            >>> c = MySubTable.from_kwargs(x=[2, 1, 2], y=[4, 5, 6])
            >>> p = MyWrapperTable.from_kwargs(child=c)
            >>> p_sort = p.sort_by("child.x")
            >>> print(p_sort.child.x.to_pylist())
            [1, 2, 2]
            >>> p_sort = p.sort_by([("child.x", "descending"), ("child.y", "ascending")])
            >>> print(p_sort.child.x.to_pylist())
            [2, 2, 1]
            >>> print(p_sort.child.y.to_pylist())
            [4, 6, 5]
        """
        if isinstance(by, str):
            by = [(by, "ascending")]

        columns = []
        names = []
        by_formatted = []
        for by_i in by:
            if isinstance(by_i, tuple):
                column, order = by_i

                # Check if the sort order for this tuple is valid
                if order != "ascending" and order != "descending":
                    raise ValueError(f"Invalid sort order: {order}")

            elif isinstance(by_i, str):
                # If the sort order is not specified, default to ascending
                column = by_i
                order = "ascending"

            else:
                raise ValueError(f"Invalid sort key: {by_i}")

            by_formatted.append((column, order))
            columns.append(self.column(column))
            names.append(column)

        table = pa.table(columns, names=names)
        indices = pc.sort_indices(table, sort_keys=by_formatted)
        return self.take(indices)

    def chunk_counts(self) -> dict[str, int]:
        """Returns the number of discrete memory chunks that make up
        each of the Table's underlying arrays. The keys of the
        resulting dictionary are the column names, and the values are
        the number of chunks for that column's data.
        """
        result = {}
        for i, field in enumerate(self.schema):
            result[field.name] = self.table.column(i).num_chunks
        return result

    def fragmented(self) -> bool:
        """Returns true if the Table has any fragmented arrays. If
        this is the case, performance might be improved by calling
        defragment on it.

        """
        return any(v > 1 for v in self.chunk_counts().values())

    def to_structarray(self) -> pa.StructArray:
        """Returns self as a StructArray.

        This only works if self is not fragmented. Call table =
        defragment(table) if table.fragmented() is True.

        :raises TableFragmentedError: if the table is fragmented.
        """
        if self.fragmented():
            raise errors.TableFragmentedError(
                "Tables cannot be converted to StructArrays while fragmented; call defragment(table) first."
            )
        arrays = [chunked_array.chunks[0] for chunked_array in self.table.columns]
        return pa.StructArray.from_arrays(arrays, fields=list(self.schema))

    def to_dataframe(
        self, flatten: bool = True, attr_handling: Literal["drop", "add_columns", "attrs"] = "attrs"
    ) -> pd.DataFrame:
        """Returns self as a pandas DataFrame.

        If flatten is true, then any nested hierarchy is flattened: if
        the Table's schema contains a struct named "foo" with field
        "a", "b", and "c", then the resulting DataFrame will include
        columns "foo.a", "foo.b", "foo.c". This is done fully for any
        deeply nested structure, for example "foo.bar.baz.c".

        If flatten is false, then that struct will be in a single
        "foo" column, and the values will of the column will be
        dictionaries representing the struct values.

        Scalar Table attributes are handled according to
        attr_handling.

        :param flatten: Whether to flatten the table's structure.
        :param attr_handling: How to handle attributes. If "drop",
        then attributes are dropped. If "add_columns", then attributes
        are added as regular columns, with the attribute value
        repeated for every row. If "attrs", then atributes are added
        to the DataFrame.attrs attribute as a dictionary.

        """
        table = self.table
        if flatten:
            table = self.flattened_table()
        df: pd.DataFrame = table.to_pandas(date_as_object=False, timestamp_as_object=False)
        if attr_handling == "drop":
            return df
        elif attr_handling == "add_columns":
            for name, val in self._flattened_attributes().items():
                df[name] = pa.repeat(val, len(self))
            return df
        elif attr_handling == "attrs":
            df.attrs = self._attrs_to_pandas()  # type: ignore
            return df
        else:
            raise ValueError(f"Unknown attr_handling value {attr_handling}")

    def column(self, column_name: str) -> pa.ChunkedArray:
        """
        Returns the column with the given name as a raw pyarrow ChunkedArray.
        Column_name can be a nested column, in which case, this function will recursively
        search for the column through the nested subtables using dot-delimited notation.

        :param column_name: The name of the column to return.
            Use dot-delimited notation for nested columns.

        Examples:
            >>> import quivr as qv
            >>> import pyarrow.compute as pc
            >>> class MySubTable(qv.Table):
            ...     x = qv.Int64Column()
            ...     y = qv.Int64Column()
            >>> class MyWrapperTable(qv.Table):
            ...     child = MySubTable.as_column()
            >>> c = MySubTable.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
            >>> p = MyWrapperTable.from_kwargs(child=c)
            >>> column_x = p.column("child.x")
            >>> print(column_x.to_pylist())
            [1, 2, 3]
        """
        if "." in column_name:
            column_name, subkey = column_name.split(".", 1)
            subtable = getattr(self, column_name)
            return subtable.column(subkey)

        return self.table.column(column_name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={len(self.table)})"

    def __len__(self) -> int:
        """
        Returns the number of rows in the Table.
        """
        return len(self.table)

    def with_table(self, table: pa.Table) -> Self:
        return self.__class__(table)

    def __getitem__(self, idx: Union[int, slice]) -> Self:
        """
        Returns a new Table containing the given row or rows.

        :param idx: The row index or slice to return.
        """
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            table = self.table[idx : idx + 1]
        else:
            table = self.table[idx]
        return self.with_table(table)

    def __iter__(self) -> Iterator[Self]:
        """
        Iterates over the rows of the Table, returning a new Table
        containing each row.
        """
        for i in range(len(self)):
            yield self[i : i + 1]

    def __eq__(self, other: Any) -> bool:
        """Returns true if the two Tables are equal. They are
        considered equal if they have the same data in their tables,
        and identical attributes and attribute values.

        :param other: The other Table to compare to.

        """
        if isinstance(other, Table):
            if not bool(self.table.equals(other.table, check_metadata=False)):
                return False
            if not self._attr_equal(other):
                return False
            return True
        if isinstance(other, pa.Table):
            return bool(self.table.equals(other, check_metadata=True))

        return False

    def _attr_equal(self, other: Self) -> bool:
        """
        Check if the attributes of two tables are equal. Recurs into subtables.
        """
        if not isinstance(other, self.__class__):
            return False

        if self.attributes() != other.attributes():
            return False

        for subtable_name in self._quivr_subtables.keys():
            self_subtable = getattr(self, subtable_name)
            other_subtable = getattr(other, subtable_name)
            if other_subtable is None:
                return False
            if not self_subtable._attr_equal(other_subtable):
                return False
        return True

    def take(self, row_indices: Union[list[int], pa.IntegerArray]) -> Self:
        """
        Return a new Table with only the rows at the given indices.

        :param row_indices: The indices of the rows to return.
        """
        return self.__class__(self.table.take(row_indices))

    def to_parquet(self, path: str, **kwargs: Any) -> None:
        """
        Write the table to a Parquet file.

        :param path: The path to write the Parquet file to.
        :param kwargs: Additional arguments to pass to pyarrow.parquet.write_table.
        """
        pyarrow.parquet.write_table(self.table, path, **kwargs)

    @classmethod
    def from_parquet(
        cls,
        path: str,
        memory_map: bool = False,
        pq_buffer_size: int = 0,
        filters: Optional[pc.Expression] = None,
        column_name_map: Optional[dict[str, str]] = None,
        validate: bool = True,
        **kwargs: AttributeValueType,
    ) -> Self:
        """Read a table from a Parquet file.

        :param path: The path to the Parquet file.
        :param memory_map: If True, memory-map the file, otherwise read it into memory.

        :param pq_buffer_size: If positive, perform read buffering
                when deserializing individual column
                chunks. Otherwise, IO calls are unbuffered.
        :param filters: An optional filter predicate to apply to the
                data. Rows which do not match the predicate will be
                removed from scanned data. For more information, see
                the PyArrow documentation on
                pyarrow.parquet.read_table and its filter parameter.
        :param column_name_map: An optional dictionary mapping column names in the Parquet file to
                column names in the resulting Table. This is useful if the Parquet file contains
                column names that are not valid Python identifiers, or if you want to rename
                columns for any other reason.
        :param validate: Whether to run column validation on the resulting Table.
        :param \\**kwargs: Additional keyword arguments to pass to Self's __init__ method.

        """
        table = cls._load_parquet_table(
            path=path,
            memory_map=memory_map,
            pq_buffer_size=pq_buffer_size,
            filters=filters,
            column_name_map=column_name_map,
        )
        return cls.from_pyarrow(table=table, validate=validate, permit_nulls=False, **kwargs)

    @classmethod
    def _load_parquet_table(
        cls,
        path: str,
        memory_map: bool,
        pq_buffer_size: int,
        filters: Optional[pc.Expression],
        column_name_map: Optional[dict[str, str]],
    ) -> pa.Table:
        if column_name_map is not None:
            for value in column_name_map.values():
                if value not in cls.schema.names:
                    raise ValueError(
                        f"Column name {value} does not exist for {cls.__name__}, so cannot rename to it"
                    )
            inverted_map = {v: k for k, v in column_name_map.items()}
            column_names = [inverted_map.get(field.name, field.name) for field in cls.schema]
            schema = pa.schema(
                [pa.field(inverted_map.get(field.name, field.name), field.type) for field in cls.schema]
            )
        else:
            column_names = [field.name for field in cls.schema]
            schema = cls.schema

        table = pyarrow.parquet.read_table(
            source=path,
            columns=column_names,
            memory_map=memory_map,
            buffer_size=pq_buffer_size,
            filters=filters,
            schema=schema,
        )
        md = pyarrow.parquet.read_metadata(path, memory_map=memory_map)
        table = table.replace_schema_metadata(md.metadata)

        if column_name_map is not None:
            table = table.rename_columns([field.name for field in cls.schema])

        return table

    def to_feather(self, path: str, **kwargs: Any) -> None:
        """
        Write the table to a Feather file.

        :param path: The path to write the Feather file to.
        :param kwargs: Additional arguments to pass to pyarrow.feather.write_feather.
        """
        pyarrow.feather.write_feather(self.table, path, **kwargs)

    @classmethod
    def from_feather(cls, path: str, validate: bool = True, **kwargs: AttributeValueType) -> Self:
        """Read a table from a Feather file.

        :param path: The path to the Feather file.
        :param validate: Whether to run column validators on the table after loading it.
        :param \\**kwargs: Additional keyword arguments to pass to Self's __init__ method.
        """
        table = pyarrow.feather.read_table(path)
        return cls.from_pyarrow(table=table, validate=validate, permit_nulls=False, **kwargs)

    def to_csv(self, path: str, attribute_columns: bool = True) -> None:
        """Write the table to a CSV file. Any nested structure is flattened.

        :param path: The path to write the CSV file to.
        :param attribute_columns: If True, store any Attributes defined for the table
            (or its subtable columns) as columns in the CSV file. If False, do not store
            any Attribute data in the CSV file.
        """
        table = self.flattened_table()
        if attribute_columns:
            for name, val in self._string_attributes().items():
                table = table.append_column(
                    name,
                    pa.repeat(val, len(table)),
                )
        pyarrow.csv.write_csv(table, path)

    @classmethod
    def from_csv(
        cls,
        input_file: Union[str, os.PathLike, IOBase],  # type: ignore
        validate: bool = True,
        **kwargs: AttributeValueType,
    ) -> Self:
        """
        Read a table from a CSV file.

        :param input_file: The path to the CSV file, or a file-like object.
        :param \\**kwargs: Additional keyword arguments to set the Table's attributes.
        """
        flat_table = pyarrow.csv.read_csv(input_file)

        # Gather any attributes from the CSV. We do this by looking for specially named
        # columns, and grabbing the value from their first row.
        attributes = {}
        attr_names = cls._attribute_metadata_keys()
        if len(flat_table) > 0:
            to_be_removed = []
            for name in flat_table.column_names:
                if name in attr_names:
                    attributes[name] = str(flat_table.column(name)[0].as_py())
                    to_be_removed.append(name)

            if len(to_be_removed) > 0:
                flat_table = flat_table.drop_columns(to_be_removed)

        attribute_meta = cls._unpack_string_metadata(attributes)

        table = cls._unflatten_table(flat_table)
        metadata = table.schema.metadata or {}
        metadata.update(attribute_meta)
        table = table.replace_schema_metadata(metadata)
        return cls.from_pyarrow(table=table, validate=validate, permit_nulls=False, **kwargs)

    def is_valid(self) -> bool:
        """Validate the table against the schema."""
        for name, validator in self._column_validators.items():
            if not validator.valid(self.table.column(name)):
                return False
        return True

    def validate(self) -> None:
        """Validate the table against the schema, raising an exception if invalid."""
        for name, validator in self._column_validators.items():
            try:
                validator.validate(self.table.column(name))
            except errors.ValidationError as e:
                raise errors.ValidationError(f"Column {name} failed validation: {str(e)}", e.failures) from e
        # Validate subtables
        for name in self._quivr_subtables.keys():
            getattr(self, name).validate()

    def invalid_mask(self) -> pa.Array:
        """Return a boolean mask indicating which rows are invalid."""
        num_rows = self.table.num_rows
        mask = np.zeros(num_rows, dtype=bool)
        for name, validator in self._column_validators.items():
            indices, _ = validator.failures(self.table.column(name))
            mask[indices.to_numpy()] = True

        # Get invalid rows from subtables
        for name in self._quivr_subtables.keys():
            subtable = getattr(self, name)
            subtable_mask = subtable.invalid_mask().to_numpy(zero_copy_only=False)
            mask = mask | subtable_mask
        return pa.array(mask, type=pa.bool_())

    def separate_invalid(self) -> Tuple[Self, Self]:
        """
        Separates rows that have invalid data from those that have valid data.

        Returns:
            Tuple[Self, Self]: A tuple of two Tables. The first Table contains the rows that
            passed validation, and the second Table contains the rows that failed validation.
        """
        # Separate the rows that do not validate
        failure_indices = self.invalid_mask()
        valid = self.apply_mask(pyarrow.compute.invert(failure_indices))
        invalid = self.apply_mask(failure_indices)
        return valid, invalid

    def null_mask(self) -> pa.Array:
        """Return a boolean mask indicating which rows of the entire table are null."""
        # Get the null mask for each column
        flattened_table = self.flattened_table()
        mask = pa.repeat(True, len(flattened_table))
        for name in flattened_table.column_names:
            mask = pc.and_(mask, pc.is_null(flattened_table.column(name)))
        return pa.array(mask, type=pa.bool_())

    @classmethod
    def empty(cls, **kwargs: AttributeValueType) -> Self:
        """Create an empty instance of the table.

        :param \\**kwargs: Additional keyword arguments to set the Table's attributes.
        """
        data = [[] for _ in range(len(cls.schema))]  # type: ignore
        empty_table = pa.table(data, schema=cls.schema)
        return cls.from_pyarrow(table=empty_table, validate=False, permit_nulls=False, **kwargs)

    @classmethod
    def nulls(cls, size: int, **kwargs: AttributeValueType) -> Self:
        """Create a table with nulls.

        :param size: The number of rows to create.
        :param \\**kwargs: Additional keyword arguments to set the Table's attributes.


        Even for tables which do not permit nulls in their columns, it is possible
        to create a table with nulls in the context of SubTableColumn. So for
        both of these cases, we need a method to populate tables with null values
        for all columns, while also setting the Table's attributes.
        """
        null_array = pa.repeat(None, size)
        data = [null_array for _ in range(len(cls.schema))]  # type: ignore
        null_table = pa.table(data, schema=cls.schema)
        return cls.from_pyarrow(table=null_table, validate=False, permit_nulls=True, **kwargs)

    def attributes(self) -> dict[str, Any]:
        """Return a dictionary of the table's attributes."""
        return {name: getattr(self, name) for name in self._quivr_attributes}

    def _attrs_to_pandas(self) -> dict[str, Any]:
        """Return a dictionary of the table's attributes, suitable
        for passing to pandas as DataFrame attrs.
        """
        d = {}
        for name, descriptor in self._quivr_attributes.items():
            value = getattr(self, name)
            d[name] = value

        for name in self._quivr_subtables.keys():
            d[name] = getattr(self, name)._attrs_to_pandas()

        return d

    @classmethod
    def _schema_attrs_from_pandas(cls, pd_attrs: dict[Hashable, Any]) -> dict[str, Any]:
        """Load attributes from a pandas DataFrame attrs dict into the
        pyarrow schema metadata format.

        """
        d = {}
        for name, descriptor in cls._quivr_attributes.items():
            value = pd_attrs.get(name)
            if value is not None:
                d[name] = value

        for name in cls._quivr_subtables.keys():
            subattrs = pd_attrs.get(name)
            if subattrs is not None:
                subtable_type = cls._quivr_subtables[name].table_type
                subattr_values = subtable_type._schema_attrs_from_pandas(subattrs)
                for k, v in subattr_values.items():
                    d[f"{name}.{k}"] = v

        return d

    def _flattened_attributes(self) -> dict[str, Any]:
        """
        Return a dictionary of the table's attributes, including those of subtables.

        Subtable attributes are flattened into the dictionary, with the subtable name
        prepended to the attribute name, separated by a period. For example, if the
        table has a subtable called "subtable", and that subtable has an attribute
        called "attribute", then the returned dictionary will have an entry called
        "subtable.attribute".
        """
        d = {}
        for name, descriptor in self._quivr_attributes.items():
            value = getattr(self, name)
            d[name] = value
        for name in self._quivr_subtables.keys():
            subattribs = getattr(self, name)._flattened_attributes()
            for subname, subval in subattribs.items():
                d[f"{name}.{subname}"] = subval
        return d

    def _string_attributes(self) -> dict[str, str]:
        """Return a dictionary of the table's attributes.

        Attributes are presented in their string form.
        """
        d = {}
        for name, descriptor in self._quivr_attributes.items():
            value = getattr(self, name)
            d[name] = descriptor.to_string(value)
        for name in self._quivr_subtables.keys():
            subattribs = getattr(self, name)._string_attributes()
            for subname, subval in subattribs.items():
                d[f"{name}.{subname}"] = subval
        return d

    @classmethod
    def _unpack_string_metadata(cls, metadata: dict[str, str]) -> dict[bytes, bytes]:
        result = {}

        for k, v in metadata.items():
            if "." in k:
                # This is metadata for a subtable; we need to dig into it to find the
                # descriptor for the attribute.
                subtable_name, subtable_key = k.split(".", 1)
                subtable_item = cls._quivr_subtables[subtable_name].table_type._unpack_string_metadata(
                    {subtable_key: v}
                )
                result[k.encode("utf8")] = next(iter(subtable_item.values()))
            else:
                descriptor = cls._quivr_attributes.get(k)
                if descriptor is None:
                    raise AttributeError(f"unexpected missing descriptor with name={k}")
                result[k.encode("utf8")] = descriptor.to_bytes(descriptor.from_string(v))
        return result

    def _metadata_for_column(self, column_name: str) -> dict[bytes, bytes]:
        """Return a dictionary of metadata associated with a subtable column."""
        result: dict[bytes, bytes] = {}
        if self.table.schema.metadata is None:
            return result
        column_name_bytes = (column_name + ".").encode("utf8")
        for key, value in self.table.schema.metadata.items():
            if key.startswith(column_name_bytes):
                result[key[len(column_name_bytes) :]] = value
        return result

    @classmethod
    def _attribute_metadata_keys(cls) -> set[str]:
        """Return a set of all subtable attribute names."""
        result = {attr.name for attr in cls._quivr_attributes.values()}
        for column in cls._quivr_subtables.values():
            attr_fields = column.table_type._quivr_attributes
            for attr in attr_fields.values():
                result.add(f"{column.name}.{attr.name}")

            children = column.table_type._attribute_metadata_keys()
            for key in children:
                result.add(f"{column.name}.{key}")
        return result

    @classmethod
    def _attribute_descriptor(cls, metadata_key: str) -> attr_module.Attribute[Any]:
        """Return an attribute descriptor associated with a
        particular (possibly dot-delimited, nested) metadata key.

        """
        if "." not in metadata_key:
            return cls._quivr_attributes[metadata_key]
        subtable_name, subtable_key = metadata_key.split(".", 1)
        subtable: type[Table] = cls._quivr_subtables[subtable_name].table_type
        return subtable._attribute_descriptor(subtable_key)

    @classmethod
    def _encode_attr_dict(cls, attrs: dict[str, Any]) -> dict[bytes, bytes]:
        """Encode a dictionary of attributes into a dictionary of
        bytes. This is the only format permitted by pyarrow. Keys
        should be dot-delimited to indicate subtable attributes.

        """
        result: dict[bytes, bytes] = {}
        for k, v in attrs.items():
            descriptor = cls._attribute_descriptor(k)
            pytyped = descriptor._type(v)
            result[k.encode("utf8")] = descriptor.to_bytes(pytyped)
        return result

    def apply_mask(self, mask: pa.BooleanArray | npt.NDArray[np.bool_] | list[bool]) -> Self:
        """
        Return a new table with rows filtered to match a boolean mask.

        The mask must have the same length as the table. At each index, if the mask's
        value is True, the row will be included in the new table; if False, it will be
        excluded.

        If the mask is a pyarrow BooleanArray, it must not have any null values.

        :param mask: A boolean mask to apply to the table.
        """
        if len(mask) != len(self):
            raise ValueError("mask must be the same length as the table")
        if isinstance(mask, pa.BooleanArray):
            if mask.null_count > 0:
                raise ValueError("mask must not contain null values")
        return self.__class__(self.table.filter(mask))

    def where(self, expr: pc.Expression) -> Self:
        """
        Return a new table with rows filtered to match an expression.

        The expression must be a pyarrow Expression that evaluates to a boolean array.

        :param expr: A pyarrow Expression to apply to the table.

        Examples:
            >>> import quivr as qv
            >>> import pyarrow.compute as pc
            >>> class MyTable(qv.Table):
            ...     x = qv.Int64Column()
            ...     y = qv.Int64Column()
            >>> t = MyTable.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
            >>> filtered = t.where(pc.field("x") > 1)
            >>> print(filtered.x.to_pylist())
            [2, 3]
            >>> print(filtered.y.to_pylist())
            [5, 6]
        """
        return self.__class__(self.table.filter(expr))

    def __arrow_array__(self, type: Optional[pa.DataType] = None) -> pa.StructArray:
        """
        Implements the Arrow array protocol by returning as StructArray.
        """
        array = self.to_structarray()
        if type is None:
            return array
        return array.cast(type)

    def _column_obj(self, name: str) -> columns.Column:
        """Return the Column object for a column."""
        return getattr(self.__class__, name)  # type: ignore

    def set_column(self, name: str, data: DataSourceType) -> Self:
        """
        Return a copy of the table with a particular column replaced with new data.

        :param name: The name of the column to replace.
        :param data: The new column data.
        """
        if "." in name:
            name, subkey = name.split(".", 1)
            # name should reference a subtable.
            subtable = getattr(self, name)
            subtable_new = subtable.set_column(subkey, data)
            return self.set_column(name, subtable_new)

        column = self._column_obj(name)

        if data is None:
            data = column._nulls(len(self))

        table = column._set_on_pyarrow_table(self.table, data)
        return self.from_pyarrow(table=table, validate=True, permit_nulls=False)

    def unique_indices(
        self, subset: Optional[List[str]] = None, keep: Literal["first", "last"] = "first"
    ) -> pa.Array:
        """
        Get the indices of the first or last occurrence of each unique row in the table. A subset of
        columns can be specified to consider when determining uniqueness. If no subset is specified then
        all columns are used.

        :param subset: Subset of columns to consider when determining uniqueness.
        :param keep: If there are duplicate rows then keep the first or last row.
        """
        # Flatten the table so nested columns are dot-delimited at the top level
        flattened_table = self.flattened_table()

        # If subset is not specified then use all the columns
        if subset is None:
            subset = [c for c in flattened_table.column_names]

        # Add an index column to the flattened table
        flattened_table = flattened_table.add_column(0, "index", pa.array(np.arange(len(flattened_table))))

        if keep not in ["first", "last"]:
            raise ValueError(f"keep must be 'first' or 'last', got {keep}.")

        agg_func = keep
        indices = (
            flattened_table.group_by(subset, use_threads=False)
            .aggregate([("index", agg_func)])
            .column(f"index_{agg_func}")
        ).combine_chunks()
        return indices

    def drop_duplicates(
        self,
        subset: Optional[List[str]] = None,
        keep: Literal["first", "last"] = "first",
    ) -> Self:
        """
        Drop duplicate rows from a `~quivr.Table`. This function is similar to
        `~pandas.DataFrame.drop_duplicates` but it supports nested columns (representing
        nested tables).

        :param subset: Subset of columns to consider when dropping duplicates. If not specified then
            all columns are used.
        :param keep: If there are duplicate rows then keep the first or last row.

        """
        indices = self.unique_indices(subset=subset, keep=keep)
        return self.take(indices)
