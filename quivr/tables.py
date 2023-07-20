from __future__ import annotations

import os
import sys
from io import IOBase

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from typing import Any, ClassVar, Iterator, Optional, Type, TypeAlias, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.parquet

from . import columns
from .attributes import Attribute
from .errors import TableFragmentedError, ValidationError
from .schemagraph import _walk_schema
from .validators import Validator

AttributeValueType: TypeAlias = Union[int, float, str]
DataSourceType: TypeAlias = Union[pa.Array, list[Any], "Table", pd.Series, npt.NDArray[Any]]
AnyTable = TypeVar("AnyTable", bound="Table")


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
    instances.

    :cvar schema: The pyarrow schema for this table.
    :vartype schema: pyarrow.Schema

    :ivar table: The underlying :class:`pyarrow.Table` for this Table instance.
    :vartype table: pyarrow.Table
    """

    schema: ClassVar[pa.Schema]
    table: pa.Table

    _quivr_subtables: ClassVar[dict[str, columns.SubTableColumn[Any]]]
    _quivr_attributes: ClassVar[dict[str, Attribute[Any]]]
    _column_validators: ClassVar[dict[str, Validator]]

    def __init_subclass__(cls: Type["Table"], **kwargs: Any):
        fields = []
        column_validators = {}
        subtables = {}
        attributes = {}
        for name, obj in cls.__dict__.items():
            if isinstance(obj, columns.Column):
                fields.append(obj.pyarrow_field())
                if obj.validator is not None:
                    column_validators[name] = obj.validator
                if isinstance(obj, columns.SubTableColumn):
                    subtables[name] = obj
            elif isinstance(obj, Attribute):
                attributes[name] = obj

        # Generate a pyarrow schema
        schema = pa.schema(fields)
        cls.schema = schema

        # Keep track of subtables
        cls._quivr_subtables = subtables

        # Add attributes
        cls._quivr_attributes = attributes

        # Add validators
        cls._column_validators = column_validators

        # If the subclass has an __init__ override, it must have a
        # with_table override as well.
        if "__init__" in cls.__dict__ and "with_table" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} has an __init__ override, but does not have a with_table "
                + "override. You must implement both or neither."
            )

        super().__init_subclass__(**kwargs)

    def __init__(self, table: pa.Table, **kwargs: AttributeValueType):
        self.table = table
        for name, value in kwargs.items():
            if name in self._quivr_attributes:
                setattr(self, name, value)
            else:
                raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    @classmethod
    def from_data(
        cls,
        data: Union[pa.Table, dict[str, Any], list[Any], pd.DataFrame, None] = None,
        validate: bool = True,
        **kwargs: Union[AttributeValueType, DataSourceType],
    ) -> Self:
        """Create an instance of the Table and populate it with data.

        This is a convenience method which tries to infer the right
        underlying constructors to use based on the type of data. It
        can also accept keyword-style arguments to pass data in. If
        you know the data's structure well in advance, the more
        precise constructors (from_arrays, from_pylist, etc) should be
        preferred.

        If the validate argument is True, the data will be validated
        against the table's schema. If validation fails, a
        :class:`ValidationError` will be raised.

        For example:

            >>> import quivr
            >>> class MyTable(quivr.Table):
            ...     a = quivr.StringColumn()
            ...     b = quivr.Int64Column()
            ...
            >>> # All of these are equivalent:
            >>> MyTable.from_data({"a": ["a", "b"], "b": [1, 2]})
            MyTable(size=2)
            >>> MyTable.from_data([{"a": "a", "b": 1}, {"a": "b", "b": 2}])
            MyTable(size=2)
            >>> MyTable.from_data(a=["a", "b"], b=[1, 2])
            MyTable(size=2)
            >>> import numpy as np
            >>> MyTable.from_data(a=np.array(["a", "b"]), b=np.array([1, 2]))
            MyTable(size=2)

        :param data: The data to populate the table with.
        :param validate: Whether to validate the data against the table's schema.
        :param \\**kwargs: More data in keyword argument form. The keys
          can be column names or attribute names. The values should be
          arrays, lists, or pyarrow Arrays.
        :type \\**kwargs: Union[:obj:`AttributeValueType`, :obj:`DataSourceType`]

        """
        if data is None:
            instance = cls.from_kwargs(**kwargs)
        else:
            attrib_kwargs = cls._attribute_kwargs_from_kwargs(kwargs)
            if isinstance(data, pa.Table):
                instance = cls(table=data, **attrib_kwargs)
            elif isinstance(data, dict):
                instance = cls.from_pydict(data, **attrib_kwargs)
            elif isinstance(data, list):
                if len(data) == 0:
                    instance = cls.from_rows(data, **attrib_kwargs)
                elif isinstance(data[0], dict):
                    instance = cls.from_rows(data, **attrib_kwargs)
                elif isinstance(data[0], list):
                    instance = cls.from_lists(data, **attrib_kwargs)
                else:
                    raise TypeError(f"Unsupported type: {type(data[0])}")
            elif isinstance(data, pd.DataFrame):
                instance = cls.from_dataframe(data, **attrib_kwargs)
            else:
                raise TypeError(f"Unsupported type: {type(data)}")

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
    def from_kwargs(cls, **kwargs: Union[DataSourceType, AttributeValueType]) -> Self:
        """Create a Table instance from keyword arguments.

        Each keyword argument corresponds to a column in the Table.

        The keys should correspond to column names or attribute names.

        For columns, the values should be arrays, lists, or pyarrow Arrays.

        For attributes, the values should be the appropriate type for that attribute.

        :param \\**kwargs: The data to populate the table with.
        :type \\**kwargs: Union[:obj:`AttributeValueType`, :obj:`DataSourceType`]

        """
        arrays: list[Union[None, pa.Array]] = []
        size = None

        # We don't know the size of the table until we've found a
        # field in the schema which corresponds to a kwarg with data.
        # Therefore, we need to keep track of which columns are empty
        # *before* we've discovered the size of the table so we can
        # populate them with nulls later.
        empty_columns = []

        metadata: dict[bytes, bytes] = {}
        for i, field in enumerate(cls.schema):
            column_name = field.name
            value = kwargs.pop(column_name, None)

            if value is None:
                if not field.nullable:
                    raise ValueError(f"Missing non-nullable column {column_name}")
                else:
                    if size is not None:
                        arrays.append(pa.nulls(size, type=cls.schema[i].type))
                    else:
                        # We'll have to wait until we get to a non-None column
                        # to figure out the size.
                        empty_columns.append(i)
                        arrays.append(None)
                    continue
            if size is None:
                size = len(value)  # type: ignore
            elif len(value) != size:
                raise ValueError(
                    f"Column {column_name} has wrong length {len(value)} (first column has length {size})"
                )

            if isinstance(value, Table):
                field_meta = value.table.schema.metadata
                if field_meta is not None:
                    for key, val in field_meta.items():
                        key = (field.name + "." + key.decode("utf-8")).encode("utf-8")
                        metadata[key] = val
                arrays.append(value.to_structarray())
            elif isinstance(value, pa.Array):
                arrays.append(value)
            elif isinstance(value, np.ndarray):
                arrays.append(pa.array(value, type=field.type))
            elif isinstance(value, list):
                arrays.append(pa.array(value, type=field.type))
            elif isinstance(value, pd.Series):
                arrays.append(pa.array(value, type=field.type))
            else:
                raise TypeError(f"Unsupported type for {column_name}: {type(value)}")

        if size is None:
            raise ValueError("No data provided")

        for idx in empty_columns:
            arrays[idx] = pa.nulls(size, type=cls.schema[idx].type)
        attrib_kwargs = cls._attribute_kwargs_from_kwargs(kwargs)
        return cls.from_arrays(arrays, metadata=metadata, **attrib_kwargs)

    @classmethod
    def from_arrays(
        cls,
        arrays: list[pa.Array],
        metadata: Optional[dict[bytes, bytes]] = None,
        **kwargs: AttributeValueType,
    ) -> Self:
        """
        Create a Table object from a list of arrays.

        :param arrays: A list of pyarrow.Array objects.
        :param metadata: An optional dictionary of metadata to attach to the Table.
        :param \\**kwargs: Additional keyword arguments for any Table attributes.
        :type \\**kwargs: :obj:`AttributeValueType`
        :return: A Table object.
        """
        if metadata is None:
            metadata = {}
        schema = cls.schema.with_metadata(metadata)
        table = pa.Table.from_arrays(arrays, schema=schema)
        return cls(table=table, **kwargs)

    @classmethod
    def from_pydict(
        cls, d: dict[str, Union[pa.array, list[Any], npt.NDArray[Any]]], **kwargs: AttributeValueType
    ) -> Self:
        table = pa.Table.from_pydict(d, schema=cls.schema)
        return cls(table=table, **kwargs)

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]], **kwargs: AttributeValueType) -> Self:
        """
        Create a Table object from a list of dictionaries.

        :param rows: A list of values. Each value corresponds to a row in the table.
        :param \\**kwargs: Additional keyword arguments for any Table attributes.
        :type \\**kwargs: :obj:`AttributeValueType`
        :returns: A Table object.

        Examples:
            >>> import quivr
            >>> class Inner(quivr.Table):
            ...     a = quivr.StringColumn()
            ...
            >>> class Outer(quivr.Table):
            ...     z = quivr.StringColumn()
            ...     i = Inner.as_column()
            ...
            >>> data = [{"z": "v1", "i": {"a": "v1_in"}}, {"z": "v2", "i": {"a": "v2_in"}}]
            >>> Outer.from_rows(data)
            Outer(size=2)
        """
        table = pa.Table.from_pylist(rows, schema=cls.schema)
        return cls(table=table, **kwargs)

    @classmethod
    def from_lists(cls, lists: list[list[Any]], **kwargs: AttributeValueType) -> Self:
        """Create a Table object from a list of lists.

        Each inner list corresponds to a column in the Table. They
        should be specified in the same order as the columns in the
        class.

        :param lists: A list of lists. Each inner list corresponds to a column in the table.
        :param \\**kwargs: Additional keyword arguments for any Table attributes.
        :type \\**kwargs: :obj:`AttributeValueType`
        :returns: A Table object.

        """
        table = pa.Table.from_arrays(list(map(pa.array, lists)), schema=cls.schema)
        return cls(table=table, **kwargs)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, **kwargs: AttributeValueType) -> Self:
        """Load a DataFrame into the Table.

        If the DataFrame is missing any of the Table's columns, an
        error is raised. If the DataFrame has extra columns, they are
        ignored.

        This function cannot load "flattened" dataframes. This only
        matters for nested Tables which contain other Table
        definitions as columns. For that use case, either load an
        unflattened DataFrame, or use from_flat_dataframe.

        :param df: A pandas DataFrame containing the data to load.
        :param \\**kwargs: Additional keyword arguments for any Table attributes.
        :type \\**kwargs: :obj:`AttributeValueType`
        """

        table = pa.Table.from_pandas(df, schema=cls.schema)
        return cls(table=table, **kwargs)

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
    def from_flat_dataframe(cls, df: pd.DataFrame, **kwargs: AttributeValueType) -> Self:
        """Load a flattened DataFrame into the Table.

        .. caution::
          Known bug: Doesn't correctly interpret fixed-length lists.

        :param df: A pandas DataFrame containing the data to load.
        :param \\**kwargs: Additional keyword arguments for any Table attributes.
        :type \\**kwargs: :obj:`AttributeValueType`

        """
        struct_fields = []
        for field in cls.schema:
            if pa.types.is_struct(field.type):
                struct_fields.append(field)

        if len(struct_fields) == 0:
            return cls(table=pa.Table.from_pandas(df, schema=cls.schema), **kwargs)

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
            sa = pa.StructArray.from_arrays(arrays, fields=list(field.type))
            struct_arrays[df_key] = sa

            # Clean the fields out
            df = df.drop(field_columns, axis="columns")

        _walk_schema(root, visitor, None)
        # Now build a table back up. Grab the root-level struct array.
        sa = struct_arrays[""]

        table_arrays = []
        for subfield in cls.schema:
            # Pull out the fields of that root-level struct array.
            table_arrays.append(sa.field(subfield.name))

        table = pa.Table.from_arrays(table_arrays, schema=cls.schema)
        return cls(table=table, **kwargs)

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
        column_name equals value.

        :param column_name: The name of the column to select on.
        :param value: The value to match.
        """
        table = self.table.filter(pc.field(column_name) == value)
        return self.__class__(table)

    def sort_by(self, by: Union[str, list[tuple[str, str]]]) -> Self:
        """Sorts the Table by the given column name (or multiple
        columns). This operation requires a copy, and returns a new
        Table using the copied data.

        by should be a column name to sort by, or a list of (column,
        order) tuples, where order can be "ascending" or "descending".

        :param by: The column name or list of (column, order) tuples to sort by.
        """
        table = self.table.sort_by(by)
        return self.__class__(table)

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
            raise TableFragmentedError(
                "Tables cannot be converted to StructArrays while fragmented; call defragment(table) first."
            )
        arrays = [chunked_array.chunks[0] for chunked_array in self.table.columns]
        return pa.StructArray.from_arrays(arrays, fields=list(self.schema))

    def to_dataframe(self, flatten: bool = True) -> pd.DataFrame:
        """Returns self as a pandas DataFrame.

        If flatten is true, then any nested hierarchy is flattened: if
        the Table's schema contains a struct named "foo" with field
        "a", "b", and "c", then the resulting DataFrame will include
        columns "foo.a", "foo.b", "foo.c". This is done fully for any
        deeply nested structure, for example "foo.bar.baz.c".

        If flatten is false, then that struct will be in a single
        "foo" column, and the values will of the column will be
        dictionaries representing the struct values.

        :param flatten: Whether to flatten the table's structure.
        """
        table = self.table
        if flatten:
            table = self.flattened_table()
        df: pd.DataFrame = table.to_pandas()
        return df

    def column(self, column_name: str) -> pa.ChunkedArray:
        """
        Returns the column with the given name as a raw pyarrow ChunkedArray.

        :param column_name: The name of the column to return.
        """
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
        :param \\**kwargs: Additional keyword arguments to pass to Self's __init__ method.

        """
        table = cls._load_parquet_table(
            path=path,
            memory_map=memory_map,
            pq_buffer_size=pq_buffer_size,
            filters=filters,
            column_name_map=column_name_map,
        )
        return cls(table=table, **kwargs)

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
            inverted_map = {v: k for k, v in column_name_map.items()} if column_name_map else {}
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
    def from_feather(cls, path: str, **kwargs: AttributeValueType) -> Self:
        """Read a table from a Feather file.

        :param path: The path to the Feather file.
        :param \\**kwargs: Additional keyword arguments to pass to Self's __init__ method.
        """
        return cls(table=pyarrow.feather.read_table(path), **kwargs)

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
        cls, input_file: Union[str, os.PathLike, IOBase], **kwargs: AttributeValueType  # type: ignore
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
        return cls(table, **kwargs)

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
            except ValidationError as e:
                raise ValidationError(f"Column {name} failed validation: {str(e)}", e.failures) from e

    @classmethod
    def empty(cls, **kwargs: AttributeValueType) -> Self:
        """Create an empty instance of the table.

        :param \\**kwargs: Additional keyword arguments to set the Table's attributes.
        """
        data = [[] for _ in range(len(cls.schema))]  # type: ignore
        empty_table = pa.table(data, schema=cls.schema)
        return cls(table=empty_table, **kwargs)

    def attributes(self) -> dict[str, Any]:
        """Return a dictionary of the table's attributes."""
        return {name: getattr(self, name) for name in self._quivr_attributes}

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

    def apply_mask(self, mask: pa.BooleanArray | np.ndarray[bool, Any] | list[bool]) -> Self:
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
            >>> from quivr import Table, Int64Column
            >>> import pyarrow.compute as pc
            >>> class MyTable(Table):
            ...     x = Int64Column()
            ...     y = Int64Column()
            >>> t = MyTable.from_data(x=[1, 2, 3], y=[4, 5, 6])
            >>> filtered = t.where(pc.field("x") > 1)
            >>> print(filtered.x.to_pylist())
            [2, 3]
            >>> print(filtered.y.to_pylist())
            [5, 6]
        """
        return self.__class__(self.table.filter(expr))
