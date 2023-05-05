import os
from io import IOBase
from typing import Any, ClassVar, Optional, Self, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.parquet

from .errors import TableFragmentedError
from .fields import Field, MetadataDict, SubTableField
from .schemagraph import _walk_schema


class Table:
    schema: ClassVar[pa.Schema]
    table: pa.Table

    def __init_subclass__(cls, **kwargs):
        fields = []
        for name, field in cls.__dict__.items():
            if isinstance(field, Field):
                fields.append(field.pyarrow_field())

        # Generate a pyarrow schema
        schema = pa.schema(fields)
        cls.schema = schema
        super().__init_subclass__(**kwargs)

    def __init__(self, pa_table: pa.Table):
        self.table = pa_table

    @classmethod
    def from_data(cls, data: Optional[Any] = None, **kwargs) -> Self:
        """
        Create an instance of the Table and populate it with data.

        This is a convenience method which tries to infer the right
        underlying constructors to use based on the type of data. It
        can also accept keyword-style arguments to pass data in. If
        you know the data's structure well in advance, the more
        precise constructors (from_arrays, from_pylist, etc) should be
        preferred.

        Examples:
            >>> import quivr
            >>> class MyTable(quivr.TableBase):
            ...     schema = pyarrow.schema([
            ...         pyarrow.field("a", pyarrow.string()),
            ...         pyarrow.field("b", pyarrow.int64()),
            ...     ])
            ...
            >>> # All of these are equivalent:
            >>> MyTable.from_data([["a", 1], ["b", 2]])
            MyTable(size=2)
            >>> MyTable.from_data({"a": ["a", "b"], "b": [1, 2]})
            MyTable(size=2)
            >>> MyTable.from_data([{"a": "a", "b": 1}, {"a": "b", "b": 2}])
            MyTable(size=2)
            >>> MyTable.from_data(a=["a", "b"], b=[1, 2])
            MyTable(size=2)
            >>> import numpy as np
            >>> MyTable.from_data(a=np.array(["a", "b"]), b=np.array([1, 2]))
            MyTable(size=2)
        """
        if data is None:
            return cls.from_kwargs(**kwargs)

        if isinstance(data, pa.Table):
            return cls(pa_table=data)
        if isinstance(data, dict):
            return cls.from_pydict(data)
        if isinstance(data, list):
            if len(data) == 0:
                return cls.from_rows(data)
            if isinstance(data[0], dict):
                return cls.from_rows(data)
            elif isinstance(data[0], list):
                return cls.from_lists(data)
        if isinstance(data, pd.DataFrame):
            return cls.from_dataframe(data)
        raise TypeError(f"Unsupported type: {type(data)}")

    @classmethod
    def as_field(cls, nullable: bool = True, metadata: Optional[MetadataDict] = None) -> SubTableField[Self]:
        return SubTableField(cls, nullable=nullable, metadata=metadata)

    @classmethod
    def from_kwargs(cls, **kwargs) -> Self:
        """Create a Table instance from keyword arguments.

        Each keyword argument corresponds to a field in the Table.

        The keys should correspond to the field names, and the values
        can be a list, numpy array, pyarrow array, or Table instance.
        """
        arrays = []
        for column_name in cls.schema.names:
            if column_name not in kwargs:
                raise ValueError(f"Missing column {column_name}")
            value = kwargs[column_name]
            if isinstance(value, Table):
                arrays.append(value.to_structarray())
            elif isinstance(value, pa.Array):
                arrays.append(value)
            elif isinstance(value, np.ndarray):
                arrays.append(pa.array(value))
            elif isinstance(value, list):
                arrays.append(pa.array(value))
            else:
                raise TypeError(f"Unsupported type for {column_name}: {type(value)}")
        return cls.from_arrays(arrays)

    @classmethod
    def from_arrays(cls, arrays: list[pa.array]) -> Self:
        """Create a Table object from a list of arrays.

        Args:
            arrays: A list of pyarrow.Array objects.

        Returns:
            A Table object.

        """
        table = pa.Table.from_arrays(arrays, schema=cls.schema)
        return cls(pa_table=table)

    @classmethod
    def from_pydict(cls, d: dict[str, Union[pa.array, list, np.ndarray]]) -> Self:
        table = pa.Table.from_pydict(d, schema=cls.schema)
        return cls(pa_table=table)

    @classmethod
    def from_rows(cls, rows: list[dict]) -> Self:
        """
        Create a Table object from a list of dictionaries.

        Args:
            rows: A list of values. Each value corresponds to a row in the table.

        Returns:
            A Table object.

        Examples:
            >>> import quivr
            >>> class Inner(quivr.Table):
            ...     a = quivr.StringField()
            ...
            >>> class Outer(quivr.TableBase):
            ...     z = quivr.StringField()
            ...     i = Inner.as_field()
            ...
            >>> data = [{"z": "v1", "i": {"a": "v1_in"}}, {"z": "v2", "i": {"a": "v2_in"}}]
            >>> Outer.from_pylist(data)
            Outer(size=2)
        """
        table = pa.Table.from_pylist(rows, schema=cls.schema)
        return cls(pa_table=table)

    @classmethod
    def from_lists(cls, lists: list[list]) -> Self:
        """Create a Table object from a list of lists.

        Each inner list corresponds to a field in the Table. They
        should be specified in the same order as the fields in the
        class.

        Args:
            lists: A list of lists. Each inner list corresponds to a column in the table.

        Returns:
            A TableBase object.

        """
        table = pa.Table.from_arrays(list(map(pa.array, lists)), schema=cls.schema)
        return cls(pa_table=table)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """Load a DataFrame into the Table.

        If the DataFrame is missing any of the Table's columns, an
        error is raised. If the DataFrame has extra columns, they are
        ignored.

        This function cannot load "flattened" dataframes. This only
        matters for nested Tables which contain other Table
        definitions as fields. For that use case, either load an
        unflattened DataFrame, or use from_flat_dataframe.
        """

        table = pa.Table.from_pandas(df, schema=cls.schema)
        return cls(pa_table=table)

    @classmethod
    def _unflatten_table(cls, table: pa.Table):
        """Unflatten a Table.

        This is used when loading a flattened DataFrame into a nested
        Table. It takes a Table with a flat schema, and returns a
        Table with a nested schema.

        """
        struct_fields = []

        for field in cls.schema:
            if pa.types.is_struct(field.type):
                struct_fields.append(field)

        if len(struct_fields) == 0:
            return cls(pa_table=table)

        # Walk the schema, and build a StructArray for each embedded
        # type.

        def struct_array_for(field: pa.Field, ancestors: list[pa.Field]):
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
    def from_flat_dataframe(cls, df: pd.DataFrame):
        """Load a flattened DataFrame into the Table.

        known bug: Doesn't correctly interpret fixed-length lists.
        """
        struct_fields = []
        for field in cls.schema:
            if pa.types.is_struct(field.type):
                struct_fields.append(field)

        if len(struct_fields) == 0:
            return cls(pa_table=pa.from_dataframe(df, schema=cls.schema))

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

        def visitor(field: pa.Field, ancestors: list[pa.Field]):
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
        return cls(pa_table=table)

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

        """
        table = self.table.filter(pc.field(column_name) == value)
        return self.__class__(table)

    def sort_by(self, by: Union[str, list[tuple[str, str]]]) -> Self:
        """Sorts the Table by the given column name (or multiple
        columns). This operation requires a copy, and returns a new
        Table using the copied data.

        by should be a column name to sort by, or a list of (column,
        order) tuples, where order can be "ascending" or "descending".

        """
        table = self.table.sort_by(by)
        return self.__class__(table)

    def chunk_counts(self) -> dict[str, int]:
        """Returns the number of discrete memory chunks that make up
        each of the Table's underlying arrays. The keys of the
        resulting dictionary are the field names, and the values are
        the number of chunks for that field's data.

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

        """
        table = self.table
        if flatten:
            table = self.flattened_table()
        return table.to_pandas()

    def column(self, field_name: str) -> pa.ChunkedArray:
        """Returns the column with the given name as a raw pyarrow ChunkedArray."""
        return self.table.column(field_name)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={len(self.table)})"

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        # TODO: This comes out a little funky. You get chunked arrays
        # instead of arrays. Is there a way to flatten them safely?

        if isinstance(idx, int):
            return self.__class__(self.table[idx : idx + 1])
        return self.__class__(self.table[idx])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i : i + 1]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Table):
            return self.table.equals(other.table)
        if isinstance(other, pa.Table):
            return self.table.equals(other)
        return False

    def take(self, row_indices: Union[list[int], pa.IntegerArray]) -> Self:
        """Return a new Table with only the rows at the given indices."""
        return self.__class__(self.table.take(row_indices))

    def to_parquet(self, path: str, **kwargs):
        """Write the table to a Parquet file."""
        pyarrow.parquet.write_table(self.table, path, **kwargs)

    @classmethod
    def from_parquet(cls, path: str, **kwargs):
        """Read a table from a Parquet file."""
        return cls(pa_table=pyarrow.parquet.read_table(path, **kwargs))

    def to_feather(self, path: str, **kwargs):
        """Write the table to a Feather file."""
        pyarrow.feather.write_feather(self.table, path, **kwargs)

    @classmethod
    def from_feather(cls, path: str, **kwargs):
        """Read a table from a Feather file."""
        return cls(pa_table=pyarrow.feather.read_table(path, **kwargs))

    def to_csv(self, path: str):
        """Write the table to a CSV file. Any nested structure is flattened."""
        pyarrow.csv.write_csv(self.flattened_table(), path)

    @classmethod
    def from_csv(cls, input_file: Union[str, os.PathLike, IOBase]):
        """Read a table from a CSV file."""
        flat_table = pyarrow.csv.read_csv(input_file)
        return cls(pa_table=cls._unflatten_table(flat_table))
