"""Table base classes."""

import functools
import pickle
from typing import Any, Optional, Self, Union
from io import IOBase
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.parquet

from .errors import TableFragmentedError
from .schemagraph import _walk_schema, compute_depth

_METADATA_MODEL_KEY = b"__quivr_model_pickle"
_METADATA_NAME_KEY = b"__quivr_model_name"
_METADATA_UNPICKLE_KWARGS_KEY = b"__quivr_model_unpickle_kwargs"


class TableMetaclass(type):
    """TableMetaclass is a metaclass which attaches accessors
    to Tables based on their schema class-level attribute.

    Each field in the class's schema becomes an attribute on the class.

    """

    def __new__(cls, name, bases, attrs):
        # Invoked when a class is created. We use this to generate
        # accessors for the class's schema's fields.
        if "schema" not in attrs:
            raise TypeError(f"Table {name} requires a schema attribute")
        if not isinstance(attrs["schema"], pa.Schema):
            raise TypeError(f"Table {name} schema attribute must be a pyarrow.Schema")
        accessors = dict(cls.generate_accessors(attrs["schema"]))
        attrs.update(accessors)

        # Compute the depth of the schema, which is used when flattening.
        attrs["_schema_depth"] = compute_depth(attrs["schema"])

        return super().__new__(cls, name, bases, attrs)

    def generate_accessors(schema: pa.Schema):
        """Generate all the property accessors for the schema's fields.

        Each field is accessed by name. When getting the field, its
        underlying value is unloaded out of the Arrow array. If the
        field has a model attached to it, the model is instantiated
        with the data. Otherwise, the data is returned as-is.

        """

        def getter(_self, field: pa.Field):
            return _self.column(field.name)

        def setter(_self, value: Any):
            raise NotImplementedError("Tables are immutable")

        def deleter(_self):
            raise NotImplementedError("Tables are immutable")

        for idx, field in enumerate(schema):
            g = functools.partial(getter, field=field)
            prop = property(fget=g, fset=setter, fdel=deleter)
            yield (field.name, prop)


class TableBase(metaclass=TableMetaclass):
    table: pa.Table
    schema: pa.Schema = pa.schema([])
    _schema_depth: int

    def __init__(self, table: pa.Table):
        self.table = []
        if not isinstance(table, pa.Table):
            raise TypeError(f"Data must be a pyarrow.Table for {self.__class__.__name__}")
        if table.schema != self.schema:
            raise TypeError(f"Data schema must match schema for {self.__class__.__name__}")
        self.table = table

    @classmethod
    def from_data(cls, data: Optional[Any] = None, **kwargs) -> Self:
        """Create an instance of the TableBase and populate it with data.

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
            return cls(table=data)
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
            return cls.from_pandas(data)
        if isinstance(data, np.ndarray):
            return cls.from_numpy(data)
        raise TypeError(f"Unsupported data type: {type(data)}")

    @classmethod
    def from_kwargs(cls, **kwargs) -> Self:
        """Create a TableBase object from keyword arguments.

        Each keyword argument corresponds to a column in the table.

        Each keyword value can be a list, numpy array, pyarrow array,
        or TableBase instance.

        """

        arrays = []
        for column_name in cls.schema.names:
            if column_name not in kwargs:
                raise ValueError(f"Missing column {column_name}")
            value = kwargs[column_name]
            if isinstance(value, TableBase):
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
        """Create a TableBase object from a list of arrays.

        Args:
            arrays: A list of pyarrow.Array objects.

        Returns:
            A TableBase object.

        """
        table = pa.Table.from_arrays(arrays, schema=cls.schema)
        return cls(table=table)

    @classmethod
    def from_pydict(cls, d: dict[str, Union[pa.array, list, np.ndarray]]):
        table = pa.Table.from_pydict(d, schema=cls.schema)
        return cls(table=table)

    @classmethod
    def from_rows(cls, l: list[dict]):
        """
        Create a TableBase object from a list of dictionaries.

        Args:
            l: A list of values. Each value corresponds to a row in the table.

        Returns:
            A TableBase object.

        Examples:
            >>> import quivr
            >>> class Inner(quivr.TableBase):
            ...     schema = pyarrow.schema([pyarrow.field("a", pyarrow.string())])
            ...
            >>> class Outer(quivr.TableBase):
            ...     schema = pyarrow.schema([pyarrow.field("z", pyarrow.string()), Inner.as_field("i")])
            ...
            >>> data = [{"z": "v1", "i": {"a": "v1_in"}}, {"z": "v2", "i": {"a": "v2_in"}}]
            >>> Outer.from_pylist(data)
            Outer(size=2)
        """
        table = pa.Table.from_pylist(l, schema=cls.schema)
        return cls(table=table)

    @classmethod
    def from_lists(cls, l: list[list]) -> Self:
        """Create a TableBase object from a list of lists.

        Each inner list corresponds to a column in the table. They
        should be specified in the same order as the columns in the
        schema.

        Args:
            l: A list of lists. Each inner list corresponds to a column in the table.

        Returns:
            A TableBase object.
        """
        table = pa.Table.from_arrays(list(map(pa.array, l)), schema=cls.schema)
        return cls(table=table)

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
        return cls(table=table)

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
            return cls(table=pa.from_dataframe(df, schema=cls.schema))

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
            return cls(table=pa.from_dataframe(df, schema=cls.schema))

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
        return cls(table=table)

    def flattened_table(self) -> pa.Table:
        """Completely flatten the Table's underlying Arrow table,
        taking into account any nested structure, and return the data
        table itself.

        """
        table = self.table
        for i in range(self._schema_depth - 1):
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

    @classmethod
    def as_field(cls, name: str, nullable: bool = True, metadata: Optional[dict] = None):
        metadata = metadata or {}
        metadata[_METADATA_NAME_KEY] = cls.__name__
        metadata[_METADATA_MODEL_KEY] = pickle.dumps(cls)
        field = pa.field(name, pa.struct(cls.schema), nullable=nullable, metadata=metadata)
        return field

    def column(self, field_name: str):
        field = self.schema.field(field_name)
        if field.metadata is not None and _METADATA_MODEL_KEY in field.metadata:
            # If the field has type information attached to it in
            # metadata, pull it out. The metadata store the model (as
            # a class object), and may optionally have some keyword
            # arguments to be used when instantiating the model from
            # the data.
            model = pickle.loads(field.metadata[_METADATA_MODEL_KEY])
            if _METADATA_UNPICKLE_KWARGS_KEY in field.metadata:
                init_kwargs = pickle.loads(field.metadata[_METADATA_UNPICKLE_KWARGS_KEY])
            else:
                init_kwargs = {}
            table = _sub_table(self.table, field_name)
            return model(table=table, **init_kwargs)
        return self.table.column(field_name)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={len(self.table)})"

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.__class__(self.table[idx : idx + 1])
        return self.__class__(self.table[idx])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i : i + 1]

    def __eq__(self, other):
        if not isinstance(other, TableBase):
            return NotImplemented
        return self.table.equals(other.table)

    def take(self, row_indices: Union[list[int], pa.IntegerArray]) -> Self:
        """Return a new Table with only the rows at the given indices."""
        return self.__class__(self.table.take(row_indices))

    def to_parquet(self, path: str, **kwargs):
        """Write the table to a Parquet file."""
        pyarrow.parquet.write_table(self.table, path, **kwargs)

    @classmethod
    def from_parquet(cls, path: str, **kwargs):
        """Read a table from a Parquet file."""
        return cls(table=pyarrow.parquet.read_table(path, **kwargs))

    def to_feather(self, path: str, **kwargs):
        """Write the table to a Feather file."""
        pyarrow.feather.write_feather(self.table, path, **kwargs)

    @classmethod
    def from_feather(cls, path: str, **kwargs):
        """Read a table from a Feather file."""
        return cls(table=pyarrow.feather.read_table(path, **kwargs))

    def to_csv(self, path: str):
        """Write the table to a CSV file. Any nested structure is flattened."""
        pyarrow.csv.write_csv(self.flattened_table(), path)

    @classmethod
    def from_csv(cls, input_file: Union[str, os.PathLike, IOBase]):
        """Read a table from a CSV file."""
        flat_table = pyarrow.csv.read_csv(input_file)
        return cls(table=cls._unflatten_table(flat_table))


def _sub_table(tab: pa.Table, field_name: str):
    """Given a table which contains a StructArray under given field
    name, construct a table from the sub-object.

    """
    column = tab.column(field_name)
    schema = pa.schema(column.type)
    return pa.Table.from_arrays(column.flatten(), schema=schema)
