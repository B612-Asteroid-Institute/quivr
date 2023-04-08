import pyarrow as pa
from typing import Optional, TypeVar, Generic, Union
import functools
import pickle
import pandas as pd
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from .errors import TableFragmentedError

_METADATA_MODEL_KEY = b"__quiver_model_pickle"
_METADATA_NAME_KEY = b"__quiver_model_name"
_METADATA_UNPICKLE_KWARGS_KEY = b"__quiver_model_unpickle_kwargs"


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

        def setter(_self):
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

    def __init__(self, table: pa.Table):
        if not isinstance(table, pa.Table):
            raise TypeError(
                f"Data must be a pyarrow.Table for {self.__class__.__name__}"
            )
        if table.schema != self.schema:
            raise TypeError(
                f"Data schema must match schema for {self.__class__.__name__}"
            )
        self.table = table

    @classmethod
    def from_arrays(cls, l: list[pa.array]):
        table = pa.Table.from_arrays(l, schema=cls.schema)
        return cls(table=table)

    @classmethod
    def from_pydict(cls, d: dict[str, Union[pa.array, list, np.ndarray]]):
        table = pa.Table.from_pydict(d, schema=cls.schema)
        return cls(table=table)

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

    @classmethod
    def as_field(
        cls, name: str, nullable: bool = True, metadata: Optional[dict] = None
    ):
        metadata = metadata or {}
        metadata[_METADATA_NAME_KEY] = cls.__name__
        metadata[_METADATA_MODEL_KEY] = pickle.dumps(cls)
        field = pa.field(
            name, pa.struct(cls.schema), nullable=nullable, metadata=metadata
        )
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
                init_kwargs = pickle.loads(
                    field.metadata[_METADATA_UNPICKLE_KWARGS_KEY]
                )
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


def _sub_table(tab: pa.Table, field_name: str):
    """Given a table which contains a StructArray under given field
    name, construct a table from the sub-object.

    """
    column = tab.column(field_name)
    schema = pa.schema(column.type)
    return pa.Table.from_arrays(column.flatten(), schema=schema)
