from typing import Generic, Optional, TypeVar

import mmh3
import pyarrow as pa
import pyarrow.compute as pc

from .tables import TableBase

T = TypeVar("T", bound=TableBase)


class StringIndex(Generic[T]):
    """StringIndex is a simple index that maps a string column to a
    list of row indices. It can be used for fast lookups of sub-slices
    of a Table based on string values.

    Example usage:

    >>> from indexing import StringIndex
    >>> from examples.coordinates import create_example_orbits
    >>> orbits = create_example_orbits(1000)
    >>> index = StringIndex(orbits, 'object_id')
    >>> index.lookup('id990')
    Orbit(size=1)

    This is equivalent to calling table.select("object_id", "id990"),
    but it is about 5x faster. The tradeoff is that building the
    StringIndex takes about 450 microseconds for this example.

    """

    def __init__(self, table: T, column: str, size: int = 1000):
        self.table = table
        self.column = column
        self.idx = self._build_index(size)

    def _build_index(self, size: int):
        index_array = [None for _ in range(size)]
        data = self.table.table.column(self.column)
        assert data.type == pa.string()

        for i, value in enumerate(data.to_numpy()):
            slot = mmh3.hash(value) % size
            if index_array[slot] is None:
                index_array[slot] = {value: [i]}
            elif value not in index_array[slot]:
                index_array[slot][value] = [i]
            else:
                index_array[slot][value].append(i)

        null_mask = [True if x is None else False for x in index_array]

        return pa.array(index_array, type=pa.map_(pa.string(), pa.list_(pa.int32())), mask=null_mask)

    def _indices(self, value: str) -> Optional[pa.Int32Array]:
        slot = mmh3.hash(value) % len(self.idx)
        map_scalar = self.idx[slot]
        if not map_scalar.is_valid:
            return None

        index_list = pc.map_lookup(map_scalar, query_key=pa.scalar(value), occurrence="all")
        if not index_list.is_valid:
            return None

        return index_list.values.flatten()

    def lookup(self, value: str) -> Optional[T]:
        index_list = self._indices(value)
        if index_list is None:
            return None

        return self.table.take(index_list)
