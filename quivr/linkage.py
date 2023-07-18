from typing import Any, Iterator, Optional

import pyarrow as pa

from . import tables


class ArrowArrayIndex:
    """
    Represents an index over the values in a PyArrow Array.
    """

    index: dict[pa.Scalar, pa.UInt64Array]
    values: set[pa.Scalar]

    def __init__(self, array: pa.Array):
        self.index = {}
        self.values = set()

        if array.null_count > 0:
            raise ValueError("Array must not contain null values to be an index")

        in_progress_index = {}
        for i in range(len(array)):
            val = array[i]
            if val in in_progress_index:
                in_progress_index[val].append(i)
            else:
                in_progress_index[val] = [i]
            self.values.add(val)

        for val, indices in in_progress_index.items():
            self.index[val] = pa.array(indices)

    def get(self, val: pa.Scalar) -> Optional[pa.UInt64Array]:
        return self.index.get(val, None)


class Linkage:
    """
    A Linkage is a mapping of rows across two Tables.

    The mapping is defined by a pair of arrays, one for each table, that
    contain common values.

    The Linkage can be used to iterate over all the unique values in the
    linkage columns, and to select the rows from each table that match a
    particular value.
    """

    left_table: tables.Table
    right_table: tables.Table

    left_index: ArrowArrayIndex
    right_index: ArrowArrayIndex

    all_unique_values: set[pa.Scalar]

    def __init__(
        self,
        left_table: tables.Table,
        right_table: tables.Table,
        left_keys: pa.Array,
        right_keys: pa.Array,
    ):
        """
        Create a new Linkage.

        The linkage is defined by the two tables, and the two arrays of keys.

        The keys must be the same length as the tables, and must not contain
        null values.
        """
        if left_keys.null_count > 0:
            raise ValueError("Left keys must not contain null values")
        if right_keys.null_count > 0:
            raise ValueError("Right keys must not contain null values")

        if len(left_keys) != len(left_table):
            raise ValueError("Left keys must have the same length as the left table")

        if len(right_keys) != len(right_table):
            raise ValueError("Right keys must have the same length as the right table")

        self.left_table = left_table
        self.right_table = right_table

        self.left_index = ArrowArrayIndex(left_keys)
        self.right_index = ArrowArrayIndex(right_keys)

        self.all_unique_values = self.left_index.values.union(self.right_index.values)

    def select_left(self, val: Any) -> tables.Table:
        """
        Select the rows from the left table that match the given value.

        If the value is not present in the left table, then an empty table is
        returned.
        """
        if not isinstance(val, pa.Scalar):
            val = pa.scalar(val)
        return self._select_left(val)

    def _select_left(self, val: pa.Scalar) -> tables.Table:
        left_indices = self.left_index.get(val)
        if left_indices is None:
            return self.left_table.empty()
        else:
            return self.left_table.take(left_indices)

    def select_right(self, val: Any) -> tables.Table:
        """
        Select the rows from the right table that match the given value.

        If the value is not present in the right table, then an empty table is
        returned.
        """
        if not isinstance(val, pa.Scalar):
            val = pa.scalar(val)
        return self._select_right(val)

    def _select_right(self, val: pa.Scalar) -> tables.Table:
        right_indices = self.right_index.get(val)
        if right_indices is None:
            return self.right_table.empty()
        else:
            return self.right_table.take(right_indices)

    def select(self, val: Any) -> tuple[tables.Table, tables.Table]:
        """
        Select the rows from both tables that match the given value.

        If the value is not present in either table, then an empty table is
        returned for that table.
        """
        if not isinstance(val, pa.Scalar):
            val = pa.scalar(val)
        return self._select_left(val), self._select_right(val)

    def __getitem__(self, val: Any):
        return self.select(val)

    def iterate(self) -> Iterator[tuple[pa.Scalar, tables.Table, tables.Table]]:
        """
        Returns an iterator over all the unique values in the linkage, and the rows from
        each table that match that value.
        """
        for val in self.all_unique_values:
            yield val, self._select_left(val), self._select_right(val)

    def __iter__(self):
        return self.iterate()

    def __len__(self):
        """Returns the number of unique values in the linkage."""
        return len(self.all_unique_values)


def composite_array(*keys: list[pa.Array]) -> pa.Array:
    """
    Create a composite array from a list of arrays.
    """
    fields = [pa.field(f"key_{i}", k.type) for i, k in enumerate(keys)]
    return pa.StructArray.from_arrays(keys, fields=fields)
