from typing import Any, Optional
import pyarrow as pa
from . import tables
from . import columns


class ArrowStringArrayIndex:
    """
    Represents an index over the values in a PyArrow StringArray.
    """

    index: dict[pa.Scalar, pa.Int64Array]
    values: set[pa.Scalar]
    unique: bool

    def __init__(self, array: pa.Array):
        self.index = {}
        self.values = set()
        self.unique = True

        if array.null_count > 0:
            raise ValueError("Array must not contain null values to be an index")

        in_progress_index = {}
        for i in range(len(array)):
            val = array[i]
            if val in in_progress_index:
                in_progress_index[val].append(i)
                self.unique = False
            else:
                in_progress_index[val] = [i]
            self.values.add(val)

        for val, indices in in_progress_index.items():
            self.index[val] = pa.array(indices)

    def get(self, val: pa.Scalar) -> Optional[pa.Int64Array]:
        return self.index.get(val, None)


class Linkage:
    def __init__(
        self,
        left_table: tables.Table,
        right_table: tables.Table,
        left_column: columns.Column,
        right_column: columns.Column,
    ):
        self.left_table = left_table
        self.right_table = right_table
        self.left_column = left_column
        self.right_column = right_column

        self.left_index = self._build_index(left_table, left_column)
        self.right_index = self._build_index(right_table, right_column)

        self.all_unique_values = self.left_index.values.union(self.right_index.values)

    def _build_index(self, table: tables.Table, column: columns.Column) -> ArrowStringArrayIndex:
        arrow_array = table.column(column.name)
        return ArrowStringArrayIndex(arrow_array)

    def _select_left(self, val: Any) -> tables.Table:
        left_indices = self.left_index.get(val)
        if left_indices is None:
            return self.left_table.empty()
        else:
            return self.left_table.take(left_indices)

    def _select_right(self, val: Any) -> tables.Table:
        right_indices = self.right_index.get(val)
        if right_indices is None:
            return self.right_table.empty()
        else:
            return self.right_table.take(right_indices)

    def __getitem__(self, val: Any):
        if not isinstance(val, pa.Scalar):
            val = pa.scalar(val)
        return self._select_left(val), self._select_right(val)

    def __iter__(self):
        for v in self.all_unique_values:
            yield self[v]

    def __len__(self):
        return len(self.all_unique_values)
