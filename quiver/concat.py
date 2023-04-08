from typing import TypeVar, Iterator
from .tables import TableBase
import pyarrow as pa


Table = TypeVar("Table", bound=TableBase)


def concatenate(values: Iterator[Table]) -> Table:
    """
    Concatenate a collection of Tables into a single Table.
    """
    batches = []
    for v in values:
        batches += v.table.to_batches()
    table = pa.Table.from_batches(batches)
    cls = values[0].__class__
    return cls(data=table)
