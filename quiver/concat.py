from typing import Iterator, TypeVar

import pyarrow as pa

from .defragment import defragment
from .tables import TableBase

Table = TypeVar("Table", bound=TableBase)


def concatenate(values: Iterator[Table], defrag: bool = True) -> Table:
    """Concatenate a collection of Tables into a single Table.

    All input Tables must have the same schema (typically, this will
    be from being the same class).

    By default, results are compacted to be contiguous in memory,
    which involves a copy. In a tight loop, this can be very
    inefficient, so you can set the 'defrag' parameter to False to
    skip this compaction step, and instead call quiver.defragment on
    the result after the loop is complete.

    """
    batches = []
    first = True
    for v in values:
        batches += v.table.to_batches()
        if first:
            cls = v.__class__
            first = False
    table = pa.Table.from_batches(batches)
    result = cls(table=table)
    if defrag:
        result = defragment(result)
    return result
