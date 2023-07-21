from typing import Iterator

import pyarrow as pa

from . import tables
from .defragment import defragment


def concatenate(values: Iterator[tables.AnyTable], defrag: bool = True) -> tables.AnyTable:
    """Concatenate a collection of Tables into a single Table.

    All input Tables must have the same schema (typically, this will
    be from being the same class).

    By default, results are compacted to be contiguous in memory,
    which involves a copy. In a tight loop, this can be very
    inefficient, so you can set the 'defrag' parameter to False to
    skip this compaction step, and instead call :func:`defragment` on
    the result after the loop is complete.

    :param values: An iterator of :class:`Table` instances to concatenate.
    :param defrag: Whether to compact the result to be contiguous in
        memory. Defaults to True.
    """
    batches = []
    first = True
    for v in values:
        batches += v.table.to_batches()
        if first:
            cls = v.__class__
            first = False
    if len(batches) == 0:
        raise ValueError("No values to concatenate")
    table = pa.Table.from_batches(batches)
    result = cls(table=table)
    if defrag:
        result = defragment(result)
    return result
