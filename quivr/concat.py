from typing import Iterator

import pyarrow as pa

from . import defragment, errors, tables


def concatenate(values: Iterator[tables.AnyTable], defrag: bool = True) -> tables.AnyTable:
    """Concatenate a collection of Tables into a single Table.

    All input Tables be of the same class, and have the same attribute
    values (if any).

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
            first_cls = v.__class__
            first_val = v
            first = False
        else:
            if v.__class__ != first_cls:
                raise errors.TablesNotCompatibleError("All tables must be the same class to concatenate")
            if not first_val._attr_equal(v):
                raise errors.TablesNotCompatibleError(
                    "All tables must have the same attribute values to concatenate"
                )

    if len(batches) == 0:
        raise ValueError("No values to concatenate")
    table = pa.Table.from_batches(batches)
    result = first_cls.from_pyarrow(table=table)
    if defrag:
        result = defragment.defragment(result)
    return result
