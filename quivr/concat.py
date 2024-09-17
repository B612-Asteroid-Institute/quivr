from typing import Iterable, List

import pyarrow as pa

from . import defragment, errors, tables


def concatenate(
    values: Iterable[tables.AnyTable], defrag: bool = True, validate: bool = True
) -> tables.AnyTable:
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
    values_list: List[tables.AnyTable] = list(values)

    if len(values_list) == 0:
        raise ValueError("No values to concatenate")

    batches = []
    first_full = False

    # Find the first non-empty table to get the class
    for v in values_list:
        if not first_full and len(v) > 0:
            first_cls = v.__class__
            first_val = v
            first_full = True
            break

    # No non-empty tables found so lets pick the first table
    # to get the class and attributes
    if not first_full:
        first_cls = values_list[0].__class__
        first_val = values_list[0]

    # Scan the values and now make sure they are all the same class
    # as the first non-empty table
    for v in values_list:
        batches += v.table.to_batches()
        if v.__class__ != first_cls:
            raise errors.TablesNotCompatibleError("All tables must be the same class to concatenate")
        if not first_val._attr_equal(v) and len(v) > 0:
            raise errors.TablesNotCompatibleError(
                "All non-empty tables must have the same attribute values to concatenate"
            )

    if len(batches) == 0:
        return first_cls.empty()

    table = pa.Table.from_batches(batches)
    result = first_cls.from_pyarrow(table=table, validate=validate)
    if defrag:
        result = defragment.defragment(result)
    return result
