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

    # Note, we don't return immediately if there is only one table,
    # because we still want to optionally defragment the result.

    first_cls = values_list[0].__class__
    first_val = values_list[0]
    # Find the first non-empty table to get the class for attribute comparison
    for v in values_list:
        if len(v) > 0:
            first_cls = v.__class__
            first_val = v
            break

    pa_tables = []
    for v in values_list:
        if v.__class__ != first_cls:
            raise errors.TablesNotCompatibleError("All tables must be the same class to concatenate")
        if not first_val._attr_equal(v) and len(v) > 0:
            raise errors.TablesNotCompatibleError(
                "All non-empty tables must have the same attribute values to concatenate"
            )
        # Skip empty tables: their metadata can override that of non-empty
        # tables (pa.concat_tables takes schema from the first input), and
        # they contribute no rows.
        if len(v) > 0:
            pa_tables.append(v.table)

    if not pa_tables:
        # All tables were empty; return the first to preserve attributes.
        table = first_val.table
    else:
        table = pa.concat_tables(pa_tables)

    # We re-initialize the table to optionally validate and create
    # a unique object
    result = first_cls.from_pyarrow(table=table, validate=validate)

    if defrag:
        result = defragment.defragment(result)
    return result
