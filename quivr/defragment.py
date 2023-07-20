from __future__ import annotations

from . import tables


def defragment(table: tables.AnyTable) -> tables.AnyTable:
    """Condense the underlying memory which backs the table to make
    it all contiguous. This makes many operations more efficient after
    defragmentation is complete.

    :param table: The table to defragment.
    :return: The defragmented table.
    """

    combined = table.table.combine_chunks()
    return table.__class__(table=combined)
