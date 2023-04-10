from typing import TypeVar

from .tables import TableBase

GenericTable = TypeVar("GenericTable", bound=TableBase)


def defragment(table: GenericTable) -> GenericTable:
    """Condense the underlying memory which backs the table to make
    it all contiguous. This makes many operations more efficient after
    defragmentation is complete.

    """

    combined = table.table.combine_chunks()
    return table.__class__(table=combined)
