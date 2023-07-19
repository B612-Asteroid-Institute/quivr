from typing import TypeVar

from .tables import Table

#: Type hint for a generic table.
GenericTable = TypeVar("GenericTable", bound=Table)


def defragment(table: GenericTable) -> GenericTable:
    """Condense the underlying memory which backs the table to make
    it all contiguous. This makes many operations more efficient after
    defragmentation is complete.

    :param table: The table to defragment.
    :return: The defragmented table.
    """

    combined = table.table.combine_chunks()
    return table.__class__(table=combined)
