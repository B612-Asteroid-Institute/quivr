from typing import TypeVar, Iterator
from .tables import TableBase
import pyarrow as pa


GenericTable = TypeVar("GenericTable", bound=TableBase)


def defragment(table: GenericTable) -> GenericTable:
    """Condense the underlying memory which backs the table to make
    it all contiguous. This makes many operations more efficient after
    defragmentation is complete.

    """

    combined = table.table.combine_chunks()
    return table.__class__(table=combined)
