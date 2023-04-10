import pyarrow as pa

from quiver.indexing import StringIndex
from quiver.tables import TableBase


class TableWithString(TableBase):
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
            pa.field("value", pa.int64()),
        ]
    )


def test_indexing():
    table = TableWithString.from_arrays(
        [pa.array([1, 2, 3]), pa.array(["a", "b", "c"]), pa.array([4, 5, 6])]
    )
    index = StringIndex(table, "name")
    assert len(index.lookup("a")) == 1
    assert index.lookup("a").id.to_pylist() == [1]
    assert len(index.lookup("b")) == 1
    assert index.lookup("b").id.to_pylist() == [2]


def test_indexing_duplicate():
    table = TableWithString.from_arrays(
        [pa.array([1, 2, 3]), pa.array(["a", "a", "c"]), pa.array([4, 5, 6])]
    )
    index = StringIndex(table, "name")
    assert len(index.lookup("a")) == 2
    assert index.lookup("a").id.to_pylist() == [1, 2]
    assert index.lookup("b") is None
