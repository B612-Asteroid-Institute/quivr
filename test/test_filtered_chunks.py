import pyarrow as pa

import quivr as qv


def test_fast_combine_chunks_masked_array():
    # Regression test for github.com/spenczar/quivr/issues/51
    # Create a ChunkedArray with entirely masked values
    class MyTable(qv.Table):
        col = qv.Int64Column()

    table = MyTable.from_kwargs(col=[1, 2, 3, 4, 5])
    filtered = table.table.filter(pa.array([False, False, False, False, False]))
    filtered_quivr = MyTable(filtered)
    # This works
    assert filtered_quivr.column("col").to_pylist() == []
    # This fails due to something with _fast_combine_chunks
    assert filtered_quivr.col.to_pylist() == []
