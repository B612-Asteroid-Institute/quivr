from quiver.tables import TableBase
from quiver.concat import concatenate
import pyarrow as pa


class Pair(TableBase):
    schema = pa.schema(
        [
            pa.field("x", pa.int64()),
            pa.field("y", pa.int64()),
        ]
    )


class Wrapper(TableBase):
    schema = pa.schema([Pair.as_field("pair"), pa.field("id", pa.string())])


def test_create_from_arrays():
    xs = pa.array([1, 2, 3], pa.int64())
    ys = pa.array([4, 5, 6], pa.int64())
    have = Pair.from_arrays([xs, ys])
    assert len(have) == 3
    assert have.column("x").to_pylist() == [1, 2, 3]
    assert have.column("y").to_pylist() == [4, 5, 6]


def test_create_wrapped_from_arrays():
    xs = pa.array([1, 2, 3], pa.int64())
    ys = pa.array([4, 5, 6], pa.int64())
    pairs = pa.StructArray.from_arrays([xs, ys], fields=list(Pair.schema))
    ids = pa.array(["v1", "v2", "v3"], pa.string())

    have = Wrapper.from_arrays([pairs, ids])
    assert len(have) == 3
    assert have.column("id").to_pylist() == ["v1", "v2", "v3"]


def test_create_from_pydict():
    have = Pair.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert len(have) == 3
    assert have.column("x").to_pylist() == [1, 2, 3]
    assert have.column("y").to_pylist() == [4, 5, 6]


def test_table_to_structarray():
    xs = pa.array([1, 2, 3], pa.int64())
    ys = pa.array([4, 5, 6], pa.int64())
    pair = Pair.from_arrays([xs, ys])

    want = pa.StructArray.from_arrays([xs, ys], fields=list(Pair.schema))

    have = pair.to_structarray()
    assert have == want


def test_create_wrapped_from_pydict():
    have = Wrapper.from_pydict(
        {
            "id": ["v1", "v2", "v3"],
            "pair": [
                {"x": 1, "y": 2},
                {"x": 3, "y": 4},
                {"x": 5, "y": 6},
            ],
        }
    )
    assert len(have) == 3
    assert have.column("id").to_pylist() == ["v1", "v2", "v3"]


def test_generated_accessors():
    have = Pair.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert have.x.to_pylist() == [1, 2, 3]
    assert have.y.to_pylist() == [4, 5, 6]


def test_iteration():
    pair = Pair.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    values = list(pair)
    assert len(values) == 3
    assert len(values[0]) == 1
    assert len(values[0].x) == 1
    assert len(values[0].y) == 1
    assert values[0].x[0].as_py() == 1
    assert values[0].y[0].as_py() == 4

    assert values[1].x[0].as_py() == 2
    assert values[1].y[0].as_py() == 5

    assert values[2].x[0].as_py() == 3
    assert values[2].y[0].as_py() == 6


def test_chunk_counts():
    pair = Pair.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert pair.chunk_counts() == {"x": 1, "y": 1}
    pair = concatenate([pair, pair], defrag=False)
    assert pair.chunk_counts() == {"x": 2, "y": 2}


def test_check_fragmented():
    pair = Pair.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert not pair.fragmented()
    pair = concatenate([pair, pair], defrag=False)
    assert pair.fragmented()
