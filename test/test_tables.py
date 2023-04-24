import pyarrow as pa

from quivr.concat import concatenate
from quivr.tables import TableBase


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


def test_select():
    pair = Pair.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    have = pair.select("x", 3)
    assert len(have) == 1
    assert have.y[0].as_py() == 6


def test_select_empty():
    pair = Pair.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    have = pair.select("x", 4)
    assert len(have) == 0


def test_sort_by():
    pair = Pair.from_pydict({"x": [1, 2, 3], "y": [5, 1, 2]})

    sorted1 = pair.sort_by("y")
    assert sorted1.x[0].as_py() == 2
    assert sorted1.x[1].as_py() == 3
    assert sorted1.x[2].as_py() == 1

    sorted2 = pair.sort_by([("x", "descending")])
    assert sorted2.x[0].as_py() == 3
    assert sorted2.x[1].as_py() == 2
    assert sorted2.x[2].as_py() == 1

def test_from_pylist():
    data = [{"id": "1", "pair": {"x": 1, "y": 2}}, {"id": "2", "pair": {"x": 3, "y": 4}}]
    wrapper = Wrapper.from_pylist(data)

    assert wrapper.id.to_pylist() == ["1", "2"]
    assert wrapper.pair.x.to_pylist() == [1, 3]
    assert wrapper.pair.y.to_pylist() == [2, 4]


class Layer1(TableBase):
    schema = pa.schema([("x", pa.int64())])


class Layer2(TableBase):
    schema = pa.schema([("y", pa.int64()), Layer1.as_field("layer1")])


class Layer3(TableBase):
    schema = pa.schema([("z", pa.int64()), Layer2.as_field("layer2")])


def test_unflatten_table():
    data = [
        {"z": 1, "layer2": {"y": 2, "layer1": {"x": 3}}},
        {"z": 4, "layer2": {"y": 5, "layer1": {"x": 6}}},
    ]

    l3 = Layer3.from_pylist(data)
    flat_table = l3.flattened_table()

    unflat_table = Layer3._unflatten_table(flat_table)

    assert unflat_table.column("z").to_pylist() == [1, 4]

    have = Layer3(table=unflat_table)

    assert have == l3
