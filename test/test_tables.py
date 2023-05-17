import io
import textwrap

import numpy as np
import pyarrow as pa
import pytest

from quivr.concat import concatenate
from quivr.fields import Int64Field, StringField
from quivr.tables import Table


class Pair(Table):
    x = Int64Field()
    y = Int64Field()


class Wrapper(Table):
    pair = Pair.as_field()
    id = StringField()


def test_create_from_arrays():
    xs = pa.array([1, 2, 3], pa.int64())
    ys = pa.array([4, 5, 6], pa.int64())
    have = Pair.from_arrays([xs, ys])
    assert len(have) == 3
    np.testing.assert_array_equal(have.column("x"), [1, 2, 3])
    np.testing.assert_array_equal(have.column("y"), [4, 5, 6])


def test_create_wrapped_from_arrays():
    xs = pa.array([1, 2, 3], pa.int64())
    ys = pa.array([4, 5, 6], pa.int64())
    pairs = pa.StructArray.from_arrays([xs, ys], fields=list(Pair.schema))
    ids = pa.array(["v1", "v2", "v3"], pa.string())

    have = Wrapper.from_arrays([pairs, ids])
    assert len(have) == 3
    np.testing.assert_array_equal(have.column("id"), ["v1", "v2", "v3"])


def test_create_from_pydict():
    have = Pair.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert len(have) == 3
    np.testing.assert_array_equal(have.column("x"), [1, 2, 3])
    np.testing.assert_array_equal(have.column("y"), [4, 5, 6])


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
    np.testing.assert_array_equal(have.column("id"), ["v1", "v2", "v3"])


def test_generated_accessors():
    have = Pair.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    np.testing.assert_array_equal(have.x, [1, 2, 3])
    np.testing.assert_array_equal(have.y, [4, 5, 6])


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


def test_to_csv():
    data = [
        {"id": "1", "pair": {"x": 1, "y": 2}},
        {"id": "2", "pair": {"x": 3, "y": 4}},
    ]
    wrapper = Wrapper.from_rows(data)

    buf = io.BytesIO()
    wrapper.to_csv(buf)

    buf.seek(0)
    have = buf.read().decode("utf8")
    expected = textwrap.dedent(
        """
    "pair.x","pair.y","id"
    1,2,"1"
    3,4,"2"
    """
    )
    assert have.strip() == expected.strip()


def test_from_csv():
    csv = io.BytesIO(
        textwrap.dedent(
            """
    "pair.x","pair.y","id"
    1,2,"1"
    3,4,"2"
    """
        ).encode("utf8")
    )

    wrapper = Wrapper.from_csv(csv)
    np.testing.assert_array_equal(wrapper.id, ["1", "2"])
    np.testing.assert_array_equal(wrapper.pair.x, [1, 3])
    np.testing.assert_array_equal(wrapper.pair.y, [2, 4])


def test_from_pylist():
    data = [
        {"id": "1", "pair": {"x": 1, "y": 2}},
        {"id": "2", "pair": {"x": 3, "y": 4}},
    ]
    wrapper = Wrapper.from_rows(data)

    np.testing.assert_array_equal(wrapper.id, ["1", "2"])
    np.testing.assert_array_equal(wrapper.pair.x, [1, 3])
    np.testing.assert_array_equal(wrapper.pair.y, [2, 4])


class Layer1(Table):
    x = Int64Field()


class Layer2(Table):
    y = Int64Field()
    layer1 = Layer1.as_field()


class Layer3(Table):
    z = Int64Field()
    layer2 = Layer2.as_field()


def test_unflatten_table():
    data = [
        {"z": 1, "layer2": {"y": 2, "layer1": {"x": 3}}},
        {"z": 4, "layer2": {"y": 5, "layer1": {"x": 6}}},
    ]

    l3 = Layer3.from_rows(data)
    flat_table = l3.flattened_table()

    unflat_table = Layer3._unflatten_table(flat_table)

    np.testing.assert_array_equal(unflat_table.column("z"), [1, 4])

    have = Layer3(table=unflat_table)

    assert have == l3


def test_from_kwargs():
    l1 = Layer1.from_kwargs(x=[1, 2, 3])
    np.testing.assert_array_equal(l1.x, [1, 2, 3])

    l2 = Layer2.from_kwargs(y=[4, 5, 6], layer1=l1)
    np.testing.assert_array_equal(l2.y, [4, 5, 6])
    np.testing.assert_array_equal(l2.layer1.x, [1, 2, 3])

    l3 = Layer3.from_kwargs(z=[7, 8, 9], layer2=l2)
    np.testing.assert_array_equal(l3.z, [7, 8, 9])
    np.testing.assert_array_equal(l3.layer2.y, [4, 5, 6])
    np.testing.assert_array_equal(l3.layer2.layer1.x, [1, 2, 3])


def test_from_kwargs_with_missing():
    class SomeTable(Table):
        x = Int64Field(nullable=True)
        y = Int64Field(nullable=False)
        z = Int64Field(nullable=True)

    # Eliding nullable fields is OK
    have = SomeTable.from_kwargs(y=[1, 2, 3])
    assert have.x.null_count == 3
    assert have.y.null_count == 0
    assert have.z.null_count == 3
    np.testing.assert_array_equal(have.y, [1, 2, 3])

    with pytest.raises(ValueError, match="Missing non-nullable column y"):
        have = SomeTable.from_kwargs(x=[1, 2, 3])
    with pytest.raises(ValueError, match="Missing non-nullable column y"):
        have = SomeTable.from_kwargs(z=[1, 2, 3])

    # Eliding nullable fields is OK
    have = SomeTable.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    assert have.x.null_count == 0
    assert have.y.null_count == 0
    assert have.z.null_count == 3
    np.testing.assert_array_equal(have.x, [1, 2, 3])
    np.testing.assert_array_equal(have.y, [4, 5, 6])


def test_from_kwargs_with_missing_as_none():
    class SomeTable(Table):
        x = Int64Field(nullable=True)
        y = Int64Field(nullable=False)
        z = Int64Field(nullable=True)

    # Eliding nullable fields is OK
    have = SomeTable.from_kwargs(x=None, y=[1, 2, 3], z=None)
    assert have.x.null_count == 3
    assert have.y.null_count == 0
    assert have.z.null_count == 3
    np.testing.assert_array_equal(have.y, [1, 2, 3])

    with pytest.raises(ValueError, match="Missing non-nullable column y"):
        have = SomeTable.from_kwargs(x=[1, 2, 3], y=None)
    with pytest.raises(ValueError, match="Missing non-nullable column y"):
        have = SomeTable.from_kwargs(z=[1, 2, 3], y=None)

    # Eliding nullable fields is OK
    have = SomeTable.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], z=None)
    assert have.x.null_count == 0
    assert have.y.null_count == 0
    assert have.z.null_count == 3
    np.testing.assert_array_equal(have.x, [1, 2, 3])
    np.testing.assert_array_equal(have.y, [4, 5, 6])


def test_from_kwargs_raises_mismatched_sizes():
    class SomeTable(Table):
        x = Int64Field()
        y = Int64Field()

    with pytest.raises(ValueError, match=r"Column y has wrong length 4 \(first column has length 3\)"):
        SomeTable.from_kwargs(x=[1, 2, 3], y=[4, 5, 6, 7])


def test_from_kwargs_no_data():
    with pytest.raises(ValueError, match="No data provided"):
        Pair.from_kwargs()


def test_from_data_using_kwargs():
    have = Pair.from_data(x=[1, 2, 3], y=[4, 5, 6])
    np.testing.assert_array_equal(have.x, [1, 2, 3])
    np.testing.assert_array_equal(have.y, [4, 5, 6])

    # Change the ordering
    have = Pair.from_data(y=[4, 5, 6], x=[1, 2, 3])
    np.testing.assert_array_equal(have.x, [1, 2, 3])
    np.testing.assert_array_equal(have.y, [4, 5, 6])

    # Refer to a nested value
    pair = have
    wrapper = Wrapper.from_data(id=["1", "2", "3"], pair=pair)
    np.testing.assert_array_equal(wrapper.id, ["1", "2", "3"])
    np.testing.assert_array_equal(wrapper.pair.x, [1, 2, 3])


def test_from_data_using_positional_list():
    have = Pair.from_data([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(have.x, [1, 2, 3])
    np.testing.assert_array_equal(have.y, [4, 5, 6])


def test_from_data_using_positional_dict():
    have = Pair.from_data({"x": [1, 2, 3], "y": [4, 5, 6]})
    np.testing.assert_array_equal(have.x, [1, 2, 3])
    np.testing.assert_array_equal(have.y, [4, 5, 6])
