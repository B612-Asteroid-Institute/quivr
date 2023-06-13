import io
import os
import sys
import textwrap

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from quivr.attributes import StringAttribute
from quivr.columns import DictionaryColumn, Int8Column, Int64Column, StringColumn
from quivr.concat import concatenate
from quivr.errors import ValidationError
from quivr.tables import Table
from quivr.validators import lt


class Pair(Table):
    x = Int64Column()
    y = Int64Column()


class Wrapper(Table):
    pair = Pair.as_column()
    id = StringColumn()


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
    x = Int64Column()


class Layer2(Table):
    y = Int64Column()
    layer1 = Layer1.as_column()


class Layer3(Table):
    z = Int64Column()
    layer2 = Layer2.as_column()


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


def test_from_kwargs_dictionary_type():
    class SomeTable(Table):
        vals = DictionaryColumn(index_type=pa.int8(), value_type=pa.string())

    have = SomeTable.from_kwargs(vals=["a", "b", "b"])
    assert have.vals[0].as_py() == "a"


def test_from_kwargs_with_missing():
    class SomeTable(Table):
        x = Int64Column(nullable=True)
        y = Int64Column(nullable=False)
        z = Int64Column(nullable=True)

    # Eliding nullable columns is OK
    have = SomeTable.from_kwargs(y=[1, 2, 3])
    assert have.x.null_count == 3
    assert have.y.null_count == 0
    assert have.z.null_count == 3
    np.testing.assert_array_equal(have.y, [1, 2, 3])

    with pytest.raises(ValueError, match="Missing non-nullable column y"):
        have = SomeTable.from_kwargs(x=[1, 2, 3])
    with pytest.raises(ValueError, match="Missing non-nullable column y"):
        have = SomeTable.from_kwargs(z=[1, 2, 3])

    # Eliding nullable columns is OK
    have = SomeTable.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    assert have.x.null_count == 0
    assert have.y.null_count == 0
    assert have.z.null_count == 3
    np.testing.assert_array_equal(have.x, [1, 2, 3])
    np.testing.assert_array_equal(have.y, [4, 5, 6])


def test_from_kwargs_with_missing_as_none():
    class SomeTable(Table):
        x = Int64Column(nullable=True)
        y = Int64Column(nullable=False)
        z = Int64Column(nullable=True)

    # Eliding nullable columns is OK
    have = SomeTable.from_kwargs(x=None, y=[1, 2, 3], z=None)
    assert have.x.null_count == 3
    assert have.y.null_count == 0
    assert have.z.null_count == 3
    np.testing.assert_array_equal(have.y, [1, 2, 3])

    with pytest.raises(ValueError, match="Missing non-nullable column y"):
        have = SomeTable.from_kwargs(x=[1, 2, 3], y=None)
    with pytest.raises(ValueError, match="Missing non-nullable column y"):
        have = SomeTable.from_kwargs(z=[1, 2, 3], y=None)

    # Eliding nullable columns is OK
    have = SomeTable.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], z=None)
    assert have.x.null_count == 0
    assert have.y.null_count == 0
    assert have.z.null_count == 3
    np.testing.assert_array_equal(have.x, [1, 2, 3])
    np.testing.assert_array_equal(have.y, [4, 5, 6])


def test_from_kwargs_raises_mismatched_sizes():
    class SomeTable(Table):
        x = Int64Column()
        y = Int64Column()

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


class TableWithAttributes(Table):
    x = Int64Column()
    y = Int64Column()

    def __init__(self, table: pa.Table, attrib: str):
        self.attrib = attrib
        super().__init__(table)

    def with_table(self, table: pa.Table) -> Self:
        return TableWithAttributes(table, attrib=self.attrib)


class TestTableAttributes:
    def test_from_pydict(self):
        have = TableWithAttributes.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]}, attrib="foo")
        np.testing.assert_array_equal(have.x, [1, 2, 3])
        np.testing.assert_array_equal(have.y, [4, 5, 6])
        assert have.attrib == "foo"

    def test_from_rows(self):
        have = TableWithAttributes.from_rows(
            [{"x": 1, "y": 4}, {"x": 2, "y": 5}, {"x": 3, "y": 6}],
            attrib="foo",
        )
        np.testing.assert_array_equal(have.x, [1, 2, 3])
        np.testing.assert_array_equal(have.y, [4, 5, 6])
        assert have.attrib == "foo"

    def test_from_lists(self):
        have = TableWithAttributes.from_lists([[1, 2, 3], [4, 5, 6]], attrib="foo")
        np.testing.assert_array_equal(have.x, [1, 2, 3])
        np.testing.assert_array_equal(have.y, [4, 5, 6])
        assert have.attrib == "foo"

    def test_from_dataframe(self):
        have = TableWithAttributes.from_dataframe(
            pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
            attrib="foo",
        )
        np.testing.assert_array_equal(have.x, [1, 2, 3])
        np.testing.assert_array_equal(have.y, [4, 5, 6])
        assert have.attrib == "foo"

    def test_from_flat_dataframe(self):
        have = TableWithAttributes.from_flat_dataframe(
            pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
            attrib="foo",
        )
        np.testing.assert_array_equal(have.x, [1, 2, 3])
        np.testing.assert_array_equal(have.y, [4, 5, 6])
        assert have.attrib == "foo"

    def test_from_kwargs(self):
        have = TableWithAttributes.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], attrib="foo")
        np.testing.assert_array_equal(have.x, [1, 2, 3])
        np.testing.assert_array_equal(have.y, [4, 5, 6])
        assert have.attrib == "foo"

    def test_from_arrays(self):
        xs = pa.array([1, 2, 3])
        ys = pa.array([4, 5, 6])
        have = TableWithAttributes.from_arrays([xs, ys], attrib="foo")
        np.testing.assert_array_equal(have.x, [1, 2, 3])
        np.testing.assert_array_equal(have.y, [4, 5, 6])
        assert have.attrib == "foo"

    def test_from_data(self):
        have = TableWithAttributes.from_data(x=[1, 2, 3], y=[4, 5, 6], attrib="foo")
        np.testing.assert_array_equal(have.x, [1, 2, 3])
        np.testing.assert_array_equal(have.y, [4, 5, 6])
        assert have.attrib == "foo"

    def test_from_parquet(self, tmp_path):
        path = os.path.join(tmp_path, "test.parquet")
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df.to_parquet(path)

        have = TableWithAttributes.from_parquet(path, attrib="foo")
        np.testing.assert_array_equal(have.x, [1, 2, 3])
        np.testing.assert_array_equal(have.y, [4, 5, 6])
        assert have.attrib == "foo"

    def test_from_feather(self, tmp_path):
        path = os.path.join(tmp_path, "test.feather")
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df.to_feather(path)

        have = TableWithAttributes.from_feather(path, attrib="foo")
        np.testing.assert_array_equal(have.x, [1, 2, 3])
        np.testing.assert_array_equal(have.y, [4, 5, 6])
        assert have.attrib == "foo"

    def test_from_csv(self, tmp_path):
        path = os.path.join(tmp_path, "test.csv")
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df.to_csv(path)

        have = TableWithAttributes.from_csv(path, attrib="foo")
        np.testing.assert_array_equal(have.x, [1, 2, 3])
        np.testing.assert_array_equal(have.y, [4, 5, 6])
        assert have.attrib == "foo"

    def test_getitem(self):
        have = TableWithAttributes.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], attrib="foo")

        sliced = have[1:]
        np.testing.assert_array_equal(sliced.x, [2, 3])
        np.testing.assert_array_equal(sliced.y, [5, 6])
        assert sliced.attrib == "foo"

        indexed = have[1]
        np.testing.assert_array_equal(indexed.x, [2])
        np.testing.assert_array_equal(indexed.y, [5])
        assert indexed.attrib == "foo"


def test_init_subclass_with_attributes_without_withtable():
    with pytest.raises(TypeError):

        class MyTable(Table):
            x = Int64Column()
            attrib: str

            def __init__(self, table: pa.Table, attrib: str):
                self.attrib = attrib
                super().__init__(table)


def test_empty():
    have = Pair.empty()
    assert len(have.x) == 0
    assert len(have) == 0


class TestValidation:
    def test_int8_bounds(self):
        class MyTable(Table):
            x = Int8Column(validator=lt(10))

        with pytest.raises(ValidationError):
            MyTable.from_data(x=[8, 9, 10], validate=True)

        table = MyTable.from_data(x=[8, 9, 10], validate=False)
        with pytest.raises(ValidationError):
            table.validate()


def test_where_filtering():
    class InnerTable(Table):
        x = Int8Column()

    class OuterTable(Table):
        inner = InnerTable.as_column()
        y = Int8Column()
        label = StringAttribute()

    table = OuterTable.from_data(
        inner=InnerTable.from_data(x=[1, 2, 3]),
        y=[4, 5, 6],
        label="foo",
    )

    have = table.where(pc.field("y") > 4)
    assert len(have) == 2
    assert have.label == "foo"
    np.testing.assert_array_equal(have.y, [5, 6])

    have = table.where(pc.field(("inner", "x")) > 2)
    assert len(have) == 1
    assert have.label == "foo"
    np.testing.assert_array_equal(have.y, [6])


def test_apply_mask_numpy():
    values = Pair.from_data(x=[1, 2, 3], y=[4, 5, 6])

    mask = np.array([True, False, True])
    have = values.apply_mask(mask)
    np.testing.assert_array_equal(have.x, [1, 3])


def test_apply_mask_pylist():
    values = Pair.from_data(x=[1, 2, 3], y=[4, 5, 6])

    mask = [True, False, True]
    have = values.apply_mask(mask)
    np.testing.assert_array_equal(have.x, [1, 3])


def test_apply_mask_pyarrow():
    values = Pair.from_data(x=[1, 2, 3], y=[4, 5, 6])

    mask = pa.array([True, False, True], pa.bool_())
    have = values.apply_mask(mask)
    np.testing.assert_array_equal(have.x, [1, 3])


def test_apply_mask_wrong_size():
    values = Pair.from_data(x=[1, 2, 3], y=[4, 5, 6])

    mask = [True, False]
    with pytest.raises(ValueError):
        values.apply_mask(mask)


def test_apply_mask_pyarrow_with_nulls():
    values = Pair.from_data(x=[1, 2, 3], y=[4, None, 6])

    mask = pa.array([True, False, None], pa.bool_())
    with pytest.raises(ValueError):
        values.apply_mask(mask)
