import io
import os
import sys
import textwrap

if sys.version_info < (3, 11):
    pass
else:
    pass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import quivr as qv
from quivr.validators import lt


class Pair(qv.Table):
    x = qv.Int64Column()
    y = qv.Int64Column()


class Wrapper(qv.Table):
    pair = Pair.as_column()
    id = qv.StringColumn()


def test_table__eq__():
    class A(qv.Table):
        x = qv.Int64Column()
        y = qv.StringAttribute()

    class B(qv.Table):
        z = qv.Int64Column()
        a = A.as_column()
        b = qv.IntAttribute()

    a_table = A.from_kwargs(
        x=[1],
        y="y",
    )
    a_table_same = A.from_kwargs(
        x=[1],
        y="y",
    )
    a_table_attr_diff = A.from_kwargs(
        x=[1],
        y="z",
    )
    a_table_col_diff = A.from_kwargs(
        x=[0],
        y="y",
    )
    a_table_attr_col_diff = A.from_kwargs(
        x=[0],
        y="z",
    )

    assert a_table == a_table
    assert a_table == a_table_same
    assert a_table != a_table_attr_diff
    assert a_table != a_table_col_diff
    assert a_table != a_table_attr_col_diff

    b_table = B.from_kwargs(
        z=[1],
        a=a_table,
        b=1,
    )
    b_table_same = B.from_kwargs(
        z=[1],
        a=a_table,
        b=1,
    )
    b_table_subattr_diff = B.from_kwargs(
        z=[1],
        a=a_table_attr_diff,
        b=1,
    )
    b_table_attr_diff = B.from_kwargs(
        z=[1],
        a=a_table,
        b=2,
    )
    assert b_table == b_table_same
    assert b_table != b_table_subattr_diff
    assert b_table != b_table_attr_diff


def test_table__getitem__():
    # Test indexing with positive and negative integers
    table = Pair.from_kwargs(
        x=[1, 2, 3],
        y=[4, 5, 6],
    )
    assert table[0] == Pair.from_kwargs(x=[1], y=[4])
    assert table[1] == Pair.from_kwargs(x=[2], y=[5])
    assert table[2] == Pair.from_kwargs(x=[3], y=[6])
    assert table[-1] == Pair.from_kwargs(x=[3], y=[6])
    assert table[-2] == Pair.from_kwargs(x=[2], y=[5])
    assert table[-3] == Pair.from_kwargs(x=[1], y=[4])

    # Test indexing with slices
    assert table[0:2] == Pair.from_kwargs(x=[1, 2], y=[4, 5])
    assert table[1:3] == Pair.from_kwargs(x=[2, 3], y=[5, 6])
    assert table[-2:] == Pair.from_kwargs(x=[2, 3], y=[5, 6])
    assert table[:-1] == Pair.from_kwargs(x=[1, 2], y=[4, 5])


def test_table_to_structarray():
    xs = pa.array([1, 2, 3], pa.int64())
    ys = pa.array([4, 5, 6], pa.int64())
    pair = Pair.from_kwargs(x=xs, y=ys)

    want = pa.StructArray.from_arrays([xs, ys], fields=list(Pair.schema))

    have = pair.to_structarray()
    assert have == want


def test_generated_accessors():
    have = Pair.from_kwargs(
        x=[1, 2, 3],
        y=[4, 5, 6],
    )
    np.testing.assert_array_equal(have.x, [1, 2, 3])
    np.testing.assert_array_equal(have.y, [4, 5, 6])


def test_iteration():
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
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
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    assert pair.chunk_counts() == {"x": 1, "y": 1}
    pair = qv.concatenate([pair, pair], defrag=False)
    assert pair.chunk_counts() == {"x": 2, "y": 2}


def test_check_fragmented():
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    assert not pair.fragmented()
    pair = qv.concatenate([pair, pair], defrag=False)
    assert pair.fragmented()


def test_select():
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    have = pair.select("x", 3)
    assert len(have) == 1
    assert have.y[0].as_py() == 6


def test_select_empty():
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    have = pair.select("x", 4)
    assert len(have) == 0


def test_sort_by():
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[5, 1, 2])

    sorted1 = pair.sort_by("y")
    assert sorted1.x[0].as_py() == 2
    assert sorted1.x[1].as_py() == 3
    assert sorted1.x[2].as_py() == 1

    sorted2 = pair.sort_by([("x", "descending")])
    assert sorted2.x[0].as_py() == 3
    assert sorted2.x[1].as_py() == 2
    assert sorted2.x[2].as_py() == 1


def test_to_csv():
    wrapper = Wrapper.from_kwargs(
        id=["1", "2"],
        pair=Pair.from_kwargs(
            x=[1, 3],
            y=[2, 4],
        ),
    )

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
    wrapper = Wrapper.from_kwargs(
        id=["1", "2"],
        pair=Pair.from_kwargs(
            x=[1, 3],
            y=[2, 4],
        ),
    )

    np.testing.assert_array_equal(wrapper.id, ["1", "2"])
    np.testing.assert_array_equal(wrapper.pair.x, [1, 3])
    np.testing.assert_array_equal(wrapper.pair.y, [2, 4])


class Layer1(qv.Table):
    x = qv.Int64Column()


class Layer2(qv.Table):
    y = qv.Int64Column()
    layer1 = Layer1.as_column()


class Layer3(qv.Table):
    z = qv.Int64Column()
    layer2 = Layer2.as_column()


def test_unflatten_table():
    l3 = Layer3.from_kwargs(
        z=[1, 4],
        layer2=Layer2.from_kwargs(
            y=[2, 5],
            layer1=Layer1.from_kwargs(
                x=[3, 6],
            ),
        ),
    )
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
    class SomeTable(qv.Table):
        vals = qv.DictionaryColumn(index_type=pa.int8(), value_type=pa.string())

    have = SomeTable.from_kwargs(vals=["a", "b", "b"])
    assert have.vals[0].as_py() == "a"


def test_from_kwargs_with_missing():
    class SomeTable(qv.Table):
        x = qv.Int64Column(nullable=True)
        y = qv.Int64Column(nullable=False)
        z = qv.Int64Column(nullable=True)

    # Eliding nullable columns is OK
    have = SomeTable.from_kwargs(y=[1, 2, 3])
    assert have.x.null_count == 3
    assert have.y.null_count == 0
    assert have.z.null_count == 3
    np.testing.assert_array_equal(have.y, [1, 2, 3])

    with pytest.raises(qv.InvalidColumnDataError, match="received no data"):
        have = SomeTable.from_kwargs(x=[1, 2, 3])
    with pytest.raises(qv.InvalidColumnDataError, match="received no data"):
        have = SomeTable.from_kwargs(z=[1, 2, 3])

    # Eliding nullable columns is OK
    have = SomeTable.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    assert have.x.null_count == 0
    assert have.y.null_count == 0
    assert have.z.null_count == 3
    np.testing.assert_array_equal(have.x, [1, 2, 3])
    np.testing.assert_array_equal(have.y, [4, 5, 6])


def test_from_kwargs_with_missing_as_none():
    class SomeTable(qv.Table):
        x = qv.Int64Column(nullable=True)
        y = qv.Int64Column(nullable=False)
        z = qv.Int64Column(nullable=True)

    # Eliding nullable columns is OK
    have = SomeTable.from_kwargs(x=None, y=[1, 2, 3], z=None)
    assert have.x.null_count == 3
    assert have.y.null_count == 0
    assert have.z.null_count == 3
    np.testing.assert_array_equal(have.y, [1, 2, 3])

    with pytest.raises(qv.InvalidColumnDataError):
        have = SomeTable.from_kwargs(x=[1, 2, 3], y=None)
    with pytest.raises(qv.InvalidColumnDataError):
        have = SomeTable.from_kwargs(z=[1, 2, 3], y=None)

    # Eliding nullable columns is OK
    have = SomeTable.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], z=None)
    assert have.x.null_count == 0
    assert have.y.null_count == 0
    assert have.z.null_count == 3
    np.testing.assert_array_equal(have.x, [1, 2, 3])
    np.testing.assert_array_equal(have.y, [4, 5, 6])


def test_from_kwargs_raises_mismatched_sizes():
    class SomeTable(qv.Table):
        x = qv.Int64Column()
        y = qv.Int64Column()

    with pytest.raises(qv.InvalidColumnDataError):
        SomeTable.from_kwargs(x=[1, 2, 3], y=[4, 5, 6, 7])


def test_from_kwargs_no_data():
    class NullablePair(qv.Table):
        x = qv.Int64Column(nullable=True)
        y = qv.Int64Column(nullable=True)

    with pytest.raises(ValueError, match="No data provided"):
        NullablePair.from_kwargs()


def test_from_kwargs_missing_nullable_subtable():
    class Pair(qv.Table):
        x = qv.Int64Column()
        y = qv.Int64Column()

    class Wrapper(qv.Table):
        x = qv.Int64Column()
        pairs = Pair.as_column(nullable=True)

    have = Wrapper.from_kwargs(x=[1, 2, 3])
    assert have.x.null_count == 0
    # This isn't exactly what we'd like to see. I'd like it if this
    # could be written more like "assert have.pairs.null_count ==
    # 3". But pairs is a quivr.Table, not a pyarrow.Array; it doesn't
    # have a null count directly. The best we can do is look at the
    # structs inside.
    assert have.pairs.x.null_count == 3
    assert have.pairs.y.null_count == 3

    # This reflects the behavior we'd like to see, but it's not
    # really the way the API works.
    have_sa = have.to_structarray()
    assert have_sa.field("x").null_count == 0
    assert have_sa.field("pairs").null_count == 3

    # Test that we can round-trip this.
    have2 = Wrapper.from_kwargs(
        x=have.x,
        pairs=have.pairs,
    )
    assert have2.x.null_count == 0
    assert have2.pairs.x.null_count == 3
    assert have2.pairs.y.null_count == 3


def test_set_missing_column_with_nullable_subtable():
    class Pair(qv.Table):
        x = qv.Int64Column()
        y = qv.Int64Column()

    class Wrapper(qv.Table):
        x = qv.Int64Column()
        pairs = Pair.as_column(nullable=True)

    have = Wrapper.from_kwargs(x=[1, 2, 3])
    have.pairs = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    assert have.x.null_count == 0
    assert have.pairs.x.null_count == 0
    assert have.pairs.y.null_count == 0


class TableWithAttributes(qv.Table):
    x = qv.Int64Column()
    y = qv.Int64Column()
    attrib = qv.StringAttribute()


class TestTableAttributes:
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


def test_empty():
    have = Pair.empty()
    assert len(have.x) == 0
    assert len(have) == 0


class TestValidation:
    def test_int8_bounds(self):
        class MyTable(qv.Table):
            x = qv.Int8Column(validator=lt(10))

        with pytest.raises(qv.ValidationError):
            MyTable.from_kwargs(x=[8, 9, 10])

        table = MyTable.from_kwargs(x=[8, 9, 10], validate=False)
        with pytest.raises(qv.ValidationError):
            table.validate()


class TestTableEqualityBenchmarks:
    @pytest.mark.benchmark(group="table-equality")
    def test_identical_small_tables(self, benchmark):
        table1 = TableWithAttributes.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], attrib="foo")
        table2 = TableWithAttributes.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], attrib="foo")
        benchmark(table1.__eq__, table2)

    @pytest.mark.benchmark(group="table-equality")
    def test_identical_large_tables(self, benchmark):
        table1 = TableWithAttributes.from_kwargs(x=np.arange(10000), y=np.arange(10000), attrib="foo")
        table2 = TableWithAttributes.from_kwargs(x=np.arange(10000), y=np.arange(10000), attrib="foo")
        benchmark(table1.__eq__, table2)

    @pytest.mark.benchmark(group="table-equality")
    def test_different_small_tables(self, benchmark):
        table1 = TableWithAttributes.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], attrib="foo")
        table2 = TableWithAttributes.from_kwargs(x=[1, 2, 3], y=[4, 5, 7], attrib="foo")
        benchmark(table1.__eq__, table2)

    @pytest.mark.benchmark(group="table-equality")
    def test_different_large_tables(self, benchmark):
        table1 = TableWithAttributes.from_kwargs(x=np.arange(10000), y=np.arange(10000), attrib="foo")
        table2 = TableWithAttributes.from_kwargs(x=np.arange(10000), y=np.arange(10000) + 1, attrib="foo")
        benchmark(table1.__eq__, table2)

    @pytest.mark.benchmark(group="table-equality")
    def test_small_tables_different_attributes(self, benchmark):
        table1 = TableWithAttributes.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], attrib="foo")
        table2 = TableWithAttributes.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], attrib="bar")
        benchmark(table1.__eq__, table2)

    @pytest.mark.benchmark(group="table-equality")
    def test_large_tables_different_attributes(self, benchmark):
        table1 = TableWithAttributes.from_kwargs(x=np.arange(10000), y=np.arange(10000), attrib="foo")
        table2 = TableWithAttributes.from_kwargs(x=np.arange(10000), y=np.arange(10000), attrib="bar")
        benchmark(table1.__eq__, table2)


def test_where_filtering():
    class InnerTable(qv.Table):
        x = qv.Int8Column()

    class OuterTable(qv.Table):
        inner = InnerTable.as_column()
        y = qv.Int8Column()
        label = qv.StringAttribute()

    table = OuterTable.from_kwargs(
        inner=InnerTable.from_kwargs(x=[1, 2, 3]),
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
    values = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])

    mask = np.array([True, False, True])
    have = values.apply_mask(mask)
    np.testing.assert_array_equal(have.x, [1, 3])


def test_apply_mask_pylist():
    values = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])

    mask = [True, False, True]
    have = values.apply_mask(mask)
    np.testing.assert_array_equal(have.x, [1, 3])


def test_apply_mask_pyarrow():
    values = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])

    mask = pa.array([True, False, True], pa.bool_())
    have = values.apply_mask(mask)
    np.testing.assert_array_equal(have.x, [1, 3])


def test_apply_mask_wrong_size():
    values = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])

    mask = [True, False]
    with pytest.raises(ValueError):
        values.apply_mask(mask)


def test_apply_mask_pyarrow_with_nulls():
    class NullablePair(qv.Table):
        x = qv.Int8Column(nullable=True)
        y = qv.Int8Column(nullable=True)

    values = NullablePair.from_kwargs(x=[1, 2, 3], y=[4, None, 6])

    mask = pa.array([True, False, None], pa.bool_())
    with pytest.raises(ValueError):
        values.apply_mask(mask)


def test_from_pyarrow_table():
    table = pa.table({"x": [1, 2, 3], "y": [4, 5, 6]}, schema=Pair.schema)
    have = Pair.from_pyarrow(table)
    assert isinstance(have, Pair)
    assert have.x.equals(pa.array([1, 2, 3], pa.int64()))
    assert have.y.equals(pa.array([4, 5, 6], pa.int64()))


def test_from_pyarrow_table_missing_column():
    table = pa.table({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        Pair.from_pyarrow(table)


def test_from_pyarrow_table_wrong_type():
    table = pa.table({"x": ["a", "b", "c"], "y": [4, 5, 6]})
    with pytest.raises(pa.ArrowInvalid):
        Pair.from_pyarrow(table)


def test_from_pyarrow_int_type_conversions():
    class Int8Pair(qv.Table):
        x = qv.Int8Column()
        y = qv.Int8Column()

    table = pa.table({"x": pa.array([1, 2, 3], pa.int32()), "y": pa.array([4, 5, 6], pa.int64())})
    have = Int8Pair.from_pyarrow(table)
    assert isinstance(have, Int8Pair)
    assert have.x.equals(pa.array([1, 2, 3], pa.int8()))
    assert have.y.equals(pa.array([4, 5, 6], pa.int8()))


def test_from_pyarrow_int_overflow():
    class Int8Pair(qv.Table):
        x = qv.Int8Column()
        y = qv.Int8Column()

    table = pa.table({"x": pa.array([1, 2, 3], pa.int32()), "y": pa.array([4, 5, 1000], pa.int64())})
    with pytest.raises(ValueError):
        Int8Pair.from_pyarrow(table)


def test_from_pyarrow_empty_table():
    table = pa.table({"x": pa.array([], pa.int64()), "y": pa.array([], pa.int64())})
    have = Pair.from_pyarrow(table)
    assert isinstance(have, Pair)
    assert have.x.equals(pa.array([], pa.int64()))
    assert have.y.equals(pa.array([], pa.int64()))


def test_from_pyarrow_nested_table():
    table = pa.table(
        {"pair": [{"x": 1, "y": 4}, {"x": 2, "y": 5}, {"x": 3, "y": 6}], "id": ["a", "b", "c"]},
        schema=Wrapper.schema,
    )

    have = Wrapper.from_pyarrow(table)
    assert isinstance(have, Wrapper)
    assert have.id.equals(pa.array(["a", "b", "c"], pa.string()))

    assert isinstance(have.pair, Pair)
    assert have.pair.x.equals(pa.array([1, 2, 3], pa.int64()))
    assert have.pair.y.equals(pa.array([4, 5, 6], pa.int64()))


def test_from_pyarrow_missing_attribute():
    table = pa.table({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(AttributeError):
        TableWithAttributes.from_pyarrow(table)


def test_from_pyarrow_explicit_attribute():
    table = pa.table({"x": [1, 2, 3], "y": [4, 5, 6]})
    have = TableWithAttributes.from_pyarrow(table, attrib="foo")
    assert have.attrib == "foo"


def test_from_pyarrow_preserves_attributes():
    table = pa.table(
        {"x": [1, 2, 3], "y": [4, 5, 6]},
        schema=pa.schema(
            [
                pa.field("x", pa.int64(), nullable=False),
                pa.field("y", pa.int64(), nullable=False),
            ],
            metadata={"attrib": "bar"},
        ),
    )
    have = TableWithAttributes.from_pyarrow(table)
    assert have.x.equals(pa.array([1, 2, 3], pa.int64()))
    assert have.y.equals(pa.array([4, 5, 6], pa.int64()))
    assert have.attrib == "bar"


def test_from_pyarrow_preserves_nested_attributes():
    class NestedAttributeWrapper(qv.Table):
        inner = TableWithAttributes.as_column()
        name = qv.StringAttribute()

    table = pa.table(
        {
            "inner": [{"x": 1, "y": 4}, {"x": 2, "y": 5}, {"x": 3, "y": 6}],
        },
        schema=pa.schema(
            [
                pa.field(
                    "inner",
                    pa.struct(
                        [
                            pa.field("x", pa.int64(), nullable=False),
                            pa.field("y", pa.int64(), nullable=False),
                        ]
                    ),
                    nullable=False,
                ),
            ],
            metadata={"name": "foo", "inner.attrib": "bar"},
        ),
    )
    have = NestedAttributeWrapper.from_pyarrow(table)
    assert have.name == "foo"
    assert have.inner.attrib == "bar"
    assert have.inner.x.equals(pa.array([1, 2, 3], pa.int64()))
    assert have.inner.y.equals(pa.array([4, 5, 6], pa.int64()))


def test_no_forbidden_column_names():
    with pytest.raises(AttributeError):

        class T1(qv.Table):
            schema = qv.StringColumn()

    with pytest.raises(AttributeError):

        class T2(qv.Table):
            table = qv.StringColumn()

    with pytest.raises(AttributeError):

        class T3(qv.Table):
            _quivr_subtables = qv.StringColumn()

    with pytest.raises(AttributeError):

        class T4(qv.Table):
            _quivr_attributes = qv.StringColumn()

    with pytest.raises(AttributeError):

        class T5(qv.Table):
            _column_validators = qv.StringColumn()


def test_set_column():
    t = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    t2 = t.set_column("x", [7, 8, 9])

    # original should be unchanged
    assert t.x.equals(pa.array([1, 2, 3], pa.int64()))
    # new table should have new column
    assert t2.x.equals(pa.array([7, 8, 9], pa.int64()))


def test_set_column_nested():
    w = Wrapper.from_kwargs(id=["a", "b", "c"], pair=Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6]))
    w2 = w.set_column("pair.x", [7, 8, 9])

    # original should be unchanged
    assert w.pair.x.equals(pa.array([1, 2, 3], pa.int64()))
    # new table should have new column
    assert w2.pair.x.equals(pa.array([7, 8, 9], pa.int64()))


def test_set_column_nested_doubly():
    class DoublyNested(qv.Table):
        inner = Wrapper.as_column()

    dn = DoublyNested.from_kwargs(
        inner=Wrapper.from_kwargs(id=["a", "b", "c"], pair=Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6]))
    )
    dn2 = dn.set_column("inner.pair.x", [7, 8, 9])

    # original should be unchanged
    assert dn.inner.pair.x.equals(pa.array([1, 2, 3], pa.int64()))
    # new table should have new column
    assert dn2.inner.pair.x.equals(pa.array([7, 8, 9], pa.int64()))

    dn3 = dn.set_column("inner.pair", Pair.from_kwargs(x=[7, 8, 9], y=[10, 11, 12]))
    assert dn3.inner.pair.x.equals(pa.array([7, 8, 9], pa.int64()))
    assert dn3.inner.pair.y.equals(pa.array([10, 11, 12], pa.int64()))


def test_set_column_subtable():
    w = Wrapper.from_kwargs(id=["a", "b", "c"], pair=Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6]))
    w2 = w.set_column("pair", Pair.from_kwargs(x=[7, 8, 9], y=[10, 11, 12]))

    # original should be unchanged
    assert w.pair.x.equals(pa.array([1, 2, 3], pa.int64()))
    assert w.pair.y.equals(pa.array([4, 5, 6], pa.int64()))

    # new table should have new column
    assert w2.pair.x.equals(pa.array([7, 8, 9], pa.int64()))
    assert w2.pair.y.equals(pa.array([10, 11, 12], pa.int64()))


def test_set_column_null():
    class PairWithNulls(qv.Table):
        x = qv.Int64Column(nullable=True)
        y = qv.Int64Column(nullable=True)

    t = PairWithNulls.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    t2 = t.set_column("x", pa.nulls(3, pa.int64()))

    # original should be unchanged
    assert t.x.equals(pa.array([1, 2, 3], pa.int64()))
    # new table should have new column
    assert t2.x.equals(pa.array([None, None, None], pa.int64()))
