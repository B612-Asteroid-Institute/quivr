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
from quivr.validators import and_, gt, lt


class Pair(qv.Table):
    x = qv.Int64Column()
    y = qv.Int64Column()


class Wrapper(qv.Table):
    pair = Pair.as_column()
    id = qv.StringColumn()


class OmniTable(qv.Table):
    """A Table with every type of column"""

    i64 = qv.Int64Column()
    i32 = qv.Int32Column()
    i16 = qv.Int16Column()
    i8 = qv.Int8Column()
    u64 = qv.UInt64Column()
    u32 = qv.UInt32Column()
    u16 = qv.UInt16Column()
    u8 = qv.UInt8Column()
    f64 = qv.Float64Column()
    f32 = qv.Float32Column()
    bool = qv.BooleanColumn()
    string = qv.StringColumn()
    binary = qv.BinaryColumn()
    large_string = qv.LargeStringColumn()
    large_binary = qv.LargeBinaryColumn()
    fixed_binary = qv.FixedSizeBinaryColumn(byte_width=8)
    date32 = qv.Date32Column()
    date64 = qv.Date64Column()
    timestamp = qv.TimestampColumn(unit="s")
    time32 = qv.Time32Column(unit="s")
    time64 = qv.Time64Column(unit="us")
    duration = qv.DurationColumn(unit="s")
    mdni = qv.MonthDayNanoIntervalColumn()
    decimal128 = qv.Decimal128Column(precision=38, scale=9)
    decimal256 = qv.Decimal256Column(precision=76, scale=18)
    nulls = qv.NullColumn()

    list_i64 = qv.ListColumn(pa.int64())
    fixed_list_i64 = qv.FixedSizeListColumn(pa.int64(), list_size=2)
    large_list_i64 = qv.LargeListColumn(pa.int64())

    map_i64_i64 = qv.MapColumn(pa.int64(), pa.int64())

    dict_i64 = qv.DictionaryColumn(pa.int64(), pa.int64())

    subtable = Pair.as_column()

    @classmethod
    def create(cls, length: int) -> "OmniTable":
        """
        Create a new instance of the OmniTable. Values are increasing sequences up to length.
        """
        return cls.from_kwargs(
            i64=np.arange(length, dtype=np.int64),
            i32=np.arange(length, dtype=np.int32),
            i16=np.arange(length, dtype=np.int16),
            i8=np.arange(length, dtype=np.int8),
            u64=np.arange(length, dtype=np.uint64),
            u32=np.arange(length, dtype=np.uint32),
            u16=np.arange(length, dtype=np.uint16),
            u8=np.arange(length, dtype=np.uint8),
            f64=np.arange(length, dtype=np.float64),
            f32=np.arange(length, dtype=np.float32),
            bool=(np.arange(length) % 2 == 0),
            string=[f"string{i}" for i in range(length)],
            binary=[f"binary{i}".encode("utf-8") for i in range(length)],
            large_string=[f"large_string{i}" for i in range(length)],
            large_binary=[f"large_binary{i}".encode("utf-8") for i in range(length)],
            fixed_binary=["{:08}".format(i).encode("utf-8") for i in range(length)],
            date32=pa.array(np.arange(length, dtype=np.int32), type=pa.date32()),
            date64=pa.array(np.arange(length), type=pa.date64()),
            timestamp=pa.array(np.arange(length), type=pa.timestamp("s")),
            time32=pa.array(np.arange(length, dtype=np.int32), type=pa.time32("s")),
            time64=pa.array(np.arange(length), type=pa.time64("us")),
            duration=pa.array(np.arange(length), type=pa.duration("s")),
            mdni=pa.array([(1, 1, i) for i in range(length)], type=pa.month_day_nano_interval()),
            decimal128=pa.array(list(range(length)), type=pa.decimal128(38, 9)),
            decimal256=pa.array(list(range(length)), type=pa.decimal256(76, 18)),
            nulls=np.full(length, None),
            list_i64=[np.arange(3) for _ in range(length)],
            fixed_list_i64=[np.arange(2) for _ in range(length)],
            large_list_i64=[np.arange(3) for _ in range(length)],
            map_i64_i64=[{i: i} for i in range(length)],
            dict_i64=list(range(length)),
            subtable=Pair.from_kwargs(
                x=np.arange(length),
                y=np.arange(length),
            ),
        )


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


def test_select_nested():
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    wrapper = Wrapper.from_kwargs(
        id=["1", "2", "3"],
        pair=pair,
    )
    have = wrapper.select("pair.x", 3)
    assert len(have) == 1
    assert have.id[0].as_py() == "3"
    assert have.pair.y[0].as_py() == 6


def test_select_nested_doubly():
    class DoublyNested(qv.Table):
        inner = Wrapper.as_column()

    dn = DoublyNested.from_kwargs(
        inner=Wrapper.from_kwargs(id=["a", "b", "c"], pair=Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6]))
    )
    have = dn.select("inner.pair.x", 3)
    assert len(have) == 1
    assert have.inner.id[0].as_py() == "c"
    assert have.inner.pair.y[0].as_py() == 6


def test_select_attributes():
    class PairWithAttributes(qv.Table):
        x = qv.Int64Column()
        y = qv.Int64Column()
        label = qv.StringAttribute()

    pair = PairWithAttributes.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], label="foo")
    have = pair.select("x", 3)
    assert len(have) == 1
    assert have.y[0].as_py() == 6
    assert have.label == "foo"


def test_select_nested_attributes():
    class PairWithAttributes(qv.Table):
        x = qv.Int64Column()
        y = qv.Int64Column()
        label = qv.StringAttribute()

    class WrapperWithAttributes(qv.Table):
        pair = PairWithAttributes.as_column()
        id = qv.StringColumn()
        label = qv.StringAttribute()

    pair = PairWithAttributes.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], label="foo")
    wrapper = WrapperWithAttributes.from_kwargs(
        id=["1", "2", "3"],
        pair=pair,
        label="bar",
    )
    have = wrapper.select("pair.x", 3)
    assert len(have) == 1
    assert have.id[0].as_py() == "3"
    assert have.pair.y[0].as_py() == 6
    assert have.label == "bar"
    assert have.pair.label == "foo"


def test_select_nested_doubly_attributes():
    class PairWithAttributes(qv.Table):
        x = qv.Int64Column()
        y = qv.Int64Column()
        label = qv.StringAttribute()

    class WrapperWithAttributes(qv.Table):
        pair = PairWithAttributes.as_column()
        id = qv.StringColumn()
        label = qv.StringAttribute()

    class DoublyNestedWithAttributes(qv.Table):
        inner = WrapperWithAttributes.as_column()
        label = qv.StringAttribute()

    dn = DoublyNestedWithAttributes.from_kwargs(
        inner=WrapperWithAttributes.from_kwargs(
            id=["a", "b", "c"],
            pair=PairWithAttributes.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], label="foo"),
            label="bar",
        ),
        label="baz",
    )

    have = dn.select("inner.pair.x", 3)
    assert len(have) == 1
    assert have.inner.id[0].as_py() == "c"
    assert have.inner.pair.y[0].as_py() == 6
    assert have.inner.label == "bar"
    assert have.inner.pair.label == "foo"
    assert have.label == "baz"


def test_select_invalid_value():
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    have = pair.select("x", 4)
    assert len(have) == 0


def test_select_nested_invalid_value():
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    wrapper = Wrapper.from_kwargs(
        id=["1", "2", "3"],
        pair=pair,
    )
    have = wrapper.select("pair.x", 4)
    assert len(have) == 0

    have = wrapper.select("id", "4")
    assert len(have) == 0


def test_select_nested_doubly_invalid_value():
    class DoublyNested(qv.Table):
        id = qv.StringColumn()
        inner = Wrapper.as_column()

    dn = DoublyNested.from_kwargs(
        id=["1", "2", "3"],
        inner=Wrapper.from_kwargs(id=["1", "2", "3"], pair=Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])),
    )
    have = dn.select("inner.pair.x", 4)
    assert len(have) == 0

    have = dn.select("inner.id", "4")
    assert len(have) == 0

    have = dn.select("id", "d")
    assert len(have) == 0


def test_sort_by():
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[5, 1, 2])

    sorted1 = pair.sort_by([("y", "ascending")])
    assert sorted1.x.to_pylist() == [2, 3, 1]
    assert sorted1.y.to_pylist() == [1, 2, 5]

    sorted2 = pair.sort_by([("y", "descending")])
    assert sorted2.x.to_pylist() == [1, 3, 2]
    assert sorted2.y.to_pylist() == [5, 2, 1]

    sorted3 = pair.sort_by([("x", "ascending")])
    assert sorted3.x.to_pylist() == [1, 2, 3]
    assert sorted3.y.to_pylist() == [5, 1, 2]

    sorted4 = pair.sort_by([("x", "descending")])
    assert sorted4.x.to_pylist() == [3, 2, 1]
    assert sorted4.y.to_pylist() == [2, 1, 5]


def test_sort_by_with_column_names():
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[5, 1, 2])

    sorted1 = pair.sort_by("y")
    assert sorted1.x.to_pylist() == [2, 3, 1]
    assert sorted1.y.to_pylist() == [1, 2, 5]

    sorted2 = pair.sort_by("x")
    assert sorted2.x.to_pylist() == [1, 2, 3]
    assert sorted2.y.to_pylist() == [5, 1, 2]


def test_sort_by_multiple_columns():
    pair = Pair.from_kwargs(x=[2, 1, 2, 3], y=[6, 4, 5, 6])

    # Column order: y first then x (alternating ascending/descending)
    sorted1 = pair.sort_by([("y", "ascending"), ("x", "ascending")])
    assert sorted1.x.to_pylist() == [1, 2, 2, 3]
    assert sorted1.y.to_pylist() == [4, 5, 6, 6]

    sorted2 = pair.sort_by([("y", "ascending"), ("x", "descending")])
    assert sorted2.x.to_pylist() == [1, 2, 3, 2]
    assert sorted2.y.to_pylist() == [4, 5, 6, 6]

    sorted3 = pair.sort_by([("y", "descending"), ("x", "ascending")])
    assert sorted3.x.to_pylist() == [2, 3, 2, 1]
    assert sorted3.y.to_pylist() == [6, 6, 5, 4]

    sorted4 = pair.sort_by([("y", "descending"), ("x", "descending")])
    assert sorted4.x.to_pylist() == [3, 2, 2, 1]
    assert sorted4.y.to_pylist() == [6, 6, 5, 4]

    # Column order: x first then y (alternating ascending/descending)
    sorted5 = pair.sort_by([("x", "ascending"), ("y", "ascending")])
    assert sorted5.x.to_pylist() == [1, 2, 2, 3]
    assert sorted5.y.to_pylist() == [4, 5, 6, 6]

    sorted6 = pair.sort_by([("x", "ascending"), ("y", "descending")])
    assert sorted6.x.to_pylist() == [1, 2, 2, 3]
    assert sorted6.y.to_pylist() == [4, 6, 5, 6]

    sorted7 = pair.sort_by([("x", "descending"), ("y", "ascending")])
    assert sorted7.x.to_pylist() == [3, 2, 2, 1]
    assert sorted7.y.to_pylist() == [6, 5, 6, 4]

    sorted8 = pair.sort_by([("x", "descending"), ("y", "descending")])
    assert sorted8.x.to_pylist() == [3, 2, 2, 1]
    assert sorted8.y.to_pylist() == [6, 6, 5, 4]


def test_sort_by_multiple_columns_with_column_names():
    pair = Pair.from_kwargs(x=[2, 1, 2, 3], y=[6, 4, 5, 6])

    # Column order: y first then x
    sorted1 = pair.sort_by(["y", "x"])
    assert sorted1.x.to_pylist() == [1, 2, 2, 3]
    assert sorted1.y.to_pylist() == [4, 5, 6, 6]

    sorted2 = pair.sort_by(["x", "y"])
    assert sorted2.x.to_pylist() == [1, 2, 2, 3]
    assert sorted2.y.to_pylist() == [4, 5, 6, 6]


def test_sort_by_nested():
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[5, 1, 2])
    wrapper = Wrapper.from_kwargs(
        id=["1", "2", "3"],
        pair=pair,
    )

    sorted1 = wrapper.sort_by("pair.y")
    assert sorted1.id.to_pylist() == ["2", "3", "1"]
    assert sorted1.pair.x.to_pylist() == [2, 3, 1]
    assert sorted1.pair.y.to_pylist() == [1, 2, 5]

    sorted2 = wrapper.sort_by([("pair.x", "descending")])
    assert sorted2.id.to_pylist() == ["3", "2", "1"]
    assert sorted2.pair.x.to_pylist() == [3, 2, 1]
    assert sorted2.pair.y.to_pylist() == [2, 1, 5]

    # Sort by id (same as original)
    sorted3 = wrapper.sort_by([("id", "ascending")])
    assert sorted3.id.to_pylist() == ["1", "2", "3"]
    assert sorted3.pair.x.to_pylist() == [1, 2, 3]
    assert sorted3.pair.y.to_pylist() == [5, 1, 2]

    sorted4 = wrapper.sort_by([("id", "descending")])
    assert sorted4.id.to_pylist() == ["3", "2", "1"]
    assert sorted4.pair.x.to_pylist() == [3, 2, 1]
    assert sorted4.pair.y.to_pylist() == [2, 1, 5]


def test_sort_by_nested_multiple_columns():
    pair = Pair.from_kwargs(x=[2, 1, 2, 3], y=[6, 4, 5, 6])
    wrapper = Wrapper.from_kwargs(
        id=["1", "2", "3", "4"],
        pair=pair,
    )

    # Column order: y first then x
    sorted1 = wrapper.sort_by([("pair.y", "ascending"), ("pair.x", "ascending")])
    assert sorted1.id.to_pylist() == ["2", "3", "1", "4"]
    assert sorted1.pair.x.to_pylist() == [1, 2, 2, 3]
    assert sorted1.pair.y.to_pylist() == [4, 5, 6, 6]

    sorted2 = wrapper.sort_by([("pair.y", "descending"), ("pair.x", "descending")])
    assert sorted2.id.to_pylist() == ["4", "1", "3", "2"]
    assert sorted2.pair.x.to_pylist() == [3, 2, 2, 1]
    assert sorted2.pair.y.to_pylist() == [6, 6, 5, 4]

    # Column order: x first then y
    sorted3 = wrapper.sort_by([("pair.x", "ascending"), ("pair.y", "ascending")])
    assert sorted3.id.to_pylist() == ["2", "3", "1", "4"]
    assert sorted3.pair.x.to_pylist() == [1, 2, 2, 3]
    assert sorted3.pair.y.to_pylist() == [4, 5, 6, 6]

    sorted4 = wrapper.sort_by([("pair.x", "descending"), ("pair.y", "descending")])
    assert sorted4.id.to_pylist() == ["4", "1", "3", "2"]
    assert sorted4.pair.x.to_pylist() == [3, 2, 2, 1]
    assert sorted4.pair.y.to_pylist() == [6, 6, 5, 4]

    # Column order: id then x
    sorted5 = wrapper.sort_by([("pair.x", "ascending"), ("id", "ascending")])
    assert sorted5.id.to_pylist() == ["2", "1", "3", "4"]
    assert sorted5.pair.x.to_pylist() == [1, 2, 2, 3]
    assert sorted5.pair.y.to_pylist() == [4, 6, 5, 6]

    sorted6 = wrapper.sort_by([("pair.x", "descending"), ("id", "ascending")])
    assert sorted6.id.to_pylist() == ["4", "1", "3", "2"]
    assert sorted6.pair.x.to_pylist() == [3, 2, 2, 1]
    assert sorted6.pair.y.to_pylist() == [6, 6, 5, 4]


def test_sort_by_nested_doubly():
    class DoublyNested(qv.Table):
        inner = Wrapper.as_column()

    dn = DoublyNested.from_kwargs(
        inner=Wrapper.from_kwargs(id=["1", "2", "3"], pair=Pair.from_kwargs(x=[1, 2, 3], y=[5, 1, 2]))
    )

    sorted1 = dn.sort_by([("inner.pair.y", "ascending")])
    assert sorted1.inner.id.to_pylist() == ["2", "3", "1"]
    assert sorted1.inner.pair.x.to_pylist() == [2, 3, 1]
    assert sorted1.inner.pair.y.to_pylist() == [1, 2, 5]

    sorted2 = dn.sort_by([("inner.pair.x", "descending")])
    assert sorted2.inner.id.to_pylist() == ["3", "2", "1"]
    assert sorted2.inner.pair.x.to_pylist() == [3, 2, 1]
    assert sorted2.inner.pair.y.to_pylist() == [2, 1, 5]

    # Sort by id (same as original)
    sorted3 = dn.sort_by([("inner.id", "ascending")])
    assert sorted3.inner.id.to_pylist() == ["1", "2", "3"]
    assert sorted3.inner.pair.x.to_pylist() == [1, 2, 3]
    assert sorted3.inner.pair.y.to_pylist() == [5, 1, 2]

    sorted4 = dn.sort_by([("inner.id", "descending")])
    assert sorted4.inner.id.to_pylist() == ["3", "2", "1"]
    assert sorted4.inner.pair.x.to_pylist() == [3, 2, 1]
    assert sorted4.inner.pair.y.to_pylist() == [2, 1, 5]


def test_sort_by_nested_doubly_multiple_columns():
    class DoublyNested(qv.Table):
        inner = Wrapper.as_column()
        id = qv.StringColumn()

    dn = DoublyNested.from_kwargs(
        id=["1", "2", "3", "4"],
        inner=Wrapper.from_kwargs(
            id=["1", "2", "3", "4"], pair=Pair.from_kwargs(x=[2, 1, 2, 3], y=[6, 4, 5, 6])
        ),
    )

    # Column order: y first then x
    sorted1 = dn.sort_by([("inner.pair.y", "ascending"), ("inner.pair.x", "ascending")])
    assert sorted1.id.to_pylist() == ["2", "3", "1", "4"]
    assert sorted1.inner.id.to_pylist() == ["2", "3", "1", "4"]
    assert sorted1.inner.pair.x.to_pylist() == [1, 2, 2, 3]

    sorted2 = dn.sort_by([("inner.pair.y", "descending"), ("inner.pair.x", "descending")])
    assert sorted2.id.to_pylist() == ["4", "1", "3", "2"]
    assert sorted2.inner.id.to_pylist() == ["4", "1", "3", "2"]
    assert sorted2.inner.pair.x.to_pylist() == [3, 2, 2, 1]

    # Column order: x first then y
    sorted3 = dn.sort_by([("inner.pair.x", "ascending"), ("inner.pair.y", "ascending")])
    assert sorted3.id.to_pylist() == ["2", "3", "1", "4"]
    assert sorted3.inner.id.to_pylist() == ["2", "3", "1", "4"]
    assert sorted3.inner.pair.x.to_pylist() == [1, 2, 2, 3]

    sorted4 = dn.sort_by([("inner.pair.x", "descending"), ("inner.pair.y", "descending")])
    assert sorted4.id.to_pylist() == ["4", "1", "3", "2"]
    assert sorted4.inner.id.to_pylist() == ["4", "1", "3", "2"]
    assert sorted4.inner.pair.x.to_pylist() == [3, 2, 2, 1]

    # Column order: id then x
    sorted5 = dn.sort_by([("id", "ascending"), ("inner.pair.x", "ascending")])
    assert sorted5.id.to_pylist() == ["1", "2", "3", "4"]
    assert sorted5.inner.id.to_pylist() == ["1", "2", "3", "4"]
    assert sorted5.inner.pair.x.to_pylist() == [2, 1, 2, 3]

    sorted6 = dn.sort_by([("id", "descending"), ("inner.pair.x", "descending")])
    assert sorted6.id.to_pylist() == ["4", "3", "2", "1"]
    assert sorted6.inner.id.to_pylist() == ["4", "3", "2", "1"]
    assert sorted6.inner.pair.x.to_pylist() == [3, 2, 1, 2]


def test_sort_by_invalid_column():
    # Test that we raise an error if we try to sort by a column that
    # doesn't exist.
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[5, 1, 2])
    with pytest.raises(KeyError):
        pair.sort_by("z")

    # Test that we raise an error if we try to sort by a nested column
    # that doesn't exist.
    wrapper = Wrapper.from_kwargs(
        id=["1", "2", "3"],
        pair=pair,
    )
    with pytest.raises(KeyError):
        wrapper.sort_by("pair.z")


def test_sort_by_invalid_order():
    # Test that we raise an error if we try to sort with an
    # unrecognized order.
    pair = Pair.from_kwargs(x=[1, 2, 3], y=[5, 1, 2])
    with pytest.raises(ValueError):
        pair.sort_by([("x", "ascending"), ("y", "sideways")])


def test_sort_by_empty():
    # Test that we don't crash when sorting an empty table
    pair = Pair.empty()

    sorted1 = pair.sort_by("x")
    assert len(sorted1) == 0
    assert sorted1.x.to_pylist() == []


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


class TableWithDefaultAttributes(qv.Table):
    x = qv.Int64Column()
    y = qv.Int64Column()
    attrib = qv.StringAttribute(default="foo")


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


def test_invalid_mask():
    class MyTable(qv.Table):
        x = qv.Int8Column(validator=and_(lt(15), gt(10)))

    table = MyTable.from_kwargs(x=[8, 9, 10, 11, 12, 13, 14, 15, 16], validate=False)
    invalid = table.invalid_mask()
    np.testing.assert_array_equal(
        invalid.to_pylist(), [True, True, True, False, False, False, False, True, True]
    )


def test_separate_invalid():
    class MyTable(qv.Table):
        x = qv.Int8Column(validator=and_(lt(15), gt(10)))

    table = MyTable.from_kwargs(x=[8, 9, 10, 11, 12, 13, 14, 15, 16], validate=False)
    valid, invalid = table.separate_invalid()
    np.testing.assert_array_equal(valid.x, [11, 12, 13, 14])
    np.testing.assert_array_equal(invalid.x, [8, 9, 10, 15, 16])


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


def test_column():
    t = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    assert pc.all(pc.equal(t.x, t.column("x")))


def test_column_nested():
    w = Wrapper.from_kwargs(id=["a", "b", "c"], pair=Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6]))
    assert pc.all(pc.equal(w.pair.x, w.column("pair.x")))
    assert pc.all(pc.equal(w.pair.y, w.column("pair.y")))
    assert pc.all(pc.equal(w.id, w.column("id")))


def test_column_nested_doubly():
    class DoublyNested(qv.Table):
        inner = Wrapper.as_column()

    dn = DoublyNested.from_kwargs(
        inner=Wrapper.from_kwargs(id=["a", "b", "c"], pair=Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6]))
    )
    assert pc.all(pc.equal(dn.inner.pair.x, dn.column("inner.pair.x")))
    assert pc.all(pc.equal(dn.inner.pair.y, dn.column("inner.pair.y")))
    assert pc.all(pc.equal(dn.inner.id, dn.column("inner.id")))


def test_column_nulls():
    class PairWithNulls(qv.Table):
        x = qv.Int64Column(nullable=True)
        y = qv.Int64Column(nullable=True)

    t = PairWithNulls.from_kwargs(y=[4, 5, 6])
    assert pc.all(pc.equal(t.x, t.column("x")))
    assert pc.all(pc.equal(t.y, t.column("y")))

    t = PairWithNulls.from_kwargs(x=[1, 2, 3])
    assert pc.all(pc.equal(t.x, t.column("x")))
    assert pc.all(pc.equal(t.y, t.column("y")))


def test_column_nested_nulls():
    class PairWithNulls(qv.Table):
        x = qv.Int64Column(nullable=True)
        y = qv.Int64Column(nullable=True)

    class WrapperWithNulls(qv.Table):
        id = qv.StringColumn()
        pair = PairWithNulls.as_column(nullable=True)

    # Null grandchild
    w = WrapperWithNulls.from_kwargs(id=["a", "b", "c"], pair=PairWithNulls.from_kwargs(y=[4, 5, 6]))
    assert pc.all(pc.equal(w.pair.x, w.column("pair.x")))
    assert pc.all(pc.equal(w.pair.y, w.column("pair.y")))
    assert pc.all(pc.equal(w.id, w.column("id")))

    # Null child
    w = WrapperWithNulls.from_kwargs(id=["a", "b", "c"])
    assert pc.all(pc.equal(w.pair.x, w.column("pair.x")))
    assert pc.all(pc.equal(w.pair.y, w.column("pair.y")))
    assert pc.all(pc.equal(w.id, w.column("id")))


def test_column_nested_doubly_nulls():
    class PairWithNulls(qv.Table):
        x = qv.Int64Column(nullable=True)
        y = qv.Int64Column(nullable=True)

    class WrapperWithNulls(qv.Table):
        id = qv.StringColumn(nullable=True)
        pair = PairWithNulls.as_column(nullable=True)

    class DoublyNestedWithNulls(qv.Table):
        id = qv.StringColumn()
        inner = WrapperWithNulls.as_column(nullable=True)

    # Null great-grandchild
    dn = DoublyNestedWithNulls.from_kwargs(
        id=["a", "b", "c"],
        inner=WrapperWithNulls.from_kwargs(id=["a", "b", "c"], pair=PairWithNulls.from_kwargs(y=[4, 5, 6])),
    )
    assert pc.all(pc.equal(dn.inner.pair.x, dn.column("inner.pair.x")))
    assert pc.all(pc.equal(dn.inner.pair.y, dn.column("inner.pair.y")))
    assert pc.all(pc.equal(dn.inner.id, dn.column("inner.id")))
    assert pc.all(pc.equal(dn.id, dn.column("id")))

    # Null grandchild
    dn = DoublyNestedWithNulls.from_kwargs(
        id=["a", "b", "c"], inner=WrapperWithNulls.from_kwargs(id=["a", "b", "c"])
    )
    assert pc.all(pc.equal(dn.inner.pair.x, dn.column("inner.pair.x")))
    assert pc.all(pc.equal(dn.inner.pair.y, dn.column("inner.pair.y")))
    assert pc.all(pc.equal(dn.inner.id, dn.column("inner.id")))
    assert pc.all(pc.equal(dn.id, dn.column("id")))

    # Null child
    dn = DoublyNestedWithNulls.from_kwargs(id=["a", "b", "c"])
    assert pc.all(pc.equal(dn.inner.pair.x, dn.column("inner.pair.x")))
    assert pc.all(pc.equal(dn.inner.pair.y, dn.column("inner.pair.y")))
    assert pc.all(pc.equal(dn.inner.id, dn.column("inner.id")))
    assert pc.all(pc.equal(dn.id, dn.column("id")))


def test_column_empty():
    t = Pair.empty()
    assert len(t.column("x")) == 0
    assert len(t.column("y")) == 0


def test_column_nested_empty():
    w = Wrapper.empty()
    assert len(w.column("pair.x")) == 0
    assert len(w.column("pair.y")) == 0


def test_column_nested_doubly_empty():
    class DoublyNested(qv.Table):
        inner = Wrapper.as_column()

    dn = DoublyNested.empty()
    assert len(dn.column("inner.pair.x")) == 0
    assert len(dn.column("inner.pair.y")) == 0


def test_column_invalid_name():
    t = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    with pytest.raises(KeyError):
        t.column("z")


def test_column_nested_invalid_name():
    w = Wrapper.from_kwargs(id=["a", "b", "c"], pair=Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6]))
    with pytest.raises(KeyError):
        w.column("pair.z")
    with pytest.raises(AttributeError):
        w.column("wrong.x")


def test_column_nested_doubly_invalid_name():
    class DoublyNested(qv.Table):
        inner = Wrapper.as_column()

    dn = DoublyNested.from_kwargs(
        inner=Wrapper.from_kwargs(id=["a", "b", "c"], pair=Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6]))
    )
    with pytest.raises(KeyError):
        dn.column("inner.pair.z")
    with pytest.raises(AttributeError):
        dn.column("inner.wrong.x")


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


def test_set_column_none():
    class PairWithNulls(qv.Table):
        x = qv.Int64Column(nullable=True)
        y = qv.Int64Column(nullable=True)

    t = PairWithNulls.from_kwargs(x=[1, 2, 3], y=[4, 5, 6])
    t2 = t.set_column("x", None)

    # original should be unchanged
    assert t.x.equals(pa.array([1, 2, 3], pa.int64()))
    # new table should have new column
    assert t2.x.equals(pa.array([None, None, None], pa.int64()))


def test_set_column_changes_subtable_attribute():
    class Inner(qv.Table):
        x = qv.Int64Column()
        name = qv.StringAttribute()

    class Outer(qv.Table):
        inner = Inner.as_column()

    i = Inner.from_kwargs(x=[1, 2, 3], name="a")
    o = Outer.from_kwargs(inner=i)

    i2 = Inner.from_kwargs(x=[4, 5, 6], name="b")

    o2 = o.set_column("inner", i2)

    assert o2.inner.x.equals(pa.array([4, 5, 6], pa.int64()))
    assert o2.inner.name == "b"


@pytest.mark.benchmark(group="column-access")
class TestColumnAccessBenchmark:
    def test_access_f64(self, benchmark):
        class Table(qv.Table):
            x = qv.Float64Column()

        t = Table.from_kwargs(x=np.random.random(1_000_000))

        benchmark(getattr, t, "x")

    def test_access_subtable_f64(self, benchmark):
        class Inner(qv.Table):
            x = qv.Float64Column()

        class Outer(qv.Table):
            inner = Inner.as_column()

        t = Outer.from_kwargs(inner=Inner.from_kwargs(x=np.random.random(1_000_000)))

        def access():
            return t.inner.x

        benchmark(access)

    def test_access_f64_from_csv(self, tmp_path, benchmark):
        class Table(qv.Table):
            x = qv.Float64Column()

        t = Table.from_kwargs(x=np.random.random(1_000_000))
        t.to_csv(tmp_path / "test.csv")

        t2 = Table.from_csv(tmp_path / "test.csv")

        benchmark(getattr, t2, "x")

    def test_access_f64_raw(self, benchmark):
        class Table(qv.Table):
            x = qv.Float64Column()

        t = Table.from_kwargs(x=np.random.random(1_000_000))

        def raw_access():
            return t.table["x"]

        benchmark(raw_access)

    def test_access_f64_raw_and_combine(self, benchmark):
        class Table(qv.Table):
            x = qv.Float64Column()

        t = Table.from_kwargs(x=np.random.random(1_000_000))

        def raw_access():
            if t.table["x"].num_chunks == 1:
                return t.table["x"].chunk(0)
            else:
                return t.table["x"].combine_chunks()

        benchmark(raw_access)


class PairAttributed(qv.Table):
    x = qv.Int64Column()
    y = qv.Int64Column()
    name = qv.StringAttribute()
    id = qv.IntAttribute()


class WrapperAttributed(qv.Table):
    pair = PairAttributed.as_column()
    name = qv.StringAttribute()
    id = qv.IntAttribute()


class TestDataFrameConversions:
    def test_to_df_drop_attrs(self):
        t = PairAttributed.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], name="a", id=1)
        df = t.to_dataframe(attr_handling="drop")

        want = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        assert df.equals(want)

    def test_to_df_drop_attrs_wrapped(self):
        p = PairAttributed.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], name="a", id=1)
        t = WrapperAttributed.from_kwargs(pair=p, name="b", id=2)

        df = t.to_dataframe(attr_handling="drop")

        want = pd.DataFrame({"pair.x": [1, 2, 3], "pair.y": [4, 5, 6]})
        assert df.equals(want)

    def test_to_df_add_columns(self):
        t = PairAttributed.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], name="a", id=1)
        df = t.to_dataframe(attr_handling="add_columns")

        want = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "name": ["a", "a", "a"], "id": [1, 1, 1]})
        # sort_index is necessary because the order of the columns is not guaranteed
        assert df.sort_index(axis=1).equals(want.sort_index(axis=1))

    def test_to_df_add_columns_wrapped(self):
        p = PairAttributed.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], name="a", id=1)
        t = WrapperAttributed.from_kwargs(pair=p, name="b", id=2)

        df = t.to_dataframe(attr_handling="add_columns")

        want = pd.DataFrame(
            {
                "pair.x": [1, 2, 3],
                "pair.y": [4, 5, 6],
                "pair.name": ["a", "a", "a"],
                "pair.id": [1, 1, 1],
                "name": ["b", "b", "b"],
                "id": [2, 2, 2],
            }
        )
        # sort_index is necessary because the order of the columns is not guaranteed
        assert df.sort_index(axis=1).equals(want.sort_index(axis=1))

    def test_to_df_attrs(self):
        t = PairAttributed.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], name="a", id=1)
        df = t.to_dataframe(attr_handling="attrs")

        want = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        want.attrs = {"name": "a", "id": 1}
        assert df.equals(want)
        assert df.attrs == want.attrs

    def test_to_df_attrs_wrapped(self):
        p = PairAttributed.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], name="a", id=1)
        t = WrapperAttributed.from_kwargs(pair=p, name="b", id=2)

        df = t.to_dataframe(attr_handling="attrs")

        want = pd.DataFrame({"pair.x": [1, 2, 3], "pair.y": [4, 5, 6]})
        want.attrs = {"pair": {"name": "a", "id": 1}, "name": "b", "id": 2}

        assert df.equals(want)
        assert df.attrs == want.attrs

    def test_from_dataframe_missing_attrs(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        with pytest.raises(AttributeError):
            PairAttributed.from_dataframe(df)

        have = PairAttributed.from_dataframe(df, name="a", id=1)

        assert have.x.to_pylist() == [1, 2, 3]
        assert have.y.to_pylist() == [4, 5, 6]
        assert have.name == "a"
        assert have.id == 1

    def test_from_flat_dataframe_missing_attrs(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        with pytest.raises(AttributeError):
            PairAttributed.from_flat_dataframe(df)

        have = PairAttributed.from_flat_dataframe(df, name="a", id=1)

        assert have.x.to_pylist() == [1, 2, 3]
        assert have.y.to_pylist() == [4, 5, 6]
        assert have.name == "a"
        assert have.id == 1

    def test_from_dataframe_column_attrs(self):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [4, 5, 6],
                "name": ["a", "a", "a"],
                "id": [1, 1, 1],
            }
        )
        have = PairAttributed.from_dataframe(df)

        assert have.x.to_pylist() == [1, 2, 3]
        assert have.y.to_pylist() == [4, 5, 6]
        assert have.name == "a"
        assert have.id == 1

    def test_from_flat_dataframe_column_attrs_wrapped(self):
        df = pd.DataFrame(
            {
                "pair.x": [1, 2, 3],
                "pair.y": [4, 5, 6],
                "pair.name": ["a", "a", "a"],
                "pair.id": [1, 1, 1],
                "name": ["b", "b", "b"],
                "id": [2, 2, 2],
            }
        )
        have = WrapperAttributed.from_flat_dataframe(df)

        assert have.pair.x.to_pylist() == [1, 2, 3]
        assert have.pair.y.to_pylist() == [4, 5, 6]
        assert have.pair.name == "a"
        assert have.pair.id == 1
        assert have.name == "b"
        assert have.id == 2

    def test_from_dataframe_attrs(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df.attrs = {"name": "a", "id": 1}

        have = PairAttributed.from_dataframe(df)

        assert have.x.to_pylist() == [1, 2, 3]
        assert have.y.to_pylist() == [4, 5, 6]
        assert have.name == "a"
        assert have.id == 1

    def test_from_dataframe_attrs_overridden(self):
        # Prefer explicitly passed attributes over those in the dataframe
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df.attrs = {"name": "a", "id": 1}

        have = PairAttributed.from_dataframe(df, name="b", id=2)

        assert have.x.to_pylist() == [1, 2, 3]
        assert have.y.to_pylist() == [4, 5, 6]
        assert have.name == "b"
        assert have.id == 2

    def test_from_dataframe_attrs_unexpected_value(self):
        # Ignore unexpected attributes
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df.attrs = {"name": "a", "id": 1, "unexpected": "value"}

        have = PairAttributed.from_dataframe(df)

        assert have.x.to_pylist() == [1, 2, 3]
        assert have.y.to_pylist() == [4, 5, 6]
        assert have.name == "a"
        assert have.id == 1

    def test_from_flat_dataframe_attrs(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df.attrs = {"name": "a", "id": 1}

        have = PairAttributed.from_flat_dataframe(df)

        assert have.x.to_pylist() == [1, 2, 3]
        assert have.y.to_pylist() == [4, 5, 6]
        assert have.name == "a"
        assert have.id == 1

    def test_from_flat_dataframe_wrapped_attrs(self):
        df = pd.DataFrame({"pair.x": [1, 2, 3], "pair.y": [4, 5, 6]})
        df.attrs = {"pair": {"name": "a", "id": 1}, "name": "b", "id": 2}

        have = WrapperAttributed.from_flat_dataframe(df)

        assert have.pair.x.to_pylist() == [1, 2, 3]
        assert have.pair.y.to_pylist() == [4, 5, 6]
        assert have.pair.name == "a"
        assert have.pair.id == 1
        assert have.name == "b"
        assert have.id == 2

    def test_from_flat_dataframe_columnar_attrs(self):
        df = pd.DataFrame(
            {
                "pair.x": [1, 2, 3],
                "pair.y": [4, 5, 6],
                "pair.name": ["a", "a", "a"],
                "pair.id": [1, 1, 1],
                "name": ["b", "b", "b"],
                "id": [2, 2, 2],
            }
        )
        have = WrapperAttributed.from_flat_dataframe(df)

        assert have.pair.x.to_pylist() == [1, 2, 3]
        assert have.pair.y.to_pylist() == [4, 5, 6]
        assert have.pair.name == "a"
        assert have.pair.id == 1
        assert have.name == "b"
        assert have.id == 2

    def test_roundtrip(self):
        t = OmniTable.create(5)
        df = t.to_dataframe(flatten=True)

        have = OmniTable.from_flat_dataframe(df)

        # HACK: due to https://github.com/apache/arrow/issues/38050,
        # the date64 column will not correctly round-trip.  So we
        # drop it from the comparison.
        have = have.set_column("date64", t.date64)
        assert have == t

    def test_roundtrip_empty(self):
        t = OmniTable.empty()
        df = t.to_dataframe(flatten=True)

        have = OmniTable.from_flat_dataframe(df)

        assert have == t


class TestConstructors:
    def test_bench_construct_from_pyarrow(self, benchmark):
        N = 1_000_000
        x = pa.array(np.random.randint(low=-1000, high=1000, size=N), type=pa.int64())
        y = pa.array(np.random.randint(low=-1000, high=1000, size=N), type=pa.int64())
        benchmark(Pair.from_kwargs, x=x, y=y)

    def test_bench_construct_from_chunked_array(self, benchmark):
        N = 1_000_000
        N_CHUNKS = 1000
        chunks_x = [
            pa.array(np.random.randint(low=-1000, high=1000, size=N // N_CHUNKS), type=pa.int64())
            for _ in range(N_CHUNKS)
        ]
        chunks_y = [
            pa.array(np.random.randint(low=-1000, high=1000, size=N // N_CHUNKS), type=pa.int64())
            for _ in range(N_CHUNKS)
        ]
        x = pa.chunked_array(chunks_x)
        y = pa.chunked_array(chunks_y)

        benchmark(Pair.from_kwargs, x=x, y=y)

    def test_construct_from_chunked_array_wrong_type(self):
        # Constructing a Table using Pair.from_kwargs should raise an
        # error if one of the input data values is a Chunked Array
        # with incorrect type:
        chunks_x = [
            pa.array([1.5, 2.5, 3.5], type=pa.float64()),
            pa.array([4.5, 5.5, 6.5], type=pa.float64()),
        ]
        x = pa.chunked_array(chunks_x)
        y = pa.array([1, 2, 3, 4, 5, 6], type=pa.int64())

        with pytest.raises(ValueError):
            Pair.from_kwargs(x=x, y=y)
