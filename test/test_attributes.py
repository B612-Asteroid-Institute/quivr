import pandas as pd
import pyarrow as pa
import pytest

import quivr as qv


def test_multiple_attributes():
    class MyTable(qv.Table):
        vals = qv.Int64Column()
        name = qv.StringAttribute()
        id = qv.IntAttribute()
        fval = qv.FloatAttribute()

    table = MyTable.from_data(name="foo", id=1, vals=[1, 2, 3], fval=2.5)
    assert table.attributes() == {
        "name": "foo",
        "id": 1,
        "fval": 2.5,
    }


def test_nested_attribute_metadata():
    class Inner(qv.Table):
        x = qv.Int64Column()
        id1 = qv.StringAttribute()

    class Middle(qv.Table):
        inner = Inner.as_column()
        id2 = qv.StringAttribute()

    class Outer(qv.Table):
        middle = Middle.as_column()
        id3 = qv.StringAttribute()

    inner = Inner.from_kwargs(x=[1, 2, 3], id1="a")
    middle = Middle.from_kwargs(
        inner=inner,
        id2="b",
    )
    print(middle.table.schema.metadata)
    outer = Outer.from_kwargs(
        middle=middle,
        id3="c",
    )
    print(outer.table.schema.metadata)
    have = outer.table.schema.metadata
    assert have == {
        b"id3": b"c",
        b"middle.id2": b"b",
        b"middle.inner.id1": b"a",
    }


def test_nested_table():
    class Inner(qv.Table):
        x = qv.Int64Column()
        id = qv.IntAttribute()

    class Outer(qv.Table):
        name = qv.StringAttribute()
        inner = Inner.as_column()

    table = Outer.from_data(
        name="foo",
        inner=Inner.from_data(
            x=[1, 2, 3],
            id=10,
        ),
    )
    assert table.name == "foo"
    assert table.inner.id == 10


def test_nested_table_shadowing():
    # Test that reusing a name at two levels is safe.
    class Inner(qv.Table):
        x = qv.Int64Column()
        id = qv.IntAttribute()

    class Outer(qv.Table):
        id = qv.IntAttribute()
        inner = Inner.as_column()

    table = Outer.from_data(
        id=1,
        inner=Inner.from_data(
            x=[1, 2, 3],
            id=10,
        ),
    )
    assert table.id == 1
    assert table.inner.id == 10


def test_nested_table_mutability():
    class Inner(qv.Table):
        x = qv.Int64Column()
        id = qv.IntAttribute(mutable=True)

    class Outer(qv.Table):
        name = qv.StringAttribute(mutable=True)
        inner = Inner.as_column()

    table = Outer.from_data(
        name="foo",
        inner=Inner.from_data(
            x=[1, 2, 3],
            id=10,
        ),
    )
    assert table.name == "foo"
    assert table.inner.id == 10
    table.name = "bar"
    table.inner.id = 20
    assert table.name == "bar"
    assert table.inner.id == 10

    inner = table.inner
    inner.id = 30
    table.inner = inner
    assert table.name == "bar"
    assert table.inner.id == 30


def test_csv_nested_table(tmp_path):
    class Inner(qv.Table):
        x = qv.Int64Column()
        id = qv.IntAttribute()

    class Outer(qv.Table):
        name = qv.StringAttribute()
        inner = Inner.as_column()

    table = Outer.from_data(
        name="foo",
        inner=Inner.from_data(
            x=[1, 2, 3],
            id=10,
        ),
    )
    table.to_csv(tmp_path / "test.csv")

    table2 = Outer.from_csv(tmp_path / "test.csv")
    assert table2.name == "foo"
    assert table2.inner.id == 10


class TestConstructors:
    class MyTable(qv.Table):
        vals = qv.Int64Column()
        name = qv.StringAttribute()

    def test_from_dataframe(self):
        df = pd.DataFrame({"vals": [1, 2, 3]})
        table = self.MyTable.from_dataframe(df, name="foo")
        assert table.name == "foo"

    def test_from_flat_dataframe(self):
        class Inner(qv.Table):
            x = qv.Int64Column()
            id = qv.IntAttribute()

        class Outer(qv.Table):
            y = qv.Int64Column()
            inner = Inner.as_column()
            name = qv.StringAttribute()

        df = pd.DataFrame({"inner.x": [1, 2, 3], "y": [4, 5, 6]})
        table = Outer.from_flat_dataframe(df, name="foo")
        assert table.name == "foo"

    def test_from_kwargs(self):
        table = self.MyTable.from_kwargs(name="foo", vals=[1, 2, 3])
        assert table.name == "foo"

    def test_from_arrays(self):
        vals = pa.array([1, 2, 3])
        table = self.MyTable.from_arrays([vals], name="foo")
        assert table.name == "foo"

    def test_from_pydict(self):
        d = {"vals": [1, 2, 3]}
        table = self.MyTable.from_pydict(d, name="foo")
        assert table.name == "foo"

    def test_from_rows(self):
        rows = [{"vals": 1}, {"vals": 2}, {"vals": 3}]
        table = self.MyTable.from_rows(rows, name="foo")
        assert table.name == "foo"

    def test_from_lists(self):
        vals = [1, 2, 3]
        table = self.MyTable.from_lists([vals], name="foo")
        assert table.name == "foo"


class TestStringAttribute:
    class MyTable(qv.Table):
        vals = qv.Int64Column()
        name = qv.StringAttribute()
        mutable_name = qv.StringAttribute(mutable=True, default="bar")

    def test_from_data(self):
        table = self.MyTable.from_data(name="foo", vals=[1, 2, 3])

        assert table.name == "foo"
        assert len(table) == 3

    def test_mutation(self, tmp_path):
        table = self.MyTable.from_data(name="foo", mutable_name="bar", vals=[1, 2, 3])
        table.mutable_name = "baz"
        assert table.mutable_name == "baz"

        # Make sure mutation survives round trip
        table.to_parquet(tmp_path / "test.parquet")
        table2 = self.MyTable.from_parquet(tmp_path / "test.parquet")
        assert table2.mutable_name == "baz"

    def test_immutable(self):
        table = self.MyTable.from_data(name="foo", vals=[1, 2, 3])
        with pytest.raises(qv.AttributeImmutableError):
            table.name = "bar"

    def test_parquet_round_trip(self, tmp_path):
        table = self.MyTable.from_data(name="foo", vals=[1, 2, 3])
        table.to_parquet(tmp_path / "test.parquet")
        table2 = self.MyTable.from_parquet(tmp_path / "test.parquet")
        assert table2.name == "foo"

    def test_feather_round_trip(self, tmp_path):
        table = self.MyTable.from_data(name="foo", vals=[1, 2, 3])
        table.to_feather(tmp_path / "test.feather")
        table2 = self.MyTable.from_feather(tmp_path / "test.feather")
        assert table2.name == "foo"

    def test_csv_round_trip(self, tmp_path):
        table = self.MyTable.from_data(name="foo", vals=[1, 2, 3])
        table.to_csv(tmp_path / "test.csv")
        table2 = self.MyTable.from_csv(tmp_path / "test.csv")
        assert table2.name == "foo"


class TestIntAttribute:
    class MyTable(qv.Table):
        vals = qv.Int64Column()
        id = qv.IntAttribute()
        mutable_id = qv.IntAttribute(mutable=True, default=1)

    def test_from_data(self):
        table = self.MyTable.from_data(id=1, vals=[1, 2, 3])

        assert table.id == 1
        assert len(table) == 3

    def test_mutation(self):
        table = self.MyTable.from_data(id=1, vals=[1, 2, 3])
        table.mutable_id = 2
        assert table.mutable_id == 2

    def test_immutable(self):
        table = self.MyTable.from_data(id=1, vals=[1, 2, 3])
        with pytest.raises(qv.AttributeImmutableError):
            table.id = 2

    def test_parquet_round_trip(self, tmp_path):
        table = self.MyTable.from_data(id=1, vals=[1, 2, 3])
        table.to_parquet(tmp_path / "test.parquet")
        table2 = self.MyTable.from_parquet(tmp_path / "test.parquet")
        assert table2.id == 1

    def test_feather_round_trip(self, tmp_path):
        table = self.MyTable.from_data(id=1, vals=[1, 2, 3])
        table.to_feather(tmp_path / "test.feather")
        table2 = self.MyTable.from_feather(tmp_path / "test.feather")
        assert table2.id == 1

    def test_csv_round_trip(self, tmp_path):
        table = self.MyTable.from_data(id=1, vals=[1, 2, 3])
        table.to_csv(tmp_path / "test.csv")
        table2 = self.MyTable.from_csv(tmp_path / "test.csv")
        assert table2.id == 1


class TestFloatAttribute:
    class MyTable(qv.Table):
        vals = qv.Int64Column()
        id = qv.FloatAttribute()
        mutable_id = qv.FloatAttribute(mutable=True, default=1.5)

    def test_from_data(self):
        table = self.MyTable.from_data(id=1.5, vals=[1, 2, 3])

        assert table.id == 1.5
        assert len(table) == 3

    def test_mutation(self):
        table = self.MyTable.from_data(id=1.5, vals=[1, 2, 3])
        table.mutable_id = 2.5
        assert table.mutable_id == 2.5

    def test_immutable(self):
        table = self.MyTable.from_data(id=1.5, vals=[1, 2, 3])
        with pytest.raises(qv.AttributeImmutableError):
            table.id = 2.5

    def test_parquet_round_trip(self, tmp_path):
        table = self.MyTable.from_data(id=1.5, vals=[1, 2, 3])
        table.to_parquet(tmp_path / "test.parquet")
        table2 = self.MyTable.from_parquet(tmp_path / "test.parquet")
        assert table2.id == 1.5

    def test_feather_round_trip(self, tmp_path):
        table = self.MyTable.from_data(id=1.5, vals=[1, 2, 3])
        table.to_feather(tmp_path / "test.feather")
        table2 = self.MyTable.from_feather(tmp_path / "test.feather")
        assert table2.id == 1.5

    def test_csv_round_trip(self, tmp_path):
        table = self.MyTable.from_data(id=1.5, vals=[1, 2, 3])
        table.to_csv(tmp_path / "test.csv")
        table2 = self.MyTable.from_csv(tmp_path / "test.csv")
        assert table2.id == 1.5


def test_attribute_metadata_keys():
    class Inner1(qv.Table):
        id = qv.IntAttribute()

    class Inner2(qv.Table):
        a = qv.StringAttribute()
        b = qv.StringAttribute()

    class DoubleInner(qv.Table):
        id = qv.IntAttribute()

    class Middle(qv.Table):
        c = qv.StringAttribute()
        inner = DoubleInner.as_column()

    class Outer(qv.Table):
        inner1 = Inner1.as_column()
        inner2 = Inner2.as_column()
        middle = Middle.as_column()
        id = qv.IntAttribute()

    have = Outer._attribute_metadata_keys()
    assert have == {"inner1.id", "inner2.a", "inner2.b", "middle.c", "middle.inner.id", "id"}


def test_benchmark_int_attribute_access(benchmark):
    class MyTable(qv.Table):
        vals = qv.Int64Column()
        attr = qv.IntAttribute()

    table = MyTable.from_kwargs(vals=[], attr=1)
    benchmark(lambda: table.attr)


def test_benchmark_int_attribute_mutation(benchmark):
    class MyTable(qv.Table):
        vals = qv.Int64Column()
        attr = qv.IntAttribute(mutable=True)

    table = MyTable.from_kwargs(vals=[], attr=1)

    def increment(table):
        table.attr += 1

    benchmark(increment, table)


def test_benchmark_str_attribute_access(benchmark):
    class MyTable(qv.Table):
        vals = qv.Int64Column()
        attr = qv.StringAttribute()

    table = MyTable.from_kwargs(vals=[], attr="foo")
    benchmark(lambda: table.attr)


def test_benchmark_str_attribute_mutation(benchmark):
    class MyTable(qv.Table):
        vals = qv.Int64Column()
        attr = qv.StringAttribute(mutable=True)

    table = MyTable.from_kwargs(vals=[], attr="foo")

    def increment(table):
        table.attr = (table.attr + "bar")[:5]

    benchmark(increment, table)


def test_benchmark_float_attribute_access(benchmark):
    class MyTable(qv.Table):
        vals = qv.Int64Column()
        attr = qv.FloatAttribute()

    table = MyTable.from_kwargs(vals=[], attr=1.0)
    benchmark(lambda: table.attr)


def test_benchmark_float_attribute_mutation(benchmark):
    class MyTable(qv.Table):
        vals = qv.Int64Column()
        attr = qv.FloatAttribute(mutable=True)

    table = MyTable.from_kwargs(vals=[], attr=1.0)

    def increment(table):
        table.attr += 1.0

    benchmark(increment, table)
