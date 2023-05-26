import pandas as pd
import pyarrow as pa
import pytest

from quivr import FloatAttribute, Int64Field, IntAttribute, StringAttribute, Table


def test_multiple_attributes():
    class MyTable(Table):
        vals = Int64Field()
        name = StringAttribute()
        id = IntAttribute()
        fval = FloatAttribute()

    table = MyTable.from_data(name="foo", id=1, vals=[1, 2, 3], fval=2.5)
    assert table.attributes() == {
        "name": "foo",
        "id": 1,
        "fval": 2.5,
    }


@pytest.mark.xfail(reason="nested attributes not working yet")
def test_nested_table():
    class Inner(Table):
        x = Int64Field()
        id = IntAttribute()

    class Outer(Table):
        name = StringAttribute()
        inner = Inner.as_field()

    table = Outer.from_data(
        name="foo",
        inner=Inner.from_data(
            x=[1, 2, 3],
            id=10,
        ),
    )
    assert table.name == "foo"
    assert table.inner.id == 10


@pytest.mark.xfail(reason="nested attributes not working yet")
def test_nested_table_shadowing():
    # Test that reusing a name at two levels is safe.
    class Inner(Table):
        x = Int64Field()
        id = IntAttribute()

    class Outer(Table):
        id = IntAttribute()
        inner = Inner.as_field()

    table = Outer.from_data(
        id=1,
        inner=Inner.from_data(
            x=[1, 2, 3],
            id=10,
        ),
    )
    assert table.id == 1
    assert table.inner.id == 10


@pytest.mark.xfail(reason="nested attributes not working yet")
def test_csv_nested_table(tmp_path):
    class Inner(Table):
        x = Int64Field()
        id = IntAttribute()

    class Outer(Table):
        name = StringAttribute()
        inner = Inner.as_field()

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
    class MyTable(Table):
        vals = Int64Field()
        name = StringAttribute()

    def test_from_dataframe(self):
        df = pd.DataFrame({"vals": [1, 2, 3]})
        table = self.MyTable.from_dataframe(df, name="foo")
        assert table.name == "foo"

    def test_from_flat_dataframe(self):
        class Inner(Table):
            x = Int64Field()
            id = IntAttribute()

        class Outer(Table):
            y = Int64Field()
            inner = Inner.as_field()
            name = StringAttribute()

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
    class MyTable(Table):
        vals = Int64Field()
        name = StringAttribute()

    def test_from_data(self):
        table = self.MyTable.from_data(name="foo", vals=[1, 2, 3])

        assert table.name == "foo"
        assert len(table) == 3

    def test_mutation(self, tmp_path):
        table = self.MyTable.from_data(name="foo", vals=[1, 2, 3])
        table.name = "bar"
        assert table.name == "bar"

        # Make sure mutation survives round trip
        table.to_parquet(tmp_path / "test.parquet")
        table2 = self.MyTable.from_parquet(tmp_path / "test.parquet")
        assert table2.name == "bar"

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
    class MyTable(Table):
        vals = Int64Field()
        id = IntAttribute()

    def test_from_data(self):
        table = self.MyTable.from_data(id=1, vals=[1, 2, 3])

        assert table.id == 1
        assert len(table) == 3

    def test_mutation(self):
        table = self.MyTable.from_data(id=1, vals=[1, 2, 3])
        table.id = 2
        assert table.id == 2

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
    class MyTable(Table):
        vals = Int64Field()
        id = FloatAttribute()

    def test_from_data(self):
        table = self.MyTable.from_data(id=1.5, vals=[1, 2, 3])

        assert table.id == 1.5
        assert len(table) == 3

    def test_mutation(self):
        table = self.MyTable.from_data(id=1.5, vals=[1, 2, 3])
        table.id = 2.5
        assert table.id == 2.5

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
