import pytest

from quivr import FloatAttribute, Int64Field, IntAttribute, StringAttribute, Table


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

    @pytest.mark.xfail(reason="CSVs don't round-trip attributes")
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

    @pytest.mark.xfail(reason="CSVs don't round-trip attributes")
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

    @pytest.mark.xfail(reason="CSVs don't round-trip attributes")
    def test_csv_round_trip(self, tmp_path):
        table = self.MyTable.from_data(id=1.5, vals=[1, 2, 3])
        table.to_csv(tmp_path / "test.csv")
        table2 = self.MyTable.from_csv(tmp_path / "test.csv")
        assert table2.id == 1.5
