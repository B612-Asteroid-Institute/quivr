import pyarrow.compute as pc

import quivr as qv


class Pair(qv.Table):
    x = qv.Int64Column()
    y = qv.Int64Column()


class Container(qv.Table):
    pair = Pair.as_column()
    name = qv.StringColumn()


class TestParquetSerialization:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "test.parquet"
        pair = Pair.from_data(x=[1, 2, 3], y=[4, 5, 6])
        container = Container.from_data(pair=pair, name=["hello", "world", "foo"])
        container.to_parquet(path)
        assert path.exists()
        assert path.stat().st_size > 0

        have = Container.from_parquet(path)

        assert have == container

    def test_mmap(self, tmp_path):
        path = tmp_path / "test.parquet"
        pair = Pair.from_data(x=[1, 2, 3], y=[4, 5, 6])
        container = Container.from_data(pair=pair, name=["hello", "world", "foo"])
        container.to_parquet(path)

        have = Container.from_parquet(path, memory_map=True)
        assert have == container

    def test_filters(self, tmp_path):
        path = tmp_path / "test.parquet"
        pair = Pair.from_data(x=[1, 2, 3], y=[4, 5, 6])
        container = Container.from_data(pair=pair, name=["hello", "world", "foo"])
        container.to_parquet(path)
        assert path.exists()
        assert path.stat().st_size > 0

        filter1 = pc.equal(pc.field("name"), "world")
        have = Container.from_parquet(path, filters=filter1)
        assert have == container[1]

        filter2 = pc.equal(pc.field("pair", "x"), 1)
        have = Container.from_parquet(path, filters=filter2)
        assert have == container[0]

    def test_column_renaming(self, tmp_path):
        path = tmp_path / "test.parquet"
        pair = Pair.from_data(x=[1, 2, 3], y=[4, 5, 6])
        pair.to_parquet(path)

        class Pair2(qv.Table):
            a = qv.Int64Column()
            b = qv.Int64Column()

        have = Pair2.from_parquet(path, column_name_map={"x": "a", "y": "b"})
        want = Pair2.from_data(a=[1, 2, 3], b=[4, 5, 6])
        assert have == want
