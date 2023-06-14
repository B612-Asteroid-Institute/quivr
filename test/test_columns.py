import enum

import pytest

import quivr


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class TestEnumColumn:
    def test_from_string_list(self):
        class MyTable(quivr.Table):
            color = quivr.EnumColumn(Color)

        have = MyTable.from_data(color=["red", "green", "blue"])
        assert have.color.to_pylist() == [Color.RED, Color.GREEN, Color.BLUE]

    def test_from_enum_list(self):
        class MyTable(quivr.Table):
            color = quivr.EnumColumn(Color)

        have = MyTable.from_data(color=[Color.RED, Color.GREEN, Color.BLUE])
        assert have.color.to_pylist() == [Color.RED, Color.GREEN, Color.BLUE]

    def test_from_invalid_list(self):
        class MyTable(quivr.Table):
            color = quivr.EnumColumn(Color)

        with pytest.raises(ValueError):
            MyTable.from_data(color=["red", "green", "chartreuse"])

    def test_from_rows_of_strings(self):
        class MyTable(quivr.Table):
            color = quivr.EnumColumn(Color)

        have = MyTable.from_rows(
            [
                {"color": "red"},
                {"color": "green"},
                {"color": "blue"},
            ]
        )
        assert have.color.to_pylist() == [Color.RED, Color.GREEN, Color.BLUE]

    @pytest.mark.xfail(
        reason="pyarrow doesn't support reading extension scalar types out of rows like this yet"
    )
    def test_from_rows_of_enums(self):
        class MyTable(quivr.Table):
            color = quivr.EnumColumn(Color)

        have = MyTable.from_rows(
            [
                {"color": Color.RED},
                {"color": Color.GREEN},
                {"color": Color.BLUE},
            ]
        )
        assert have.color.to_pylist() == [Color.RED, Color.GREEN, Color.BLUE]

    @pytest.mark.xfail(reason="pyarrow doesn't support converting numpy arrays into dictionaries")
    def test_from_numpy_array(self):
        class MyTable(quivr.Table):
            color = quivr.EnumColumn(Color)

        import numpy as np

        have = MyTable.from_data(color=np.array(["red", "green", "blue"]))
        assert have.color.to_pylist() == [Color.RED, Color.GREEN, Color.BLUE]

    def test_from_pandas_series(self):
        class MyTable(quivr.Table):
            color = quivr.EnumColumn(Color)

        import pandas as pd

        have = MyTable.from_data(color=pd.Series(["red", "green", "blue"]))
        assert have.color.to_pylist() == [Color.RED, Color.GREEN, Color.BLUE]

    def test_from_pandas_dataframe(self):
        class MyTable(quivr.Table):
            color = quivr.EnumColumn(Color)

        import pandas as pd

        have = MyTable.from_dataframe(pd.DataFrame({"color": ["red", "green", "blue"]}))
        assert have.color.to_pylist() == [Color.RED, Color.GREEN, Color.BLUE]
