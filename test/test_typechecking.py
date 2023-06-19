import pyarrow
import pytest
from typing_extensions import assert_type

import quivr


class MyTable(quivr.Table):
    string_col = quivr.StringColumn()
    int8_col = quivr.Int8Column()
    float64_col = quivr.Float64Column()


class TestColumnTyping:
    instance = MyTable.empty()

    def test_string(self):
        assert_type(self.instance.string_col, pyarrow.StringArray)

    def test_int8(self):
        assert_type(self.instance.int8_col, pyarrow.Int8Array)

    def test_float64(self):
        assert_type(self.instance.float64_col, pyarrow.lib.DoubleArray)
