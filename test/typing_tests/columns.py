import pyarrow
from typing_extensions import assert_type

import quivr


class Pair(quivr.Table):
    x = quivr.Float64Column()


class MyTable(quivr.Table):
    string_col = quivr.StringColumn()
    int8_col = quivr.Int8Column()
    float64_col = quivr.Float64Column()

    pair_col = Pair.as_column(nullable=True)


instance = MyTable.from_data(string_col=["hello", "world"], int8_col=[1, 2], float64_col=[1.0, 2.0])

assert_type(instance, MyTable)

# Instance attributes should return pyarrow types
assert_type(instance.string_col, pyarrow.StringArray)
assert_type(instance.int8_col, pyarrow.Int8Array)
assert_type(instance.float64_col, pyarrow.lib.DoubleArray)

# Class-level attributes should be the columns themselves
assert_type(MyTable.string_col, quivr.StringColumn)

# Sub-tables should be generics
assert_type(instance.pair_col, Pair)
assert_type(MyTable.pair_col, quivr.SubTableColumn[Pair])


empty_instance = MyTable.empty()
assert_type(instance.string_col, pyarrow.StringArray)
