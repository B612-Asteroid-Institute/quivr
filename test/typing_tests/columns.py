import pyarrow
from typing_extensions import assert_type

import quivr


class MyTable(quivr.Table):
    string_col = quivr.StringColumn()
    int8_col = quivr.Int8Column()
    float64_col = quivr.Float64Column()


instance = MyTable.from_data(string_col=["hello", "world"], int8_col=[1, 2], float64_col=[1.0, 2.0])

assert_type(instance, MyTable)
assert_type(instance.string_col, pyarrow.StringArray)
assert_type(instance.int8_col, pyarrow.Int8Array)
assert_type(instance.float64_col, pyarrow.Float64Array)

empty_instance = MyTable.empty()
assert_type(instance.string_col, pyarrow.StringArray)


class AttribTable(quivr.Table):
    string_attrib = quivr.StringAttribute()
    int_attrib = quivr.IntAttribute()
    float_attrib = quivr.FloatAttribute()


attrib_instance = AttribTable.empty(string_attrib="hello", int_attrib=1, float_attrib=1.0)
assert_type(attrib_instance.string_attrib, str)
assert_type(attrib_instance.int_attrib, int)
assert_type(attrib_instance.float_attrib, float)
