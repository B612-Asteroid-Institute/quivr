from typing_extensions import assert_type

import quivr as qv


class AttribTable(qv.Table):
    string_attrib = qv.StringAttribute()
    int_attrib = qv.IntAttribute()
    float_attrib = qv.FloatAttribute()


attrib_instance = AttribTable.empty(string_attrib="hello", int_attrib=1, float_attrib=1.0)
assert_type(attrib_instance.string_attrib, str)
assert_type(attrib_instance.int_attrib, int)
assert_type(attrib_instance.float_attrib, float)
