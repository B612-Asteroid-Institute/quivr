from typing_extensions import assert_type

import quivr as qv


class MyLeftTable(qv.Table):
    id_col = qv.Int64Column()
    x = qv.Float64Column()


class MyRightTable(qv.Table):
    l_id_col = qv.Int64Column()
    y = qv.Float64Column()


left_val = MyLeftTable.from_kwargs(id_col=[1, 2, 3], x=[1.0, 2.0, 3.0])

right_val = MyRightTable.from_kwargs(l_id_col=[1, 2, 3, 4], y=[1.0, 2.0, 3.0, 4.0])

linkage = qv.Linkage(left_val, right_val, left_val.id_col, right_val.l_id_col)

assert_type(linkage, qv.Linkage[MyLeftTable, MyRightTable])

assert_type(linkage.select_left(2), MyLeftTable)
assert_type(linkage.select_right(2), MyRightTable)

assert_type(linkage.select(2), tuple[MyLeftTable, MyRightTable])

multi_linkage = qv.MultiKeyLinkage(
    left_val,
    right_val,
    {"id": left_val.id_col, "pos": left_val.x},
    {"id": right_val.l_id_col, "pos": right_val.y},
)

assert_type(multi_linkage, qv.MultiKeyLinkage[MyLeftTable, MyRightTable])
