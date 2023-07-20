from typing_extensions import assert_type

import quivr


class MyLeftTable(quivr.Table):
    id_col = quivr.Int64Column()
    x = quivr.Float64Column()


class MyRightTable(quivr.Table):
    l_id_col = quivr.Int64Column()
    y = quivr.Float64Column()


left_val = MyLeftTable.from_kwargs(id_col=[1, 2, 3], x=[1.0, 2.0, 3.0])

right_val = MyRightTable.from_kwargs(l_id_col=[1, 2, 3, 4], y=[1.0, 2.0, 3.0, 4.0])

linkage = quivr.Linkage(left_val, right_val, left_val.id_col, right_val.l_id_col)

assert_type(linkage, quivr.Linkage[MyLeftTable, MyRightTable])

assert_type(linkage.select_left(2), MyLeftTable)
assert_type(linkage.select_right(2), MyRightTable)

assert_type(linkage.select(2), tuple[MyLeftTable, MyRightTable])

multi_linkage = quivr.MultiKeyLinkage(
    left_val,
    right_val,
    {"id": left_val.id_col, "pos": left_val.x},
    {"id": right_val.l_id_col, "pos": right_val.y},
)

assert_type(multi_linkage, quivr.MultiKeyLinkage[MyLeftTable, MyRightTable])
