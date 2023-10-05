from collections.abc import Sequence
from typing import Any

import pyarrow as pa


def build_struct_array(data: Sequence[Any], fields: list[pa.Field]) -> pa.StructArray:
    """
    Cast a sequence of pandas Series into a StructArray using a given list of fields.

    The fields must be in the same order as the Series in the sequence.

    See also: https://github.com/apache/arrow/issues/35622
    """
    arrays = []
    for i, (s, f) in enumerate(zip(data, fields)):
        if isinstance(s, pa.Array):
            arrays.append(s)
            continue

        arrays.append(pa.array(s, type=f.type))

    return pa.StructArray.from_arrays(arrays, fields=fields)
