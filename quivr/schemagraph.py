from typing import Callable, NoReturn, Optional

import pyarrow as pa


def _walk_schema(
    field: pa.Field,
    visitor: Callable[[pa.Field, list[pa.Field]], NoReturn],
    ancestors: Optional[list] = None,
):
    # Visit every struct field embedded within a field in depth-first fashion.
    #
    # Runs visitor(struct_field, [ancestors...]) on each struct field.
    if ancestors is None:
        ancestors = [field]
    else:
        ancestors += [field]

    for subfield in field.type:
        if pa.types.is_struct(subfield.type):
            _walk_schema(subfield, visitor, ancestors)

    ancestors.pop()
    visitor(field, ancestors)
