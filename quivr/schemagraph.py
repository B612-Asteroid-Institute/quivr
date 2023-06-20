from typing import Callable, Optional

import pyarrow as pa


def _walk_schema(
    field: pa.Field,
    visitor: Callable[[pa.Field, list[pa.Field]], None],
    ancestors: Optional[list[pa.Field]] = None,
) -> None:
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
