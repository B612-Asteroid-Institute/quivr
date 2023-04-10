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


def compute_depth(schema: pa.Schema) -> int:
    """Compute the maximum depth of the schema, thinking of it as a tree which contains struct types."""
    max_depth = 1
    for field in schema:
        if pa.types.is_struct(field.type):
            depth = 1 + compute_depth(pa.schema(field.type))
            if depth > max_depth:
                max_depth = depth
    return max_depth
