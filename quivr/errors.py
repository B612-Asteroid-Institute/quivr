from typing import Any, Optional

import pyarrow

from . import columns


class InvariantViolatedError(RuntimeError):
    """
    Exception raised when an invariant expectation is violated.
    """

    ...


class TableFragmentedError(InvariantViolatedError):
    """
    Exception raised when a Table is fragmented, and so an operation cannot
    be performed on it.
    """

    ...


class ValidationError(Exception):
    """
    Exception raised when data provided to a Table is invalid
    according to the Table's column validators.

    :ivar failures: An array of the invalid values that caused the
        exception to be raised.
    :vartype failures: pyarrow.Array
    """

    def __init__(self, message: str, failures: pyarrow.Array = None):
        super().__init__(message)
        self.failures = failures


class InvalidColumnDefault(Exception):
    """
    Exception raised when a column default is invalid.

    :ivar default_value: The invalid default value.
    :ivar dtype: The dtype of the column.
    :ivar column_name: The name of the column, if available. This may be None.
    """

    def __init__(self, default_value: Any, dtype: pyarrow.DataType, column_name: Optional[str] = None):
        msg = f"Invalid default value {repr(default_value)} for dtype {repr(dtype)}"
        if column_name is not None:
            msg += f" in column {repr(column_name)}"
        super().__init__(msg)
        self.default_value = default_value
        self.dtype = dtype
        self.column_name = column_name


class TablesNotCompatibleError(InvariantViolatedError):
    """Exception raised when two tables are not compatible for some
    operation like concatenation.
    """

    ...


class LinkageCombinationError(InvariantViolatedError):
    """Exception raised when linkages cannot be combined because they
    have incompatible table types or keys.
    """

    ...


class AttributeImmutableError(RuntimeError):
    """Exception raised when an attempt is made to modify an immutable
    attribute.
    """

    ...


class InvalidColumnDataError(Exception):
    """
    Exception raised when invalid data is provided to populate a particular column
    """

    def __init__(self, column: columns.Column, msg: str):
        self.column = column
        msg = f"Invalid data provided for column {column.name}: {msg}"
        super().__init__(msg)
