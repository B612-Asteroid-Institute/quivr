import pyarrow


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
