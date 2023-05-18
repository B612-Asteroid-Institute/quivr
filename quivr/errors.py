import pyarrow


class InvariantViolatedError(RuntimeError):
    ...


class TableFragmentedError(InvariantViolatedError):
    ...


class ValidationError(Exception):
    def __init__(self, message: str, failures: pyarrow.Array = None):
        super().__init__(message)
        self.failures = failures
