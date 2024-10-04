from typing import Any

import pyarrow
import pyarrow.compute as pc

from . import errors


class Validator:
    """A Validator is a tool to validate that data in a
    :class:`pyarrow.Array` matches a predicate expression.

    :ivar func: The predicate function to use for validation. This
        must be a scalar or scalar aggregate function.
    :ivar args: The arguments to pass to the predicate function.
    :ivar label: A label to use when reporting validation errors.
    """

    def __init__(self, func: pc.Function, args: list[Any], label: str):
        """
        :param func: The predicate function to use for validation. This
            must be a scalar or scalar aggregate function.
        :param args: The arguments to pass to the predicate function.
        :param label: A label to use when reporting validation errors.
        """
        self._check_arity(func, args)
        if func.kind not in {"scalar_aggregate", "scalar"}:
            raise ValueError(f"Invalid function type {func.kind}, must be scalar or scalar_aggregate")
        self.func = func
        self.args = args
        self.label = label

    def _check_arity(self, func: pc.Function, args: list[Any]) -> None:
        if not func.arity == (len(args) + 1):
            raise ValueError("Invalid number of arguments")

    def evaluate(self, array: pyarrow.Array) -> pyarrow.Array:
        """
        Evaluates the predicate function on the given array.

        :param array: The array to evaluate.
        """
        return self.func.call([array, *self.args])

    def valid(self, array: pyarrow.Array) -> bool:
        """
        Returns True if the given array is valid, False otherwise.

        :param array: The array to validate.
        """
        if self.func.kind == "scalar_aggregate":
            return self.evaluate(array).as_py()  # type: ignore
        else:
            results = self.evaluate(array)
            if results.null_count == len(results):
                return True
            return pc.all(self.evaluate(array)).as_py()  # type: ignore

    def validate(self, array: pyarrow.Array) -> None:
        """
        Raises a :class:`ValidationError` if the given array is not valid.

        :param array: The array to validate.
        """
        if not self.valid(array):
            if self.func.kind == "scalar_aggregate":
                raise errors.ValidationError(f"array did not pass validator '{self.label}'")
            indices, failures = self.failures(array)
            if len(failures) == 1:
                index = indices[0].as_py()
                value = failures[0].as_py()
                msg = f"val={value}, index={index} failed validator '{self.label}'"
                raise errors.ValidationError(msg, failures)
            else:
                n_failed = len(failures)
                index = indices[0].as_py()
                value = failures[0].as_py()
                msg = (
                    f"validator '{self.label}' failed on {n_failed} values, "
                    + f"first failure: val={value}, index={index}"
                )
                raise errors.ValidationError(msg, failures)

    def failures(self, array: pyarrow.Array) -> tuple[pyarrow.Array, pyarrow.Array]:
        """
        Returns a tuple of two arrays, the first containing the indices of the invalid values,
        and the second containing the invalid values themselves.

        If the validator is a scalar aggregate function, raises a TypeError.
        """
        if self.func.kind == "scalar_aggregate":
            raise TypeError("Cannot get failures for scalar aggregate function")

        # If we have an empty array, return early
        # Otherwise we will get a segfault when calling pc.indices_nonzero
        if len(array) == 0:
            return pyarrow.array([], type=pyarrow.int64()), pyarrow.array([], type=array.type)

        invalid = pc.invert(self.evaluate(array))
        indices = pc.indices_nonzero(invalid)
        invalid_values = pc.filter(array, invalid)
        return indices, invalid_values


class IsInValidator(Validator):
    """Executes the is_in validator, which is a special case because
    it takes the value set via a FunctionOption, rather than as a
    normal argument, so it has a different arity.

    """

    def __init__(self, args: list[Any], label: str):
        func = pc.get_function("is_in")
        super().__init__(func, args, label)

    def _check_arity(self, func: pc.Function, args: list[Any]) -> None:
        if not len(args) == 1:
            raise ValueError("Invalid number of arguments")

    def evaluate(self, array: pyarrow.Array) -> pyarrow.Array:
        return self.func.call([array], self.args[0])


class AndValidator(Validator):
    def __init__(self, validators: list[Validator], label: str):
        self.validators = validators
        func = pc.get_function("and")
        super().__init__(func, validators, label)

    def _check_arity(self, func: pc.Function, args: list[Any]) -> None:
        if not len(args) > 1:
            raise ValueError("Invalid number of arguments passed to and_")

    def evaluate(self, array: pyarrow.Array) -> pyarrow.Array:
        return pc.and_(*[v.evaluate(array) for v in self.validators])


def eq(val: Any) -> Validator:
    """
    Validator that all data in a column is equal to a given value.
    """
    func = pc.get_function("equal")
    label = f"eq({val})"
    return Validator(func, [val], label)


def lt(val: Any) -> Validator:
    """
    Validator that all data in a column is less than a given value.
    """
    func = pc.get_function("less")
    label = f"lt({val})"
    return Validator(func, [val], label)


def le(val: Any) -> Validator:
    """
    Validator that all data in a column is less than or equal to a given value.
    """
    func = pc.get_function("less_equal")
    label = f"le({val})"
    return Validator(func, [val], label)


def gt(val: Any) -> Validator:
    """
    Validator that all data in a column is greater than a given value.
    """
    func = pc.get_function("greater")
    label = f"gt({val})"
    return Validator(func, [val], label)


def ge(val: Any) -> Validator:
    """
    Validator that all data in a column is greater than or equal to a given value.
    """
    func = pc.get_function("greater_equal")
    label = f"ge({val})"
    return Validator(func, [val], label)


def is_in(val: Any, fail_on_null: bool = False) -> Validator:
    """Validator that all data in a column is in a given set.

    :param val: The set of values to check against. This can be a
        list, tuple, or pyarrow.Array. If it is a list or tuple, it
        will be converted to a pyarrow.Array.

    :param fail_on_null: If True, then nulls always trigger an
      error. Otherwise, they are matched to the value set, just like
      regular values.
    """
    label = f"is_in({val})"
    if not isinstance(val, pyarrow.Array):
        val = pyarrow.array(val)
    val = pc.SetLookupOptions(value_set=val, skip_nulls=fail_on_null)
    return IsInValidator([val], label)


def and_(*validators: Validator) -> Validator:
    """
    Validator that all data in a column passes all of the given validators.
    """
    label = "and({})".format(", ".join([v.label for v in validators]))
    return AndValidator(list(validators), label)
