import numpy as np
import pyarrow as pa
import pytest

from quivr import validators
from quivr.errors import ValidationError


def test_eq():
    values = pa.array([1, 1, 1, 1], type=pa.int64())
    checker = validators.eq(1)
    assert checker.valid(values)
    checker.validate(values)

    values = pa.array([1, 1, 1, 2], type=pa.int64())
    assert not checker.valid(values)
    failures = checker.failures(values)
    indices, values = failures
    assert indices.to_pylist() == [3]
    assert values.to_pylist() == [2]

    with pytest.raises(ValidationError):
        checker.validate(values)


def test_lt():
    values = pa.array([1, 2, 3, 4], type=pa.int64())
    checker = validators.lt(5)
    assert checker.valid(values)

    values = pa.array([1, 2, 3, 5], type=pa.int64())
    assert not checker.valid(values)
    failures = checker.failures(values)
    indices, values = failures
    assert indices.to_pylist() == [3]
    assert values.to_pylist() == [5]


def test_le():
    values = pa.array([1, 2, 3, 4], type=pa.int64())
    checker = validators.le(4)
    assert checker.valid(values)

    values = pa.array([1, 2, 3, 5], type=pa.int64())
    assert not checker.valid(values)
    failures = checker.failures(values)
    indices, values = failures
    assert indices.to_pylist() == [3]
    assert values.to_pylist() == [5]


def test_gt():
    values = pa.array([1, 2, 3, 4], type=pa.int64())
    checker = validators.gt(0)
    assert checker.valid(values)

    values = pa.array([1, 2, 3, 0], type=pa.int64())
    assert not checker.valid(values)
    failures = checker.failures(values)
    indices, values = failures
    assert indices.to_pylist() == [3]
    assert values.to_pylist() == [0]


def test_ge():
    values = pa.array([1, 2, 3, 4], type=pa.int64())
    checker = validators.ge(1)
    assert checker.valid(values)

    values = pa.array([1, 2, 3, 0], type=pa.int64())
    assert not checker.valid(values)
    failures = checker.failures(values)
    indices, values = failures
    assert indices.to_pylist() == [3]
    assert values.to_pylist() == [0]


def test_is_in():
    values = pa.array(["a", "b", "c", "c"], type=pa.string())
    checker = validators.is_in(["a", "b", "c"])
    assert checker.valid(values)

    values = pa.array(["a", "b", "c", "d"], type=pa.string())
    assert not checker.valid(values)
    failures = checker.failures(values)
    indices, values = failures
    assert indices.to_pylist() == [3]
    assert values.to_pylist() == ["d"]


@pytest.mark.benchmark(group="validators")
def test_benchmark_eq(benchmark):
    values = pa.array(np.ones(1000), type=pa.int64())
    checker = validators.eq(1)
    benchmark(checker.valid, values)
