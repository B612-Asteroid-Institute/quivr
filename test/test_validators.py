import numpy as np
import pyarrow as pa
import pytest

import quivr as qv


def test_eq():
    values = pa.array([1, 1, 1, 1], type=pa.int64())
    checker = qv.eq(1)
    assert checker.valid(values)
    checker.validate(values)

    values = pa.array([1, 1, 1, 2], type=pa.int64())
    assert not checker.valid(values)
    failures = checker.failures(values)
    indices, values = failures
    assert indices.to_pylist() == [3]
    assert values.to_pylist() == [2]

    with pytest.raises(qv.ValidationError):
        checker.validate(values)


def test_lt():
    values = pa.array([1, 2, 3, 4], type=pa.int64())
    checker = qv.lt(5)
    assert checker.valid(values)

    values = pa.array([1, 2, 3, 5], type=pa.int64())
    assert not checker.valid(values)
    failures = checker.failures(values)
    indices, values = failures
    assert indices.to_pylist() == [3]
    assert values.to_pylist() == [5]


def test_le():
    values = pa.array([1, 2, 3, 4], type=pa.int64())
    checker = qv.le(4)
    assert checker.valid(values)

    values = pa.array([1, 2, 3, 5], type=pa.int64())
    assert not checker.valid(values)
    failures = checker.failures(values)
    indices, values = failures
    assert indices.to_pylist() == [3]
    assert values.to_pylist() == [5]


def test_gt():
    values = pa.array([1, 2, 3, 4], type=pa.int64())
    checker = qv.gt(0)
    assert checker.valid(values)

    values = pa.array([1, 2, 3, 0], type=pa.int64())
    assert not checker.valid(values)
    failures = checker.failures(values)
    indices, values = failures
    assert indices.to_pylist() == [3]
    assert values.to_pylist() == [0]


def test_ge():
    values = pa.array([1, 2, 3, 4], type=pa.int64())
    checker = qv.ge(1)
    assert checker.valid(values)

    values = pa.array([1, 2, 3, 0], type=pa.int64())
    assert not checker.valid(values)
    failures = checker.failures(values)
    indices, values = failures
    assert indices.to_pylist() == [3]
    assert values.to_pylist() == [0]


def test_is_in():
    values = pa.array(["a", "b", "c", "c"], type=pa.string())
    checker = qv.is_in(["a", "b", "c"])
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
    checker = qv.eq(1)
    benchmark(checker.valid, values)


def test_validating_with_nulls():
    values = pa.array([1, 1, 1, None], type=pa.int64())
    checker = qv.eq(1)
    checker.validate(values)


def test_validate_all_null():
    values = pa.array([None, None, None, None], type=pa.int64())
    checker = qv.eq(1)
    checker.validate(values)
