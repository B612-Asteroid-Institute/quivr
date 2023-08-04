import numpy as np
import pyarrow as pa
import pytest

from quivr import errors
from quivr.concat import concatenate

from .test_tables import Pair, TableWithAttributes, Wrapper


def test_concatenate():
    xs1 = pa.array([1, 2, 3], pa.int64())
    ys1 = pa.array([4, 5, 6], pa.int64())
    pair1 = Pair.from_arrays([xs1, ys1])

    xs2 = pa.array([11, 22, 33], pa.int64())
    ys2 = pa.array([44, 55, 66], pa.int64())
    pair2 = Pair.from_arrays([xs2, ys2])

    have = concatenate([pair1, pair2])
    assert len(have) == 6
    np.testing.assert_array_equal(have.x, [1, 2, 3, 11, 22, 33])


def test_concatenate_nested():
    xs1 = pa.array([1, 2, 3], pa.int64())
    ys1 = pa.array([4, 5, 6], pa.int64())
    pairs1 = pa.StructArray.from_arrays([xs1, ys1], fields=list(Pair.schema))
    ids1 = pa.array(["v1", "v2", "v3"], pa.string())
    w1 = Wrapper.from_arrays([pairs1, ids1])

    xs2 = pa.array([11, 22, 33], pa.int64())
    ys2 = pa.array([44, 55, 66], pa.int64())
    pairs2 = pa.StructArray.from_arrays([xs2, ys2], fields=list(Pair.schema))
    ids2 = pa.array(["v4", "v5", "v6"], pa.string())
    w2 = Wrapper.from_arrays([pairs2, ids2])

    have = concatenate([w1, w2])
    assert len(have) == 6
    np.testing.assert_array_equal(have.pair.x, [1, 2, 3, 11, 22, 33])
    np.testing.assert_array_equal(have.id, ["v1", "v2", "v3", "v4", "v5", "v6"])


def test_concatenate_empty():
    with pytest.raises(ValueError, match="No values to concatenate"):
        concatenate([])


@pytest.mark.benchmark(group="ops")
def test_benchmark_concatenate_100(benchmark):
    xs1 = pa.array([1, 2, 3], pa.int64())
    ys1 = pa.array([4, 5, 6], pa.int64())
    pair1 = Pair.from_arrays([xs1, ys1])

    xs2 = pa.array([11, 22, 33], pa.int64())
    ys2 = pa.array([44, 55, 66], pa.int64())
    pair2 = Pair.from_arrays([xs2, ys2])

    benchmark(concatenate, [pair1, pair2] * 50)


def test_concatenate_different_types():
    with pytest.raises(
        errors.TablesNotCompatibleError, match="All tables must be the same class to concatenate"
    ):
        concatenate([Pair.empty(), Wrapper.empty()])


def test_concatenate_different_attrs():
    t1 = TableWithAttributes.from_data(x=[1], y=[2], attrib="foo")
    t2 = TableWithAttributes.from_data(x=[3], y=[4], attrib="bar")

    with pytest.raises(
        errors.TablesNotCompatibleError, match="All tables must have the same attribute values to concatenate"
    ):
        concatenate([t1, t2])


def test_concatenate_same_attrs():
    t1 = TableWithAttributes.from_data(x=[1], y=[2], attrib="foo")
    t2 = TableWithAttributes.from_data(x=[3], y=[4], attrib="foo")
    have = concatenate([t1, t2])
    assert have.attrib == "foo"
