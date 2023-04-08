import pytest
from .test_tables import Pair, Wrapper
from quiver.concat import concatenate
import pyarrow as pa


def test_concatenate():
    xs1 = pa.array([1, 2, 3], pa.int64())
    ys1 = pa.array([4, 5, 6], pa.int64())
    pair1 = Pair.from_arrays([xs1, ys1])

    xs2 = pa.array([11, 22, 33], pa.int64())
    ys2 = pa.array([44, 55, 66], pa.int64())
    pair2 = Pair.from_arrays([xs2, ys2])

    have = concatenate([pair1, pair2])
    assert len(have) == 6
    assert have.x.to_pylist() == [1, 2, 3, 11, 22, 33]


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
    assert have.pair.x.to_pylist() == [1, 2, 3, 11, 22, 33]
    assert have.id.to_pylist() == ["v1", "v2", "v3", "v4", "v5", "v6"]


@pytest.mark.benchmark(group="ops")
def test_benchmark_concatenate_100(benchmark):
    xs1 = pa.array([1, 2, 3], pa.int64())
    ys1 = pa.array([4, 5, 6], pa.int64())
    pair1 = Pair.from_arrays([xs1, ys1])

    xs2 = pa.array([11, 22, 33], pa.int64())
    ys2 = pa.array([44, 55, 66], pa.int64())
    pair2 = Pair.from_arrays([xs2, ys2])

    benchmark(concatenate, [pair1, pair2] * 50)
