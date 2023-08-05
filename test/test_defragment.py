import pyarrow as pa
import pytest

import quivr as qv

from .test_tables import Pair


def test_defragment():
    p1 = Pair.from_kwargs(
        x=pa.array([1, 2, 3], pa.int64()),
        y=pa.array([4, 5, 6], pa.int64()),
    )
    combined = qv.concatenate([p1] * 10, defrag=False)
    assert len(combined) == 30
    assert len(combined.column("x").chunks) == 10

    defragged = qv.defragment(combined)
    assert len(defragged) == 30
    assert len(defragged.column("x").chunks) == 1

    assert len(combined.column("x").chunks) == 10


@pytest.mark.benchmark(group="ops")
def test_benchmark_defragment_100(benchmark):
    p1 = Pair.from_kwargs(
        x=pa.array([1, 2, 3], pa.int64()),
        y=pa.array([4, 5, 6], pa.int64()),
    )
    combined = qv.concatenate([p1] * 100, defrag=False)
    benchmark(qv.defragment, combined)
