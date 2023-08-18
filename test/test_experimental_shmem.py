import multiprocessing
import multiprocessing.managers
import multiprocessing.shared_memory

import pytest

import quivr as qv
from quivr.experimental import shmem


class Pair(qv.Table):
    x = qv.Int64Column()
    y = qv.Int64Column()
    name = qv.StringAttribute()


class Wrapper(qv.Table):
    pair = Pair.as_column()
    label = qv.StringAttribute()


def count_rows(table):
    return len(table)


def crash(table):
    raise RuntimeError("crash")


def test_run_on_shared_memory():
    data = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], name="hello")
    with multiprocessing.managers.SharedMemoryManager() as mgr:
        shm = shmem.to_shared_memory(data, mgr)

        result = shmem._run_on_shared_memory(shm.name, Pair, count_rows)
        assert result == 3


def test_run_on_shared_memory_crash():
    data = Pair.from_kwargs(x=[1, 2, 3], y=[4, 5, 6], name="hello")
    with multiprocessing.managers.SharedMemoryManager() as mgr:
        shm = shmem.to_shared_memory(data, mgr)

        with pytest.raises(RuntimeError, match="crash"):
            shmem._run_on_shared_memory(shm.name, Pair, crash)


def test_execute_parallel():
    data = Pair.from_kwargs(x=[1, 2, 3, 4, 5, 6], y=[4, 5, 6, 7, 8, 9], name="hello")

    results = shmem.execute_parallel(
        data, count_rows, max_workers=2, partitioning=shmem.ChunkedPartitioning(chunk_size=2)
    )
    assert results == [2, 2, 2]


def return_y_values(table):
    x = table.x.unique().to_pylist()[0]
    return x, table.y.to_pylist()


def test_execute_parallel_grouped_partitioning():
    data = Pair.from_kwargs(
        x=[1, 1, 2, 2, 3, 3, 3, 4, 5, 1],
        y=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        name="hello",
    )
    partitioning = shmem.GroupedPartitioning(group_column="x")

    results = shmem.execute_parallel(data, return_y_values, max_workers=2, partitioning=partitioning)
    results = dict(results)
    assert results == {
        1: [1, 2, 10],
        2: [3, 4],
        3: [5, 6, 7],
        4: [8],
        5: [9],
    }
