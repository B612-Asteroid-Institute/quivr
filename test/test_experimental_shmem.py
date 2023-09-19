import concurrent.futures.process
import os
import signal

import pytest

import quivr as qv
from quivr.experimental import shmem as qv_shmem


class Pair(qv.Table):
    x = qv.Float64Column()
    y = qv.Float64Column()

    def pairwise_min(self):
        return (self.x.to_numpy() * self.y.to_numpy()).min()


class NamedPair(qv.Table):
    x = qv.Float64Column()
    y = qv.Float64Column()
    name = qv.StringAttribute()

    def pairwise_min(self):
        return (self.x.to_numpy() * self.y.to_numpy()).min()

    def name_is_foo(self):
        return self.name == "foo"


class Wrapper(qv.Table):
    pair = Pair.as_column()
    id = qv.Int64Column()

    def pairwise_min(self):
        return self.pair.pairwise_min()


class NamedWrapper(qv.Table):
    pair = NamedPair.as_column()
    id = qv.Int64Column()

    def pairwise_min(self):
        return self.pair.pairwise_min()

    def name_is_foo(self):
        return self.pair.name_is_foo()


def test_execute_parallel_method():
    pairs = Pair.from_kwargs(
        x=[1, 2, 3, 4, 5, 6, 7, 8],
        y=[8, 7, 6, 5, 4, 3, 2, 1],
    )
    partitioning = qv_shmem.ChunkedPartitioning(chunk_size=2)

    results_iter = qv_shmem.execute_parallel(
        pairs, Pair.pairwise_min, partitioning=partitioning, max_workers=2
    )
    results_list = list(results_iter)

    assert len(results_list) == 4
    sorted_results = sorted(results_list)
    assert sorted_results == [8.0, 8.0, 18.0, 18.0]


def test_execute_parallel_nested_table():
    pairs = Pair.from_kwargs(
        x=[1, 2, 3, 4, 5, 6, 7, 8],
        y=[8, 7, 6, 5, 4, 3, 2, 1],
    )
    wrapper = Wrapper.from_kwargs(
        pair=pairs,
        id=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    partitioning = qv_shmem.ChunkedPartitioning(chunk_size=2)

    results_iter = qv_shmem.execute_parallel(
        wrapper, Wrapper.pairwise_min, partitioning=partitioning, max_workers=2
    )
    results_list = list(results_iter)

    assert len(results_list) == 4
    sorted_results = sorted(results_list)
    assert sorted_results == [8.0, 8.0, 18.0, 18.0]


def test_execute_parallel_attributes():
    pairs = NamedPair.from_kwargs(
        x=[1, 2, 3, 4, 5, 6, 7, 8],
        y=[8, 7, 6, 5, 4, 3, 2, 1],
        name="foo",
    )
    partitioning = qv_shmem.ChunkedPartitioning(chunk_size=2)

    results_iter = qv_shmem.execute_parallel(
        pairs, NamedPair.name_is_foo, partitioning=partitioning, max_workers=2
    )
    results_list = list(results_iter)

    assert len(results_list) == 4
    sorted_results = sorted(results_list)
    assert sorted_results == [True, True, True, True]


def test_execute_parallel_nested_attributes():
    pairs = NamedPair.from_kwargs(
        x=[1, 2, 3, 4, 5, 6, 7, 8],
        y=[8, 7, 6, 5, 4, 3, 2, 1],
        name="foo",
    )
    wrapper = NamedWrapper.from_kwargs(
        pair=pairs,
        id=[1, 2, 3, 4, 5, 6, 7, 8],
    )

    partitioning = qv_shmem.ChunkedPartitioning(chunk_size=2)

    results_iter = qv_shmem.execute_parallel(
        wrapper, NamedWrapper.name_is_foo, partitioning=partitioning, max_workers=2
    )
    results_list = list(results_iter)

    assert len(results_list) == 4
    sorted_results = sorted(results_list)
    assert sorted_results == [True, True, True, True]


def multiply_by_n(pairs: Pair, n: float) -> Pair:
    x = pairs.x.to_numpy() * n
    y = pairs.y.to_numpy() * n
    return Pair.from_kwargs(x=x, y=y)


def test_execute_parallel_extra_args():
    pairs = Pair.from_kwargs(
        x=[1, 2, 3, 4, 5, 6, 7, 8],
        y=[8, 7, 6, 5, 4, 3, 2, 1],
    )
    partitioning = qv_shmem.ChunkedPartitioning(chunk_size=2)

    results_iter = qv_shmem.execute_parallel(
        pairs, multiply_by_n, partitioning=partitioning, max_workers=2, args=[2.0]
    )
    results_list = list(results_iter)

    assert len(results_list) == 4
    sorted_results = sorted(results_list, key=lambda p: p.x[0].as_py())

    combined = qv.concatenate(sorted_results)

    assert combined.x.to_pylist() == [2, 4, 6, 8, 10, 12, 14, 16]
    assert combined.y.to_pylist() == [16, 14, 12, 10, 8, 6, 4, 2]


def test_execute_parallel_extra_kwargs():
    pairs = Pair.from_kwargs(
        x=[1, 2, 3, 4, 5, 6, 7, 8],
        y=[8, 7, 6, 5, 4, 3, 2, 1],
    )
    partitioning = qv_shmem.ChunkedPartitioning(chunk_size=2)

    results_iter = qv_shmem.execute_parallel(
        pairs,
        multiply_by_n,
        partitioning=partitioning,
        max_workers=2,
        kwargs={"n": 2.0},
    )
    results_list = list(results_iter)

    assert len(results_list) == 4
    sorted_results = sorted(results_list, key=lambda p: p.x[0].as_py())

    combined = qv.concatenate(sorted_results)

    assert combined.x.to_pylist() == [2, 4, 6, 8, 10, 12, 14, 16]
    assert combined.y.to_pylist() == [16, 14, 12, 10, 8, 6, 4, 2]


def raise_exception(pairs: Pair):
    raise ValueError("foo")


def test_execute_parallel_raise_exception():
    pairs = Pair.from_kwargs(
        x=[1, 2, 3, 4, 5, 6, 7, 8],
        y=[8, 7, 6, 5, 4, 3, 2, 1],
    )
    partitioning = qv_shmem.ChunkedPartitioning(chunk_size=2)

    results_iter = qv_shmem.execute_parallel(pairs, raise_exception, partitioning=partitioning, max_workers=2)

    with pytest.raises(ValueError):
        list(results_iter)


def crash_process(pairs: Pair):
    os.kill(os.getpid(), signal.SIGKILL)


def test_execute_parallel_crash():
    pairs = Pair.from_kwargs(
        x=[1, 2, 3, 4, 5, 6, 7, 8],
        y=[8, 7, 6, 5, 4, 3, 2, 1],
    )
    partitioning = qv_shmem.ChunkedPartitioning(chunk_size=2)

    results_iter = qv_shmem.execute_parallel(pairs, crash_process, partitioning=partitioning, max_workers=2)

    with pytest.raises(concurrent.futures.process.BrokenProcessPool):
        list(results_iter)
