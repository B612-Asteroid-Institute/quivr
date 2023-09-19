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
