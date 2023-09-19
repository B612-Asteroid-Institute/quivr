import numpy as np

import quivr as qv


class Pair(qv.Table):
    x = qv.Float64Column()
    y = qv.Float64Column()

    def pairwise_product(self):
        return self.x.to_numpy() * self.y.to_numpy()

    def min_pairwise_product(self):
        return self.pairwise_product().min()


def test_parallel_apply_closure():
    ps = Pair.from_kwargs(
        x=np.arange(10000, dtype=np.float64),
        y=np.arange(10000, dtype=np.float64),
    )

    enclosed = 2

    def multiply(p):
        return p.min_pairwise_product() * enclosed

    results = ps.parallel_apply(
        multiply,
        chunk_size=1000,
    )

    results = list(sorted(results))
    assert len(results) == 10
    assert results[0] == 0
    assert results[1] == 1000 * 1000 * 2
    assert results[2] == 2000 * 2000 * 2
    assert results[3] == 3000 * 3000 * 2
    assert results[4] == 4000 * 4000 * 2
    assert results[5] == 5000 * 5000 * 2
    assert results[6] == 6000 * 6000 * 2
    assert results[7] == 7000 * 7000 * 2
    assert results[8] == 8000 * 8000 * 2
    assert results[9] == 9000 * 9000 * 2


def test_parallel_apply_lambda():
    ps = Pair.from_kwargs(
        x=np.arange(10000, dtype=np.float64),
        y=np.arange(10000, dtype=np.float64),
    )

    results = ps.parallel_apply(
        lambda p: p.min_pairwise_product(),
        chunk_size=1000,
    )
    results = list(sorted(results))
    assert len(results) == 10
    assert results[0] == 0
    assert results[1] == 1000 * 1000
    assert results[2] == 2000 * 2000
    assert results[3] == 3000 * 3000
    assert results[4] == 4000 * 4000
    assert results[5] == 5000 * 5000
    assert results[6] == 6000 * 6000
    assert results[7] == 7000 * 7000
    assert results[8] == 8000 * 8000
    assert results[9] == 9000 * 9000


def test_parallel_apply_method():
    ps = Pair.from_kwargs(
        x=np.arange(10000, dtype=np.float64),
        y=np.arange(10000, dtype=np.float64),
    )

    results = ps.parallel_apply(
        Pair.min_pairwise_product,
        chunk_size=1000,
    )
    results = list(sorted(results))
    assert len(results) == 10
    assert results[0] == 0
    assert results[1] == 1000 * 1000
    assert results[2] == 2000 * 2000
    assert results[3] == 3000 * 3000
    assert results[4] == 4000 * 4000
    assert results[5] == 5000 * 5000
    assert results[6] == 6000 * 6000
    assert results[7] == 7000 * 7000
    assert results[8] == 8000 * 8000
    assert results[9] == 9000 * 9000


class TableWithAttribute(qv.Table):
    x = qv.Float64Column()
    y = qv.Float64Column()

    name = qv.StringAttribute()


class Wrapper(qv.Table):
    twa = TableWithAttribute.as_column()
    id = qv.StringAttribute()


def test_parallel_apply_attributes():
    twa = TableWithAttribute.from_kwargs(
        x=np.arange(10000, dtype=np.float64),
        y=np.arange(10000, dtype=np.float64),
        name="foo",
    )
    w = Wrapper.from_kwargs(
        twa=twa,
        id="bar",
    )

    results = w.parallel_apply(
        lambda p: p.twa.name,
        chunk_size=1000,
    )
    results = list(results)
    assert len(results) == 10
    assert all(r == "foo" for r in results)
