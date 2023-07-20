import numpy as np
import pyarrow as pa
import pytest

from quivr import Float64Column, Int64Column, StringColumn, Table
from quivr.linkage import Linkage, composite_array


class Observers(Table):
    code = StringColumn(nullable=False)
    x = Float64Column()
    y = Float64Column()
    z = Float64Column()


class Ephemeris(Table):
    orbit_id = StringColumn(nullable=False)
    observer_code = StringColumn(nullable=True)
    ra = Float64Column()
    dec = Float64Column()


def test_linkage_indexing():
    observers = Observers.from_kwargs(
        code=["I41", "W84", "807"],
        x=[1, 2, 3],
        y=[4, 5, 6],
        z=[7, 8, 9],
    )

    ephems = Ephemeris.from_kwargs(
        orbit_id=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        observer_code=["I41", "I41", "I41", "W84", "W84", "W84", "807", "807", "807"],
        ra=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        dec=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    linkage = Linkage(
        left_table=observers,
        right_table=ephems,
        left_keys=observers.code,
        right_keys=ephems.observer_code,
    )

    have_i41 = linkage["I41"]
    assert len(have_i41) == 2
    have_i41_observers, have_i41_ephems = have_i41
    assert have_i41_observers == Observers.from_kwargs(
        code=["I41"],
        x=[1],
        y=[4],
        z=[7],
    )
    assert have_i41_ephems == Ephemeris.from_kwargs(
        orbit_id=["1", "2", "3"],
        observer_code=["I41", "I41", "I41"],
        ra=[1, 2, 3],
        dec=[1, 2, 3],
    )


def test_linkage_iteration():
    observers = Observers.from_kwargs(
        code=["I41", "W84", "807"],
        x=[1, 2, 3],
        y=[4, 5, 6],
        z=[7, 8, 9],
    )

    ephems = Ephemeris.from_kwargs(
        orbit_id=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        observer_code=["I41", "I41", "I41", "W84", "W84", "W84", "807", "807", "807"],
        ra=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        dec=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    linkage = Linkage(
        left_table=observers,
        right_table=ephems,
        left_keys=observers.code,
        right_keys=ephems.observer_code,
    )

    for key, obs, eph in linkage:
        assert len(obs) == 1
        assert len(eph) == 3
        assert pa.compute.all(pa.compute.equal(eph.observer_code, key)).as_py()
        assert pa.compute.all(pa.compute.equal(obs.code, key)).as_py()


def test_integer_linkage():
    class LeftSide(Table):
        id = Int64Column(nullable=False)
        x = Float64Column()

    class RightSide(Table):
        id = Int64Column(nullable=False)
        leftside_id = Int64Column()

    left = LeftSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        x=[1, 2, 3, 4, 5],
    )
    right = RightSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        leftside_id=[1, 1, 1, 2, 2],
    )
    link = Linkage(left, right, left.id, right.leftside_id)

    # Bit counterintuitive, but the linkage provides empty RightSide
    # tables for leftside_id=3, 4, 5.
    assert len(link) == 5

    have_left_1, have_right_1 = link[1]
    assert have_left_1 == LeftSide.from_kwargs(
        id=[1],
        x=[1],
    )
    assert have_right_1 == RightSide.from_kwargs(
        id=[1, 2, 3],
        leftside_id=[1, 1, 1],
    )

    have_left_2, have_right_2 = link[2]
    assert have_left_2 == LeftSide.from_kwargs(
        id=[2],
        x=[2],
    )
    assert have_right_2 == RightSide.from_kwargs(
        id=[4, 5],
        leftside_id=[2, 2],
    )

    have_left_3, have_right_3 = link[3]
    assert have_left_3 == LeftSide.from_kwargs(
        id=[3],
        x=[3],
    )
    assert have_right_3 == RightSide.empty()

    have_left_4, have_right_4 = link[4]
    assert have_left_4 == LeftSide.from_kwargs(
        id=[4],
        x=[4],
    )
    assert have_right_4 == RightSide.empty()

    have_left_5, have_right_5 = link[5]
    assert have_left_5 == LeftSide.from_kwargs(
        id=[5],
        x=[5],
    )
    assert have_right_5 == RightSide.empty()


def test_composite_keys():
    class Pair(Table):
        x = Int64Column(nullable=False)
        y = Int64Column(nullable=False)

    class LeftSide(Table):
        id = Int64Column(nullable=False)
        pairs = Pair.as_column()

        def keys(self):
            return composite_array(self.id, self.pairs.x, self.pairs.y)

    class RightSide(Table):
        id = Int64Column(nullable=False)
        leftside_id = Int64Column()
        pairs = Pair.as_column()

        def keys(self):
            return composite_array(self.leftside_id, self.pairs.x, self.pairs.y)

    left = LeftSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        pairs=Pair.from_kwargs(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5]),
    )
    right = RightSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        leftside_id=[1, 1, 1, 2, 2],
        pairs=Pair.from_kwargs(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5]),
    )
    link = Linkage(left, right, left.keys(), right.keys())

    k1 = left.keys()[0]
    v = link[k1]
    assert v[0] == left[0]
    assert v[1] == right[0]


@pytest.mark.benchmark(group="linkage-creation")
@pytest.mark.parametrize("right_table_size", [100, 1000, 100000], ids=lambda x: f"right={x}")
@pytest.mark.parametrize("left_table_size", [10, 100, 1000], ids=lambda x: f"left={x}")
def test_benchmark_linkage_creation(benchmark, left_table_size, right_table_size):
    unique_observers = np.arange(left_table_size).astype(str)
    observers = Observers.from_kwargs(
        code=unique_observers,
        x=np.ones(left_table_size),
        y=np.ones(left_table_size),
        z=np.ones(left_table_size),
    )

    ephems = Ephemeris.from_kwargs(
        orbit_id=np.arange(right_table_size).astype(str),
        observer_code=np.random.choice(unique_observers, size=right_table_size),
        ra=np.ones(right_table_size),
        dec=np.ones(right_table_size),
    )

    benchmark(lambda: Linkage(observers, ephems, observers.code, ephems.observer_code))


@pytest.mark.benchmark(group="linkage-iteration")
@pytest.mark.parametrize("right_table_size", [100, 1000, 100000], ids=lambda x: f"right={x}")
@pytest.mark.parametrize("left_table_size", [10, 100, 1000], ids=lambda x: f"left={x}")
def test_benchmark_linkage_iteration(benchmark, left_table_size, right_table_size):
    unique_observers = np.arange(left_table_size).astype(str)
    observers = Observers.from_kwargs(
        code=unique_observers,
        x=np.ones(left_table_size),
        y=np.ones(left_table_size),
        z=np.ones(left_table_size),
    )

    ephems = Ephemeris.from_kwargs(
        orbit_id=np.arange(right_table_size).astype(str),
        observer_code=np.random.choice(unique_observers, size=right_table_size),
        ra=np.ones(right_table_size),
        dec=np.ones(right_table_size),
    )

    linkage = Linkage(observers, ephems, observers.code, ephems.observer_code)

    benchmark(lambda: _noop_iterate(linkage))


def _noop_iterate(iterator):
    for _ in iterator:
        pass
