import numpy as np
import pyarrow as pa
import pytest

import quivr as qv


class Observers(qv.Table):
    code = qv.StringColumn(nullable=False)
    x = qv.Float64Column()
    y = qv.Float64Column()
    z = qv.Float64Column()


class Ephemeris(qv.Table):
    orbit_id = qv.StringColumn(nullable=False)
    observer_code = qv.StringColumn(nullable=True)
    ra = qv.Float64Column()
    dec = qv.Float64Column()


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

    linkage = qv.Linkage(
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

    linkage = qv.Linkage(
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
    class LeftSide(qv.Table):
        id = qv.Int64Column(nullable=False)
        x = qv.Float64Column()

    class RightSide(qv.Table):
        id = qv.Int64Column(nullable=False)
        leftside_id = qv.Int64Column()

    left = LeftSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        x=[1, 2, 3, 4, 5],
    )
    right = RightSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        leftside_id=[1, 1, 1, 2, 2],
    )
    link = qv.Linkage(left, right, left.id, right.leftside_id)

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


class Pair(qv.Table):
    x = qv.Int64Column(nullable=False)
    y = qv.Int64Column(nullable=False)


class LeftSide(qv.Table):
    id = qv.Int64Column(nullable=False)
    pairs = Pair.as_column()


class RightSide(qv.Table):
    id = qv.StringColumn(nullable=False)
    leftside_id = qv.Int64Column()
    pairs = Pair.as_column()


class TestMultiKeyLinkages:
    left = LeftSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        pairs=Pair.from_kwargs(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5]),
    )
    right = RightSide.from_kwargs(
        id=["a", "b", "c", "d", "e"],
        leftside_id=[1, 1, 1, 2, 2],
        pairs=Pair.from_kwargs(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5]),
    )

    def test_link_composite_key(self):
        left, right = self.left, self.right
        link = qv.MultiKeyLinkage(
            left_table=left,
            right_table=right,
            left_keys={"id": left.id, "x": left.pairs.x},
            right_keys={"id": right.leftside_id, "x": right.pairs.x},
        )

        k1 = link.key(id=1, x=1)
        v = link[k1]
        assert v[0] == left[0]
        assert v[1] == right[0]

    def test_create_scalar_not_present(self):
        left, right = self.left, self.right
        link = qv.MultiKeyLinkage(
            left_table=left,
            right_table=right,
            left_keys={"id": left.id, "x": left.pairs.x},
            right_keys={"id": right.leftside_id, "x": right.pairs.x},
        )

        # Should be OK to create a scalar with values that don't exist
        k1 = link.key(id=-1, x=-1)
        v = link[k1]
        assert v[0] == left.empty()
        assert v[1] == right.empty()

    def test_create_scalar_key_invalid(self):
        left, right = self.left, self.right
        link = qv.MultiKeyLinkage(
            left_table=left,
            right_table=right,
            left_keys={"id": left.id, "x": left.pairs.x},
            right_keys={"id": right.leftside_id, "x": right.pairs.x},
        )

        # Can't make one with a key that doesn't exist
        with pytest.raises(ValueError):
            link.key(id=1, x=2, z=2)

        # Can't make one without setting all keys
        with pytest.raises(ValueError):
            link.key(id=1)

        # Can't make one with wrong dtypes
        with pytest.raises(pa.ArrowInvalid):
            link.key(id=1, x="1")

    def test_create_linkage_error_cases(self):
        left, right = self.left, self.right

        # Can't make a linkage with empty key sets
        with pytest.raises(ValueError, match="Left and right key dictionaries must not be empty"):
            qv.MultiKeyLinkage(left, right, {}, {})

        # Can't make a linkage with different keys
        with pytest.raises(ValueError, match="Left and right key dictionaries must have the same keys"):
            qv.MultiKeyLinkage(left, right, {"id": left.id}, {"foo": right.id})

        # Can't make a linkage with different key dtypes
        with pytest.raises(TypeError, match="Left key id and right key id must have the same type"):
            qv.MultiKeyLinkage(
                left,
                right,
                left_keys={"id": left.id, "x": left.pairs.x},
                right_keys={"id": right.id, "x": right.pairs.x},
            )

        # Can't make a linkage with nulls in the key
        with pytest.raises(ValueError, match="Right key x must not contain null values"):
            x_with_null = pa.array([1, 2, 3, 4, None], type=pa.int64())
            qv.MultiKeyLinkage(
                left,
                right,
                left_keys={"id": left.id, "x": left.pairs.x},
                right_keys={"id": right.leftside_id, "x": x_with_null},
            )

        # Can't make a linkage with different key lengths
        with pytest.raises(ValueError, match="Left key x must have the same length as the left table"):
            x_short = pa.array([1, 2, 3, 4], type=pa.int64())
            qv.MultiKeyLinkage(
                left,
                right,
                left_keys={"id": left.id, "x": x_short},
                right_keys={"id": right.leftside_id, "x": right.pairs.x},
            )
        with pytest.raises(ValueError, match="Right key x must have the same length as the right table"):
            x_short = pa.array([1, 2, 3, 4], type=pa.int64())
            qv.MultiKeyLinkage(
                left,
                right,
                left_keys={"id": left.id, "x": left.pairs.x},
                right_keys={"id": right.leftside_id, "x": x_short},
            )


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

    benchmark(lambda: qv.Linkage(observers, ephems, observers.code, ephems.observer_code))


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

    linkage = qv.Linkage(observers, ephems, observers.code, ephems.observer_code)

    benchmark(lambda: _noop_iterate(linkage))


def _noop_iterate(iterator):
    for _ in iterator:
        pass


def test_access_keys_via_linkage():
    class LeftSide(qv.Table):
        id = qv.Int64Column(nullable=False)
        x = qv.Float64Column()

    class RightSide(qv.Table):
        id = qv.Int64Column(nullable=False)
        leftside_id = qv.Int64Column()

    left = LeftSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        x=[1, 2, 3, 4, 5],
    )
    right = RightSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        leftside_id=[1, 1, 1, 2, 2],
    )
    link = qv.Linkage(left, right, left.id, right.leftside_id)

    assert link.left_keys == left.id
    assert link.right_keys == right.leftside_id
    assert link.left_keys != link.right_keys


def test_linkage_uses_correct_dtypes_in_select():
    class LeftSide(qv.Table):
        id = qv.UInt32Column(nullable=False)
        x = qv.Float64Column()

    class RightSide(qv.Table):
        id = qv.Int64Column(nullable=False)
        leftside_id = qv.UInt32Column()

    left = LeftSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        x=[1, 2, 3, 4, 5],
    )
    right = RightSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        leftside_id=[1, 1, 1, 2, 2],
    )
    link = qv.Linkage(left, right, left.id, right.leftside_id)

    assert link.left_keys.type == pa.uint32()
    assert link.right_keys.type == pa.uint32()

    # Using a Python literal should work
    have_left, have_right = link.select(1)
    assert len(have_left) == 1
    assert len(have_right) == 3


def test_combine_linkages():
    class LeftSide(qv.Table):
        id = qv.Int64Column(nullable=False)
        x = qv.Float64Column()

    class RightSide(qv.Table):
        id = qv.Int64Column(nullable=False)
        leftside_id = qv.Int64Column()

    left1 = LeftSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        x=[1, 2, 3, 4, 5],
    )
    right1 = RightSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        leftside_id=[1, 1, 1, 2, 3],
    )
    link1 = qv.Linkage(left1, right1, left1.id, right1.leftside_id)

    left2 = LeftSide.from_kwargs(
        id=[6, 7, 8, 9, 10],
        x=[6, 7, 8, 9, 10],
    )
    right2 = RightSide.from_kwargs(
        id=[6, 7, 8, 9, 10],
        leftside_id=[3, 3, 3, 4, 4],
    )
    link2 = qv.Linkage(left2, right2, left2.id, right2.leftside_id)

    combined = qv.combine_linkages([link1, link2])
    assert len(combined) == 10

    have_left, have_right = combined.select(1)
    assert len(have_left) == 1
    assert len(have_right) == 3

    have_left, have_right = combined.select(3)
    assert len(have_left) == 1
    assert len(have_right) == 4


def test_combine_linkages_with_different_keys():
    """
    Combining linkages with different key types should raise an error.
    """

    class LeftSide(qv.Table):
        id = qv.Int64Column(nullable=False)
        x = qv.Float64Column()

    class RightSide(qv.Table):
        id = qv.Int64Column(nullable=False)
        leftside_id = qv.Int64Column()
        y = qv.Float64Column()

    left1 = LeftSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        x=[1, 2, 3, 4, 5],
    )
    right1 = RightSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        leftside_id=[1, 1, 1, 2, 3],
        y=[1, 2, 3, 4, 5],
    )
    link1 = qv.Linkage(left1, right1, left1.id, right1.leftside_id)

    left2 = LeftSide.from_kwargs(
        id=[6, 7, 8, 9, 10],
        x=[6, 7, 8, 9, 10],
    )
    right2 = RightSide.from_kwargs(
        id=[6, 7, 8, 9, 10],
        leftside_id=[3, 3, 3, 4, 4],
        y=[6, 7, 8, 9, 10],
    )
    link2 = qv.Linkage(left2, right2, left2.x, right2.y)

    with pytest.raises(qv.LinkageCombinationError):
        qv.combine_linkages([link1, link2])


def test_combine_linkages_with_different_tables():
    """
    Combining linkages with different table types should error.
    """

    class LeftSide(qv.Table):
        id = qv.Int64Column(nullable=False)
        x = qv.Float64Column()

    class RightSide1(qv.Table):
        id = qv.Int64Column(nullable=False)
        leftside_id = qv.Int64Column()

    class RightSide2(qv.Table):
        id = qv.Int64Column(nullable=False)
        leftside_id = qv.Int64Column()

    left1 = LeftSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        x=[1, 2, 3, 4, 5],
    )
    right1 = RightSide1.from_kwargs(
        id=[1, 2, 3, 4, 5],
        leftside_id=[1, 1, 1, 2, 3],
    )
    link1 = qv.Linkage(left1, right1, left1.id, right1.leftside_id)

    left2 = LeftSide.from_kwargs(
        id=[6, 7, 8, 9, 10],
        x=[6, 7, 8, 9, 10],
    )
    right2 = RightSide2.from_kwargs(
        id=[6, 7, 8, 9, 10],
        leftside_id=[3, 3, 3, 4, 4],
    )
    link2 = qv.Linkage(left2, right2, left2.id, right2.leftside_id)

    with pytest.raises(qv.LinkageCombinationError):
        qv.combine_linkages([link1, link2])


def test_combine_multi_key_linkages():
    left1 = LeftSide.from_kwargs(
        id=[1, 2, 3, 4, 5],
        pairs=Pair.from_kwargs(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5]),
    )
    right1 = RightSide.from_kwargs(
        id=["a", "b", "c", "d", "e"],
        leftside_id=[1, 2, 1, 2, 2],
        pairs=Pair.from_kwargs(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5]),
    )
    link1 = qv.MultiKeyLinkage(
        left_table=left1,
        right_table=right1,
        left_keys={"id": left1.id, "x": left1.pairs.x},
        right_keys={"id": right1.leftside_id, "x": right1.pairs.x},
    )

    left2 = LeftSide.from_kwargs(
        id=[6, 7, 8, 9, 10],
        pairs=Pair.from_kwargs(x=[6, 7, 8, 9, 10], y=[6, 7, 8, 9, 10]),
    )
    right2 = RightSide.from_kwargs(
        id=["f", "g", "h", "i", "j"],
        leftside_id=[2, 3, 6, 7, 8],
        pairs=Pair.from_kwargs(x=[2, 3, 6, 7, 8], y=[6, 7, 8, 9, 10]),
    )
    link2 = qv.MultiKeyLinkage(
        left_table=left2,
        right_table=right2,
        left_keys={"id": left2.id, "x": left2.pairs.x},
        right_keys={"id": right2.leftside_id, "x": right2.pairs.x},
    )

    combined = qv.combine_multilinkages([link1, link2])

    k1 = combined.key(id=2, x=2)
    have_left, have_right = combined.select(k1)
    assert len(have_left) == 1
    assert len(have_right) == 2
