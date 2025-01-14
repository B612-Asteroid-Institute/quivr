import numpy as np
import pyarrow as pa
import pytest

import quivr as qv

from .test_tables import Pair, TableWithAttributes, TableWithDefaultAttributes, Wrapper


def test_concatenate():
    xs1 = pa.array([1, 2, 3], pa.int64())
    ys1 = pa.array([4, 5, 6], pa.int64())
    pair1 = Pair.from_kwargs(
        x=xs1,
        y=ys1,
    )

    xs2 = pa.array([11, 22, 33], pa.int64())
    ys2 = pa.array([44, 55, 66], pa.int64())
    pair2 = Pair.from_kwargs(
        x=xs2,
        y=ys2,
    )

    have = qv.concatenate([pair1, pair2])
    assert len(have) == 6
    np.testing.assert_array_equal(have.x, [1, 2, 3, 11, 22, 33])


def test_concatenate_nested():
    xs1 = pa.array([1, 2, 3], pa.int64())
    ys1 = pa.array([4, 5, 6], pa.int64())
    pairs1 = pa.StructArray.from_arrays([xs1, ys1], fields=list(Pair.schema))
    ids1 = pa.array(["v1", "v2", "v3"], pa.string())
    w1 = Wrapper.from_kwargs(
        pair=pairs1,
        id=ids1,
    )

    xs2 = pa.array([11, 22, 33], pa.int64())
    ys2 = pa.array([44, 55, 66], pa.int64())
    pairs2 = pa.StructArray.from_arrays([xs2, ys2], fields=list(Pair.schema))
    ids2 = pa.array(["v4", "v5", "v6"], pa.string())
    w2 = Wrapper.from_kwargs(
        pair=pairs2,
        id=ids2,
    )

    have = qv.concatenate([w1, w2])
    assert len(have) == 6
    np.testing.assert_array_equal(have.pair.x, [1, 2, 3, 11, 22, 33])
    np.testing.assert_array_equal(have.id, ["v1", "v2", "v3", "v4", "v5", "v6"])


def test_concatenate_empty_list():
    with pytest.raises(ValueError, match="No values to concatenate"):
        qv.concatenate([])


def test_concatenate_empty_tables():
    t1 = Pair.empty()
    t2 = Pair.empty()
    have = qv.concatenate([t1, t2])
    assert len(have) == 0

    have = qv.concatenate([t1])
    assert len(have) == 0


def test_concatenate_no_validate():

    class OtherTable(qv.Table):
        y = qv.Int64Column(validator=qv.le(0))

    class ValidationTable(qv.Table):
        x = qv.Int64Column(validator=qv.ge(0))
        subtable = OtherTable.as_column(nullable=True)

    valid = ValidationTable.from_kwargs(
        x=[1], subtable=OtherTable.from_kwargs(y=[-1], validate=False), validate=False
    )
    invalid_x = ValidationTable.from_kwargs(x=[-1], subtable=[None], validate=False)
    invalid_subtable = ValidationTable.from_kwargs(
        x=[1], subtable=OtherTable.from_kwargs(y=[1], validate=False), validate=False
    )

    with pytest.raises(qv.ValidationError, match="Column x failed validation"):
        qv.concatenate([invalid_x, valid])

    with pytest.raises(qv.ValidationError, match="Column y failed validation"):
        qv.concatenate([valid, invalid_subtable])

    have = qv.concatenate([invalid_x, valid], validate=False)
    assert len(have) == 2


@pytest.mark.benchmark(group="ops")
def test_benchmark_concatenate_100(benchmark):
    xs1 = pa.array([1, 2, 3], pa.int64())
    ys1 = pa.array([4, 5, 6], pa.int64())
    pair1 = Pair.from_kwargs(
        x=xs1,
        y=ys1,
    )

    xs2 = pa.array([11, 22, 33], pa.int64())
    ys2 = pa.array([44, 55, 66], pa.int64())
    pair2 = Pair.from_kwargs(
        x=xs2,
        y=ys2,
    )

    benchmark(qv.concatenate, [pair1, pair2] * 50)


def test_concatenate_different_types():
    with pytest.raises(qv.TablesNotCompatibleError, match="All tables must be the same class to concatenate"):
        qv.concatenate([Pair.empty(), Wrapper.empty()])


def test_concatenate_different_attrs():
    t1 = TableWithAttributes.from_kwargs(x=[1], y=[2], attrib="foo")
    t2 = TableWithAttributes.from_kwargs(x=[3], y=[4], attrib="bar")

    with pytest.raises(
        qv.TablesNotCompatibleError,
        match="All non-empty tables must have the same attribute values to concatenate",
    ):
        qv.concatenate([t1, t2])


def test_concatenate_default_attrs_empty():
    t1 = TableWithDefaultAttributes.empty()  # This will default to attrib="foo"
    t2 = TableWithDefaultAttributes.from_kwargs(x=[3], y=[4], attrib="bar")
    t3 = TableWithDefaultAttributes.from_kwargs(x=[3], y=[4], attrib="bar")
    have = qv.concatenate([t1, t2, t3])
    assert have.attrib == "bar"


def test_concatenate_same_attrs():
    t1 = TableWithAttributes.from_kwargs(x=[1], y=[2], attrib="foo")
    t2 = TableWithAttributes.from_kwargs(x=[3], y=[4], attrib="foo")
    have = qv.concatenate([t1, t2])
    assert have.attrib == "foo"


def test_concatenate_no_values():
    with pytest.raises(ValueError, match="No values to concatenate"):
        qv.concatenate([])


def test_concatenate_empty_tables_preserve_attributes():
    class TableWithAttrs(qv.Table):
        x = qv.Int64Column()
        y = qv.Int64Column()
        name = qv.StringAttribute(default="default")
        id = qv.IntAttribute(default=0)

    # Create two empty tables with non-default attributes
    t1 = TableWithAttrs.from_kwargs(x=[], y=[], name="foo", id=1)
    t2 = TableWithAttrs.from_kwargs(x=[], y=[], name="bat", id=3)
    
    # Concatenate them and verify we get an empty table with the same attributes
    have = qv.concatenate([t1, t2])
    assert len(have) == 0
    assert have.name == "foo"  # Not "default"
    assert have.id == 1  # Not 0

    # Also verify it works when concatenating with a non-empty table
    # Attributes should be preserved from the non-empty tables
    t3 = TableWithAttrs.from_kwargs(x=[1], y=[2], name="bar", id=2)
    have = qv.concatenate([t1, t2, t3])
    assert len(have) == 1
    assert have.name == "bar"
    assert have.id == 2

