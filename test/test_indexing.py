import numpy as np
import pyarrow as pa

from quivr.attributes import StringAttribute
from quivr.columns import Int64Column, StringColumn, ForeignKeyColumn
from quivr.indexing import StringIndex
from quivr.tables import Table


class TableWithString(Table):
    id = Int64Column()
    name = StringColumn()
    value = Int64Column()


def test_indexing():
    table = TableWithString.from_arrays([pa.array([1, 2, 3]), pa.array(["a", "b", "c"]), pa.array([4, 5, 6])])
    index = StringIndex(table, "name")
    assert len(index.lookup("a")) == 1
    np.testing.assert_array_equal(index.lookup("a").id, [1])
    assert len(index.lookup("b")) == 1
    np.testing.assert_array_equal(index.lookup("b").id, [2])


def test_indexing_duplicate():
    table = TableWithString.from_arrays([pa.array([1, 2, 3]), pa.array(["a", "a", "c"]), pa.array([4, 5, 6])])
    index = StringIndex(table, "name")
    assert len(index.lookup("a")) == 2
    np.testing.assert_array_equal(index.lookup("a").id, [1, 2])
    assert index.lookup("b") is None


class Person(Table):
    id = StringColumn()  # implies: not nullable, not updateable
    name = StringColumn()
    age = Int64Column()

    attrib = StringAttribute()


class Team(Table):
    name = StringColumn()
    people = ForeignKeyColumn(Person, on=Person.id)  # warns if other column is nullable or updateable
    


def test_foreign_key():
    persons = Person.from_kwargs(
        id=["1", "2", "3"],
        name=["Alice", "Bob", "Charlie"],
        age=[20, 30, 40],
        attrib="foo",
    )

    teams = Team.from_kwargs(
        name=["Team 1", "Team 2"],
        people=Team.people.reference(persons, [["1", "2"], ["3", "1"]]),
    )

    joined = list(teams.people)

    assert len(joined) == 2
    assert len(joined[0]) == 2
    assert len(joined[1]) == 2

    # note that results are unsorted. so we sort by id to make the test consistent.
    want1 = Person.from_kwargs(
        id=["1", "2"],
        name=["Alice", "Bob"],
        age=[20, 30],
        attrib="foo",
    )
    want2 = Person.from_kwargs(
        id=["1", "3"],
        name=["Alice", "Charlie"],
        age=[20, 40],
        attrib="foo",
    )
    assert joined[0].sort_by("id") == want1
    assert joined[1].sort_by("id") == want2
    
    # updates to referenced columns are reflected
    persons.age = [21, 30, 40]
    joined = list(teams.people)
    want1 = Person.from_kwargs(
        id=["1", "2"],
        name=["Alice", "Bob"],
        age=[21, 30],
        attrib="foo",
    )
    want2 = Person.from_kwargs(
        id=["1", "3"],
        name=["Alice", "Charlie"],
        age=[21, 40],
        attrib="foo",
    )
    assert joined[0].sort_by("id") == want1
    assert joined[1].sort_by("id") == want2
    
