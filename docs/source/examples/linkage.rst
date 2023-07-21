.. _linkage-example:

Linkages Example
================

Let's imagine we have two tables: one is a table of people, and the
other is a table of their pets. The people table has a unique
identifier for each person, and the pets table has a column that
contains the identifier of the person who owns the pet.


.. literalinclude:: ./linkages.py
   :language: python
   :lines: 1-11

For example:

+-----------------------------------+
|**Owners**                         |
+----------+------------+-----------+
|       id |       name |       age |
+==========+============+===========+
|    1     |    Bob     |    30     |
+----------+------------+-----------+
|    2     |    Sue     |    25     |
+----------+------------+-----------+
|    3     |    Joe     |    40     |
+----------+------------+-----------+
|    4     |    Mary    |    35     |
+----------+------------+-----------+
|    5     |    John    |    50     |
+----------+------------+-----------+

+----------------------------------------------------+
| **Pets**                                           |
+-----------+----------+---------------+-------------+
|     name  | owner_id |     owner_id  |     species |
+===========+==========+===============+=============+
|   Fido    |    1     |       1       |    Dog      |
+-----------+----------+---------------+-------------+
|   Spot    |    1     |       1       |    Dog      |
+-----------+----------+---------------+-------------+
| Mittens   |    2     |       2       |    Cat      |
+-----------+----------+---------------+-------------+
|  Rover    |    3     |       3       |    Dog      |
+-----------+----------+---------------+-------------+
|   Lucy    |    4     |       4       |    Dog      |
+-----------+----------+---------------+-------------+
| Whiskers  |    5     |       5       |    Cat      |
+-----------+----------+---------------+-------------+
|   Max     |    5     |       5       |    Dog      |
+-----------+----------+---------------+-------------+

       
Let's suppose we want to get the average age of people who own a
particular species of pet.

The problem linkages solve
--------------------------

One way without linkages would be to construct a new table that
merges the two tables together:

.. literalinclude:: ./linkages.py
   :language: python
   :lines: 15-17

But this table is inefficient: it contains extra copies of the owner
information, and it includes columns we won't use like the owner's
name. It also requires us to duplicate the owner information for each
pet they own.

Take a look:

+----------+------------+-----------+----------+---------------+-------------+
| owner.id | owner.name | owner.age | pet.name | pet.owner_id  | pet.species |
+==========+============+===========+==========+===============+=============+
|    1     |    Bob     |    30     |   Fido   |       1       |    Dog      |
+----------+------------+-----------+----------+---------------+-------------+
|    1     |    Bob     |    30     |   Spot   |       1       |    Dog      |
+----------+------------+-----------+----------+---------------+-------------+
|    2     |    Sue     |    25     | Mittens  |       2       |    Cat      |
+----------+------------+-----------+----------+---------------+-------------+
|    3     |    Joe     |    40     |  Rover   |       3       |    Dog      |
+----------+------------+-----------+----------+---------------+-------------+
|    4     |    Mary    |    35     |   Lucy   |       4       |    Dog      |
+----------+------------+-----------+----------+---------------+-------------+
|    5     |    John    |    50     | Whiskers |       5       |    Cat      |
+----------+------------+-----------+----------+---------------+-------------+
|    5     |    John    |    50     |   Max    |       5       |    Dog      |
+----------+------------+-----------+----------+---------------+-------------+


Building a Linkage
------------------

Instead, we'd like to link the two tables together. We can do this by using :obj:`quivr.Linkage`:

.. literalinclude:: ./linkages.py
   :language: python
   :lines: 21-33

Using the linkage
-----------------

The linkage has two main methods: :obj:`quivr.Linkage.select` and
:obj:`quivr.Linkage.iterate`. These methods are also aliased for
ergonomics: you can use ``linkage[...]`` to select rows, and ``for row
in linkage`` to iterate over rows.

Let's select the info associated with the person with ID=1:

.. literalinclude:: ./linkages.py
   :language: python
   :lines: 37-46

We can also iterate over all the groups in the linkage. This yields
out tuples of ``(id, owner, pets)`` in our case. In general, it yields
the ``(key, left_table, right_table)`` for each unique ``key`` found
in the two tables.

.. literalinclude:: ./linkages.py
   :language: python
   :lines: 50-56

One thing to notice here is that **linkages are unsorted**. The order
that the keys were provided does not necessarily correspond to the
order of the rows yielded from the linkage.

Using the linkage in computation
--------------------------------

Let's put this together to compute the average age of people who own
cats and dogs:

.. literalinclude:: ./linkages.py
   :language: python
   :lines: 60-74

There are a few things to note here.

1. Iteration yields ``Table`` instances. Each group is a table that
   contains the rows from the left and right tables that match the
   key. In this case, the left table is the ``owners`` table, and the
   right table is the ``pets`` table.

2. The ``Table`` instances are **views** into the original tables. No
   data is copied. This makes linkages very efficient.

3. Combining across multiple groups is handled with
   :obj:`quivr.concatenate`. This utility function takes a list of
   tables and concatenates them together.


