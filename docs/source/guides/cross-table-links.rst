Linkages: Working with Multiple Tables
======================================

.. currentmodule:: quivr

.. note::

   See also: :ref:`linkage-example` for a worked example of using
   linkages.

It is common with tabular data to have relationships between different
tables. :obj:`quivr.Linkage` provides a mechanism for representing
these relationships in a compact and efficient way.

Linkages can represent one-to-one, many-to-one, or many-to-many
relationships. They're most efficient in the one-to-one scenario, but
all types of relations work.

Creating Linkages
-----------------

Linkages are created by using the :obj:`quivr.Linkage`
constructor. You must pass in two tables, termed the "left" and
"right" tables, and two :obj:`pyarrow.Array`\ s of keys ("left_keys"
and "right_keys") that specify how the tables are related.

It's worth emphasizing that the key arrays do not need to directly be
columns on the tables. They could be computed from column data, or
even be arbitrary data.

But there are some **requirements of the keys**:

1. The left keys need to be the same length as the left
   table. Likewise, the right keys must be the same length as the
   right table.
2. There must not be any nulls in the left keys or the right keys.
3. The key arrays must be identically typed (e.g. both ``uint8``, both
   ``int32``, both ``string``, etc.)

There are some things that are *not* requirements which should be
pointed out explicitly:

- The keys don't need to be unique. If there are multiple rows with
  the same key, then the linkage will return all of them.
- The keys don't need to be sorted.
- The left and right keys don't need to contain the same values. If
  they're entirely different, then linkages will only return results
  for the side that has a matching key.


Operations on Linkages
----------------------

Once you have a linkage, you get four basic operations:

1. :obj:`Linkage.select` - Select rows from both tables that match a
   given key.
2. :obj:`Linkage.iterate` - Iterate over all the unique keys across
   both tables, yielding the key along with the matching rows from the
   left and right tables.
3. :obj:`Linkage.select_left` - Select rows from the left table that
   match a given key.
4. :obj:`Linkage.select_right` - Select rows from the right table that
   match a given key.

Each of these methods returns :obj:`Table` objects directly, which are
built as sliced views of the original underlying data. This means that
they're very efficient, and don't require any copying of data.

Since the data inside of Tables are immutable, you can't modify the
results of these operations. But you can use them as inputs to other
operations, or write them to disk, or whatever else you want to do
with them.

If a key is entirely absent from one of the tables, you'll get an
empty table (as produced with :meth:`Table.empty`).

In addition, linkages keep references to their original tables and
keys. You can access them with the ``Linkage.left_table``,
``Linkage.right_table``, ``Linkage.left_keys``, and
``Linkage.right_keys`` attributes.

Sorted Iteration
----------------

Linkages are not sorted, but you can iterate over them in sorted order
by using the original key arrays.

For example, if you have a linkage ``lnk`` and you want to iterate
over it in order of increasing ID, you can do this:

.. code-block:: python

   import pyarrow as pa
   keys = pa.concat_arrays([lnk.left_keys, lnk.right_keys]).unique().sort()
   for k in keys:
       left, right = lnk.select(k)
       # do something with the left and right tables

If you happen to know that the keys are already unique, or already
sorted, you might be able to skip some of those calls, or you might
even be able to only use one of the key arrays.


Slicing and Filtering
---------------------

The simplest way to do slicing and filtering is to slice and filter
the original table. This works if you're providing one of the Table's
columns directly as the key array, which is the most common case.

For example, if I have a linkage ``link``, keyed by the ``id`` column
of its left table, and its left table has a field named ``value`` that
I'd like to filter on, I could use :meth:`Table.where`:

.. code-block:: python

   import pyarrow as pa
   import pyarrow.compute as pc

   # Filter the left table to positive value
   left_positive = lnk.left_table.where(pc.field("value") > 0)

   for id in left_positive.id:
       left, right = lnk.select(id)
       # do something with the left and right tables

Similarly, to slice (for example, to get the first 10 rows), you can
use slicing syntax on the table:

.. code-block:: python

   left_first_10 = lnk.left_table[:10]

   for id in left_first_10.id:
       left, right = lnk.select(id)
       # do something with the left and right tables

If you're using a key array that isn't a column on the table, then
you'll need to do some more sophisticated bookkeeping, but the basic
idea should be the same: generate an iterable of the keys you want,
and then use the linkage to select the rows that match.


Linking on multiple arrays
--------------------------

Linkages can be created on multiple arrays of data. See the
:obj:`MultiKeyLinkage` documentation for thorough details.

The basic idea is that instead of providing left and right key arrays,
you provide dictionaries of key arrays. The keys of the dictionaries
are used to identify the arrays, and the values are the arrays
themselves.

The dictionaries must have the same keys, and the arrays for a
particular key must be identically typed.

Lookups get a bit trickier with multiple keys, since you'll need to
construct a composite key with just the right shape to match the
linkage. The :meth:`MultiKeyLinkage.key` method is provided to help
with this.
