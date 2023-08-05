.. _attributes:

Table Attributes
================

.. currentmodule:: quivr

Attributes are descriptors attached to :class:`Table` classes. They
describe scalar values associated with an entire table of
data. Instead of repeating the value for every row, attributes
efficiently store it just once.

Attributes are preserved through serialization, because they are
stored in table metadata.

Example usage:

.. code-block:: py

   import quivr as qv

   class Observations(qv.Table):
       x = qv.Float64Column()
       y = qv.Float64Column()
       z = qv.Float64Column()

       reference_frame = qv.StringAttribute(default="earth")


    obs = Observations.from_data(
       x=[1, 2, 3],
       y=[4, 5, 6]
       z=[7, 8, 9],
       reference_frame="mars"
    )

    print(obs.reference_frame)  # prints 'mars'

Mutability
----------

By default, attributes are immutable: they can be set when a Table
instance is created, but they cannot be modified. If you need an
attribute to be mutable, you can do so by passing ``mutable=True`` to
the Attribute constructor:

.. code-block:: py

   class Dataset(qv.Table):
       measurements = qv.Int64Column()
       source = qv.StringAttribute(mutable=True)

   p = Dataset.from_data(measurements=[1, 2, 3], source="earth")
   p.source = "mars"  # only if p.source is mutable!

Mutability can have confusing behavior when working with nested tables
(like with :meth:`Table.as_column`). When you chain accessors to
access a sub-table, your changes to that sub-table's attributes won't
be reflected back in the original table unless you explicitly
re-attach the sub-table to the parent. This is easier to explain with code:

.. code-block:: py

   class Dataset(qv.Table):
       measurements = qv.Int64Column()
       source = qv.StringAttribute(mutable=True)

   class Database(qv.Table):
       datasets = Dataset.as_column()

   d = Database.from_data(
       datasets=Dataset.from_data(
           measurements=[1, 2, 3],
           source="earth",
       )
   )
   # Don't do this:
   d.datasets.source = "mars"
   print(d.datasets.source)  # prints "earth" - the value is unchanged!

   # Do this instead:
   ds = d.datasets
   ds.source = "mars"
   d.datasets = ds
   print(d.datasets.source)  # Now prints "mars"


Reference
---------

.. autoclass:: StringAttribute

.. autoclass:: FloatAttribute

.. autoclass:: IntAttribute
