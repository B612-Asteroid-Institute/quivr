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

   from quivr import Table, StringAttribute, Float64Column

   class Observations(Table):
       x = Float64Column()
       y = Float64Column()
       z = Float64Column()

       reference_frame = StringAttribute(default="earth")


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

   class Dataset(Table):
       measurements = Int64Column()
       source = StringAttribute(mutable=True)

   p = Dataset.from_data(measurements=[1, 2, 3], source="earth")
   p.source = "mars"  # only if p.source is mutable!

Reference
---------

.. autoclass:: StringAttribute
      
.. autoclass:: FloatAttribute
      
.. autoclass:: IntAttribute

	       
