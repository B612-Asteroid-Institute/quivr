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
    

.. autoclass:: StringAttribute
      
.. autoclass:: FloatAttribute
      
.. autoclass:: IntAttribute

	       
