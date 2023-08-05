Basic Usage
===========

.. currentmodule:: quivr

Using ``quivr`` starts with writing some classes which describe the
data you're working with. You write a :obj:`Table` definition using
``quivr``'s :obj:`Column` and :obj:`Attribute` types to describe the
data.

Here's an example, a basic table of X-Y-Z positions:

.. code-block:: python

  import quivr as qv

  class Positions(qv.Table):
      x = qv.Float64Column()
      y = qv.Float64Column()
      z = qv.Float64Column()

This describes a table with three columns, each holding 64-bit
floating point values.

We can construct an instance of this with Python values using
:obj:`Table.from_kwargs`:

.. code-block:: python

   positions = Positions.from_kwargs(
       x=[1, 2, 3, 4],
       y=[5, 6, 7, 8],
       z=[9, 10, 11, 12],
   )
   print(positions)
   # Positions(size=4)

The ``positions`` instance in this case holds three Arrow Arrays of
data, each of length 4. You can access them by their field names as
defined on the ``Positions`` class:

.. code-block:: python

   print(positions.x)
   # [
   #   1,
   #   2,
   #   3,
   #   4
   # ]

Arrow Arrays have `an extensive API
<https://arrow.apache.org/docs/python/generated/pyarrow.Array.html>`_
which supports rich computation. Here are a few examples:

.. code-block:: python

   # Convert to a Numpy array and use Numpy methods:
   print(positions.x.to_numpy().min())
   # 1.0

   # Use pyarrow.compute:
   import pyarrow.compute as pc
   print(pc.min(positions.y))
   # 5.0

   # Multiply one array by another, element-wise
   print(pc.multiply(positions.x, positions.y))
   # [
   #   5,
   #   12,
   #   21,
   #   32
   # ]

Many other constructors are available as well to make an instance from
pandas DataFrames, Apache Parquet files, and other input sources.

Attaching methods to encapsulate logic
--------------------------------------

If you're familiar with Numpy or Pandas, you have probably written
lots of functions for encapsulating logic. For example, let's say you
want to compute distances from these position values - that is, you
want to compute ``sqrt(x**2 + y**2 + z**2)``.

With a Pandas DataFrame, you might have written it this way:

.. code-block:: python

   def distances(position_df: pd.DataFrame) -> pd.Series:
       return (
           position_df["x"] * position_df["x"]
	   + position_df["y"] * position_df["y"]
	   + position_df["z"] * position_df["z"]
       ).sqrt()

This code works, but it's tricky. The caller needs to have a dataframe
with the correct columns, correctly typed, and tehre is nothing in the
function signature that explains this. It's easy to pass in a
malformed DataFrame.

In quivr, this sort of logic is instead encapsulated with a method
defined on the ``Positionss`` class directly:

.. code-block:: python

  import pyarrow.compute as pc

  class Positions(qv.Table):
      x = qv.Float64Column()
      y = qv.Float64Column()
      z = qv.Float64Column()

      def distances(self) -> pa.Array:
          xs = pc.multiply(self.x, self.x)
	  ys = pc.multiply(self.y, self.y)
	  zs = pc.multiply(self.z, self.z)
	  sum = pc.add(xs, ys)
	  sum = pc.add(sum, zs)
	  return pc.sqrt(sum)


That method uses the `PyArrow Compute API
<https://arrow.apache.org/docs/python/compute.html>`_, which is
efficient and precise, but rather verbose. You might prefer to work
through Numpy:


.. code-block:: python

  import numpy as np

  class Positions(qv.Table):
      x = qv.Float64Column()
      y = qv.Float64Column()
      z = qv.Float64Column()

      def distances(self) -> np.array:
          x = self.x.to_numpy()
	  y = self.y.to_numpy()
	  z = self.z.to_numpy()

	  return np.sqrt(x*x + y*y + z*z)

The :meth:`pyarrow.Array.to_numpy` method is very efficient. In
typical cases, it involves no computation or memory copy, so you can
feel comfortable using ``to_numpy`` liberally. There are a few caveats
to be aware of, though:

Adding more columns
-------------------

Tables can have more columns with different types. For example, you
might want to add a ``measured_by`` column to the ``Positions`` table
to represent the entity that measured the position:

.. code-block:: python

  class Positions(qv.Table):
      x = qv.Float64Column()
      y = qv.Float64Column()
      z = qv.Float64Column()
      measured_by = qv.StringColumn()

There are many types of columns. See :ref:`columns_api` in the API
reference for the full list.

Composition
-----------

A central feature of quivr is Table composition. The idea is that
Tables can be composed together into larger entities with
sub-tables. This amplifies the benefits of attaching methods to Table
classes. These two tools combine to form a powerful language for
data-oriented programming.

To use a table compositionally, you use the :meth:`Table.as_column`
class method. For example, let's remove the ``measuered_by`` column
from ``Positions``, and instead represent that concept with a richer
``Measurers`` Table:

.. code-block:: python

   class Measurers(qv.Table):
       id = qv.UInt32Column()
       name = qv.StringColumn()

We can now make a wrapping Table called ``Measurements`` which will
store both ``Positions`` and their ``Measurers``:

.. code-block:: python

   class Measurements(qv.Table):
       position = Positions.as_column()
       measurer = Measurers.as_column()
       measured_at = qv.TimestampColumn(unit="s")

This ``Measurements`` Table now composes the ``Positions`` and
``Measurers`` tables into one structure in memory. The underlying data
layout is still tabular, and the sub-tables shared common
indexes. The following table may help visualize this:

+----------+----------+----------+-----------+-------------------+---------------------+
|position.x|position.y|position.z|measurer.id|measurer.name      |measured_at          |
+==========+==========+==========+===========+===================+=====================+
|4.1       |5.0       |6.2       |0          |Enrico Fermi       |2018-09-14T16:32:01Z |
+----------+----------+----------+-----------+-------------------+---------------------+
|4.3       |4.8       |7.1       |0          |Enrico Fermi       |2018-09-14T16:32:08Z |
+----------+----------+----------+-----------+-------------------+---------------------+
|4.2       |3.7       |7.2       |1          |Albert Einstein    |2018-09-14T16:33:21Z |
+----------+----------+----------+-----------+-------------------+---------------------+
|4.0       |6.2       |7.3       |0          |Enrico Fermi       |2018-09-14T16:35:22Z |
+----------+----------+----------+-----------+-------------------+---------------------+
|4.5       |4.4       |7.3       |1          |Albert Einstein    |2018-09-14T16:36:38Z |
+----------+----------+----------+-----------+-------------------+---------------------+

You could construct a table with that data like this:

.. code-block:: python

   measurements = Measurements.from_kwargs(
       position=Positions.from_kwargs(
           x=[4.1, 4.3, 4.2, 4.0, 4.5],
	   y=[5.0, 4.8, 3.7, 6.2, 4.4],
	   z=[6.2, 7.1, 7.2, 7.3, 7.3],
       ),
       measurer=Measurers.from_kwargs(
           id=[0, 0, 1, 0, 1],
	   name=[
	       "Enrico Fermi",
	       "Enrico Fermi",
	       "Albert Einstein",
	       "Enrico Fermi",
	       "Albert Einstein"
	   ]
       ),
       measured_at=[
           datetime.datetime(2018, 9, 14, 16, 32, 1),
           datetime.datetime(2018, 9, 14, 16, 32, 8),
           datetime.datetime(2018, 9, 14, 16, 33, 21),
           datetime.datetime(2018, 9, 14, 16, 35, 22),
           datetime.datetime(2018, 9, 14, 16, 36, 38),
       ]
   )

You can access the subtables with normal Python dot-style notation,
You'll get an instance of the Table class you defined, as you might
expect, which means you can call any attached methods:

.. code-block:: python

   print(measurements.position)
   # Positions(size=5)
   
   print(measurements.position.distances())
   # [ 8.95823643  9.58853482  9.11975877 10.37930634  9.63846461]

And of course, the wrapping class can have methods. This can let you
build sophisticated computations while managing complexity:

.. code-block:: python

   class Measurements(qv.Table):
       position = Positions.as_column()
       measurer = Measurers.as_column()
       measured_at = qv.TimestampColumn(unit="s")

       def max_distance_by_measurer(self):
           maxes = {}
           unique_ids = self.measurer.id.unique().to_numpy()
	   for id in unique_ids:
	       # Mask with 'true' for every row where measurer.id = id
	       mask = pc.equal(self.measurer.id, id)

	       # This makes a view into the data using the given mask
	       # as a filter:
	       positions = self.position.apply_mask(mask)
	       maxes[id] = positions.distances().max()
	       
	   return maxes
	   
There are a lot more features to quivr that you can use to manage your
data, but you already know the most important ones. To summarize:

1. Define a Table class using Columns which describes your data.
2. Attach methods to the Table class to describe your computations
3. Use Table composition to manage complexity

What to look at next
--------------------

Some of the more advanced features you might be interested in include:

- :ref:`Attributes <attributes>` for attaching scalar (non-tabular) data to Tables
- :ref:`Linkages <linkage_guide>` to represent relationships between tables
- :ref:`Validators <validators_api>` to validate that data matches conditions
- :ref:`Serialization <serde_guide>` for working with data in Parquet and other formats
- :ref:`Handling Nulls <null_guide>` to be safe when dealing with missing values

