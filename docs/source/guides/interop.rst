.. _interop_guide:

Interoperability
================

.. currentmodule:: quivr

quivr is designed to work well with the rest of the Python data
ecosystem. Tables have methods and tools for working with Pandas,
Numpy, and Parquet.


Throughout this guide, the following basic Table definitions will be used:

.. code-block:: py

   import quivr as qv

   class Positions(qv.Table):
       x = qv.Float64Column()
       y = qv.Float64Column()

   class Measurers(qv.Table):
       id = qv.UInt32Column()
       name = qv.StringColumn()

   class Measurements(qv.Table):
       position = Positions.as_column()
       measurer = Measurers.as_column()
       dataset = qv.StringAttribute()

Instantiated with these values:

.. code-block:: py

  positions = Positions.from_kwargs(
      x=[4.8, 4.9, None, 8.0, 5.3],
      y=[3.1, None, 3.2, 3.2, 3.3],
  )
  measurers = Measurers.from_kwargs(
      id=[0, 1, 1, None, 0],
      name=["Alice", "Bob", "Bob", None, "Alice"],
  )
  measurements = Measurements.from_kwargs(
      position=positions,
      measurer=measurers,
      dataset="atlas",
  )

Pandas
------

Tables can be converted to and from Pandas DataFrames.

To Pandas
#########

.. code-block:: py

   df = measurements.to_dataframe()
   print(df)

.. code-block:: text

	position.x  position.y  measurer.id measurer.name
  0         4.8         3.1          0.0         Alice
  1         4.9         NaN          1.0           Bob
  2         NaN         3.2          1.0           Bob
  3         8.0         3.2          NaN          None
  4         5.3         3.3          0.0         Alice


Column names in the dataframe correspond to the field names in the
Table instance. For subtables, column names are dot-delimited, as you
can see above.

This is called a "flattened" dataframe. You can choose an unflattened
form, if you prefer, which results in Python dictionaries at each
"cell" of the pandas DataFrame:

.. code-block:: py

   df = measurements.to_dataframe(flatten=False)
   print(df)

.. code-block:: text

		position                      measurer
  0   {'x': 4.8, 'y': 3.1}  {'id': 0.0, 'name': 'Alice'}
  1  {'x': 4.9, 'y': None}    {'id': 1.0, 'name': 'Bob'}
  2  {'x': None, 'y': 3.2}    {'id': 1.0, 'name': 'Bob'}
  3   {'x': 8.0, 'y': 3.2}    {'id': None, 'name': None}
  4   {'x': 5.3, 'y': 3.3}  {'id': 0.0, 'name': 'Alice'}



Types might get converted during to_dataframe. This conversion follows
behavior set by the PyArrow library. If a numeric column has any null
values, the column is converted to 64-bit floating point values and
the nulls are converted into ``NaN`` values.

Table :ref:`Attributes <attributes>` can be preserved in this
conversion to a DataFrame. By default, attributes are stored on the
:obj:`pandas.DataFrame.attrs` attribute, which is a dictionary of
global attributes for the DataFrame.

.. code-block:: py

   df = measurements.to_dataframe()
   print(df.attrs)

.. code-block:: text

   {"dataset": "atlas"}

If attributes are set on sub-tables, they'll be stored in a
dot-delimited fashion:

.. code-block:: py

   class Detections(qv.Table):
       measure = Measurements.as_column()
       label = qv.IntAttribute()

   dets = Detections.from_kwargs(measure=measurements, label=42)

   df = dets.to_dataframe()
   print(df.attrs)

.. code-block:: text

   {"measure.dataset": "atlas", "label": 42}

Alternatively, you can represent attributes with an additional column
in the DataFrame. The value will be repeated for every row:

.. code-block:: py

   df = measurements.to_dataframe(attr_handling="add_columns")
   print(df)

.. code-block:: text

	position.x  position.y  measurer.id measurer.name  dataset
  0         4.8         3.1          0.0         Alice      atlas
  1         4.9         NaN          1.0           Bob      atlas
  2         NaN         3.2          1.0           Bob      atlas
  3         8.0         3.2          NaN          None      atlas
  4         5.3         3.3          0.0         Alice      atlas


From Pandas
###########

You can read from Pandas using these methods
:meth:`Table.from_dataframe` and
:meth:`Table.from_flat_dataframe`. Loading from flat dataframes is
only needed when loading a Table that contains subtables.

You can specify any attributes in the constructor explicitly when
loading from a DataFrame if they are not present:

.. code-block:: py

   measurements2 = Measurements.from_flat_dataframe(df, dataset="atlas")

:meth:`Table.from_dataframe` and :meth:`Table.from_flat_dataframe`
will attempt to infer attribute values if they are not explicitly
passed. They will look for columns which match attribute names, and
will also check in the dataframe's `attrs` property, expecting the
same serialization as described above in the previous section.

In addition, :meth:`Table.from_kwargs` can handle :obj:`pandas.Series`
objects as input parameters, so you can do something like this:

.. code-block:: py

   measurements3 = Measurements.from_kwargs(
       position=Positions.from_kwargs(
           x=df['position.x'],
	   y=df['position.y']
       ),
       measurer=Measurers.from_kwargs(
           id=df['measurer.id'].fillna(0).astype("uint32"),
	   name=df['measurer.name'],
       )
       dataset="atlas"
   )

Limitations
###########

Since Pandas Series don't support null values (but quivr/Arrow arrays
*do*), you'll see some loss of fidelity when going from quivr into
Pandas datastructures and back.

For more information, see `the Arrow documentation on the subject
<https://arrow.apache.org/docs/python/pandas.html#type-differences>`_.

Arrow
-----

Arrow is the native backing system for quivr's tables. All data is
stored internally using Arrow arrays and schemas. As a result, data
always works with Arrow losslessly.

To Arrow
########

The underlying Arrow data behind a Table can be accessed several ways:

1. Columns can be accessed to get individual Arrow Arrays of data.
2. The entire backing Arrow table can be accessed.
3. The quivr table can be reshaped and presented as an Arrow StructArray.

To Arrow Arrays
***************

For a column named ``foo`` on a table instance named ``tab``,
``tab.foo`` will get the column's data and present it as an Arrow
array directly:

.. code-block:: py

   print(type(positions.x))
   # pyarrow.lib.DoubleArray

The mapping of types is comprehensively documented in the `API reference for quivr Columns <column_api>`_.

To Arrow Tables
***************

The underlying :obj:`pyarrow.Table` instance can always be accessed on a quivr
Table instance through Table.table instance attribute:

.. code-block:: py

   print(type(positions.table))
   # pyarrow.lib.Table

The :obj:`pyarrow.Table` is a useful structure that can then be used
for low-level operations. For example, to make a list of
:obj:`pyarrow.RecordBatch` objects which describe the table's data in
batches suitable for serializing and even communicating over a
network, you can use ``Table.table.to_batches()``:

.. code-block:: py

   for batch in positions.table.to_batches():
       send(batch)

The :obj:`pyarrow.Table` holds *all* of the data associated with the
quivr Table, including attributes. Attributes are stored in schema
metadata, which is a dictionary with :obj:`bytes` keys and
values. Attributes are encoded with their name for the key (with a
dot-delimited prefix for attributes on sub-tables) and with a
byte-encoding of their value.

.. code-block:: py

   print(measurements.table.schema.metadata)
   # {b'dataset': b'atlas'}

To Arrow StructArrays
*********************

Tables have a :meth:`Table.to_structarray()` method which can be used
to construct a :obj:`pyarrow.StructArray`. This array holds PyArrow
:obj:`pyarrow.struct` instances. The struct will have a schema which
matches that of the Table:

.. code-block:: py

    print(measurements.to_structarray()[0])
    # [('position', {'x': 4.8, 'y': 3.1}), ('measurer', {'id': 0, 'name': 'Alice'})]

From Arrow
##########

You can bring Arrow data into quivr to populate Table instances.

First, you can use :meth:`Table.from_pyarrow()` to populate a quivr
Table's data with the contents of a PyArrow Table. When you do so, the
:obj:`pyarrow.Table`'s metadata will be preserved, and will be used to
set Table Attribute values, if they're set on the quivr Table class
definition. You can provide them as keyword arguments if they aren't
present in the :obj:`pyarrow.Table`'s schema, as well. If they're
present in both, then the keyword arguments take precedence.

The source :obj:`pyarrow.Table` should have a schema which matches the
:obj:`quivr.Table`'s definition. "Matching" is a somewhat unrigorous
concept currently, but the schema "matches" if:

- the source schema has a matching field for all non-nullable fields, and
- all fields present in the source schema can be cast to the
  corresponding types in the destination table.

Notably, it is *not* necessary that the source pyarrow Table have
exactly the same fields as the destination quivr Table. Extra fields
will be ignored. This means that Tables with a subset of columns
defined can be used to project a view into a small number of the
columns of a larger PyArrow Table.

In addition to :meth:`Table.from_pyarrow()`, you can pass in
:obj:`pyarrow.Array`\ s when constructing a Table using
:meth:`Table.from_kwargs`. Any of the keyword arguments' values can be
pyarrow Arrays.

Numpy
-----

To Numpy
########

The Arrays that are retrieved when accessing a quivr Table instance's
columns can be cast as Numpy arrays using the
:meth:`pyarrow.Array.to_numpy()` method. See `Arrow documentation on
Numpy integration <https://arrow.apache.org/docs/python/numpy.html>`_
for more information.

From Numpy
##########

Numpy arrays can be passed in to the :meth:`Table.from_kwargs`
constructor.

Parquet
-------

See :ref:`serde_parquet`.
