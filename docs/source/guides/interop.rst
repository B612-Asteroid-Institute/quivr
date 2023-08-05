.. _interop_guide:

Interoperability
================

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

.. code-block:: txt

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

.. code-block:: txt

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

Table :ref:`Attributes <attributes>` aren't preserved in the
conversion to a DataFrame, since there is nowhere to store them.

From Pandas
###########

You can read from Pandas using these methods
:meth:`Table.from_dataframe` and
:meth:`Table.from_flat_dataframe`. Loading from flat dataframes is
only needed when loading a Table that contains subtables.

You need to specify any attributes in the constructor explicitly when
loading from a DataFrame. There's nowhere on the DataFrame itself
where they could be stored.

.. code-block:: py

   measurements2 = Measurements.from_flat_dataframe(df, dataset="atlas")


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

Arrow
-----

TODO: write this

 - Handling in :meth:`Table.from_kwargs`
 - :meth:`Table.to_structarray`

Numpy
-----

TODO: write this

 - Handling in :meth:`Table.from_kwargs`
 - to_numpy of array values

Parquet
-------

See :ref:`serde_parquet`_.
