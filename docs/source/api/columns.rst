.. _columns_api:

Columns
=======

.. currentmodule:: quivr

Columns are descriptions of the data that populates a given
:class:`Table`. Columns are associated with tables by writing them as
class attributes on :class:`Table` subclasses. For example:

.. code-block:: py

   import quivr as qv
   
   class Person(qv.Table):
       name = qv.StringColumn()
       age = qv.Uint8Column()
       favorite_books = qv.ListColumn(quivr.StringColumn())

The ``Person`` Table defined above has three columns: a string name, a
uint8 age, and a list of strings for books.

All of the column types below inherit from :class:`Column`, and so
they are Descriptors: they all have :meth:`Column.__get__`,
:meth:`Column.__set__`, and :meth:`Column.__set_name__` methods.


.. _simple_types:

Simple Types
------------

These are the simplest column types. They all are initialized the same way:

.. code-block:: py

   import quivr as qv

   class MyTable(qv.Table)
       # With no arguments, you get a non-nullable column that doesn't validate data.
       # This could be qv.Uint8Column(), or qv.StringColumn(), whatever
       qv.Int32Column()

       # Pass nullable=True to get a nullable column.
       qv.Int32Column(nullable=True)

       # Pass a validator to check the input data against a constraint.
       qv.Int32Column(validator=qv.gt(0))

       # Pass metadata for arbitrary extra information to
       # include. Metadata should be a string-to-string dictionary.
       qv.Int32Column(metadata={'units': 'seconds'})

       # Set a default value to be used if any inputs are null
       qv.Int32Column(nullable=True, default=3)


When you access a column of a primitive type on a :class:`Table`
instance, you get a :class:`pyarrow.Array` back. The data type of the
array is described in this table:

+---------------------------+-----------------------------------+------------------------------+
|Column Type                |Data Type                          |Description                   |
+===========================+===================================+==============================+
|:class:`StringColumn`      |:class:`pyarrow.StringArray`       |UTF-8 string data. Strings    |
|                           |                                   |must all be less than 2\      |
|                           |                                   |:sup:`31` characters long.    |
+---------------------------+-----------------------------------+------------------------------+
|:class:`LargeStringColumn` |:class:`pyarrow.LargeStringArray`  |UTF-8 string data up to 2\    |
|                           |                                   |:sup:`63` characters long.    |
+---------------------------+-----------------------------------+------------------------------+
|:class:`Int8Column`        |:class:`pyarrow.Int8Array`         |8-bit signed integers         |
|                           |                                   |(-128 to 127)                 |
+---------------------------+-----------------------------------+------------------------------+
|:class:`Int16Column`       |:class:`pyarrow.Int16Array`        |16-bit signed integers        |
|                           |                                   |(-32,768 to 32,767)           |
+---------------------------+-----------------------------------+------------------------------+
|:class:`Int32Column`       |:class:`pyarrow.Int32Array`        |32-bit signed integers        |
|                           |                                   |(-2\ :sup:`31` to 2\          |
|                           |                                   |:sup:`31` - 1)                |
+---------------------------+-----------------------------------+------------------------------+
|:class:`Int64Column`       |:class:`pyarrow.Int64Array`        |64-bit signed integers        |
|                           |                                   |(-2\ :sup:`63` to 2\          |
|                           |                                   |:sup:`63` - 1)                |
+---------------------------+-----------------------------------+------------------------------+
|:class:`UInt8Column`       |:class:`pyarrow.UInt8Array`        |8-bit unsigned integers (0    |
|                           |                                   |to 255)                       |
+---------------------------+-----------------------------------+------------------------------+
|:class:`UInt16Column`      |:class:`pyarrow.UInt16Array`       |16-bit unsigned integers      |
|                           |                                   |(0 to 65,535)                 |
|                           |                                   |                              |
+---------------------------+-----------------------------------+------------------------------+
|:class:`UInt32Column`      |:class:`pyarrow.UInt32Array`       |32-bit unsigned integers      |
|                           |                                   |(0 to 2\ :sup:`32` - 1)       |
|                           |                                   |                              |
+---------------------------+-----------------------------------+------------------------------+
|:class:`UInt64Column`      |:class:`pyarrow.UInt64Array`       |64-bit unsigned integers      |
|                           |                                   |(0 to 2\ :sup:`64` - 1)       |
|                           |                                   |                              |
+---------------------------+-----------------------------------+------------------------------+
|:class:`Float16Column`     |:class:`pyarrow.HalfFloatArray`    |16-bit floating point         |
|                           |                                   |values                        |
|                           |                                   |                              |
+---------------------------+-----------------------------------+------------------------------+
|:class:`Float32Column`     |:class:`pyarrow.FloatArray`        |32-bit floating point         |
|                           |                                   |values                        |
|                           |                                   |                              |
+---------------------------+-----------------------------------+------------------------------+
|:class:`Float64Column`     |:class:`pyarrow.DoubleArray`       |64-bit floating point         |
|                           |                                   |values                        |
|                           |                                   |                              |
+---------------------------+-----------------------------------+------------------------------+
|:class:`BooleanColumn`     |:class:`pyarrow.BooleanArray`      |Boolean (true/false)          |
|                           |                                   |values                        |
|                           |                                   |                              |
+---------------------------+-----------------------------------+------------------------------+
|:class:`NullColumn`        |:class:`pyarrow.NullArray`         |A zero-sized array of         |
|                           |                                   |nulls                         |
|                           |                                   |                              |
+---------------------------+-----------------------------------+------------------------------+
|:class:`BinaryColumn`      |:class:`pyarrow.BinaryArray`       |Arbitrary binary blobs,       |
|                           |                                   |variably sized, up to 2\      |
|                           |                                   |:sup:`31` bytes long each     |
|                           |                                   |(about 4GB)                   |
+---------------------------+-----------------------------------+------------------------------+
|:class:`LargeBinaryColumn` |:class:`pyarrow.LargeBinaryArray`  |Arbitrary binary blobs,       |
|                           |                                   |variably sized, up to 2\      |
|                           |                                   |:sup:`63` bytes long each     |
|                           |                                   |(about 9 exabytes)            |
+---------------------------+-----------------------------------+------------------------------+

.. autoclass:: StringColumn
   :members:

.. autoclass:: LargeStringColumn
   :members:

.. autoclass:: Int8Column
   :members:
      
.. autoclass:: Int16Column
   :members:
      
.. autoclass:: Int32Column
   :members:
      
.. autoclass:: Int64Column
   :members:
      
.. autoclass:: UInt8Column
   :members:
      
.. autoclass:: UInt16Column
   :members:
      
.. autoclass:: UInt32Column
   :members:
      
.. autoclass:: UInt64Column
   :members:
      
.. autoclass:: Float16Column
   :members:
      
.. autoclass:: Float32Column
   :members:
      
.. autoclass:: Float64Column
   :members:

.. autoclass:: BooleanColumn
   :members:
      
.. autoclass:: NullColumn
   :members:

.. autoclass:: BinaryColumn
   :members:

.. autoclass:: LargeBinaryColumn
   :members:

Fixed-Size Binary Data
-----------------------

:class:`BinaryColumn` and :class:`LargeBinaryColumn` work with
variably-sized binary.  If every item is of identical size, you can
use :class:`FixedSizeBinaryColumn` to save some overhead.
      
.. autoclass:: FixedSizeBinaryColumn
   :members:

Decimals
--------

Decimal data uses fixed-point, which guarantees that it has a
consistent number of significant digits.

Decimal columns can be either 128-bit or 256-bit. When you set them
up, you provide the "precision" and "scale" to be used.

.. autoclass:: Decimal128Column
   :members:
      
.. autoclass:: Decimal256Column
   :members:


Time-Related Types
------------------
      
.. autoclass:: TimestampColumn
   :members:
      
.. autoclass:: DurationColumn
   :members:
      
.. autoclass:: Time32Column
   :members:
      
.. autoclass:: Time64Column
   :members:
      
.. autoclass:: Date32Column
   :members:
      
.. autoclass:: Date64Column
   :members:
      
.. autoclass:: MonthDayNanoIntervalColumn
   :members:

Structured Data
---------------

Columns can contain nested structural data. With these types, each
*row* of the column contains some structure.

It is easy to get confused by this: All columns are "lists" in a
sense, but :class:`ListColumn` is a two-dimensional structure. Each
row of the column is, itself, a list.
      
.. autoclass:: ListColumn
   :members:

.. autoclass:: FixedSizeListColumn
   :members:
	       
.. autoclass:: LargeListColumn
   :members:
            
.. autoclass:: MapColumn
   :members:
		     
.. autoclass:: StructColumn
   :members:
      
.. autoclass:: SubTableColumn
   :members:
		     
   SubTableColumns are generally created through
   :meth:`Table.as_column`, and are not created directly.

Types For Encoding Efficiency
-----------------------------
      
.. autoclass:: DictionaryColumn
   :members:
      
.. autoclass:: RunEndEncodedColumn
   :members:
      
   
Base Class
----------

All of the column types inherit from the :class:`Column` base
class. Users are not expected to use :class:`Column`
directly. Instead, use one of the appropriate subclasses.
		   
.. autoclass:: Column
   :members:
   :special-members: __get__, __set__, __set_name__

Typing Helpers
--------------

.. autotypevar:: quivr.columns.T

.. autodata:: MetadataDict

.. autodata:: Byteslike

