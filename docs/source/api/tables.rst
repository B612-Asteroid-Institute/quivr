Tables
======

.. currentmodule:: quivr

The Table Class
+++++++++++++++

.. autoclass:: Table

Constructors
------------

  Table instances should generally be built out of the
  constructors. The :meth:`Table.__init__` method only accepts a "raw"
  :class:`pyarrow.Table`.

  See also: :ref:`serde`.

.. automethod:: Table.from_data
.. automethod:: Table.from_kwargs
.. automethod:: Table.from_arrays
.. automethod:: Table.from_pydict
.. automethod:: Table.from_rows
.. automethod:: Table.from_lists
.. automethod:: Table.from_dataframe
.. automethod:: Table.from_flat_dataframe
.. automethod:: Table.empty
.. automethod:: Table.with_table
.. automethod:: Table.__init__
		  
.. _serde:
  
Serialization/Deserialization
-----------------------------

These methods handle serializing the Table to and from other
formats, most often for writing to or reading from disk.
  
.. automethod:: Table.from_parquet
.. automethod:: Table.from_feather
.. automethod:: Table.from_csv
.. automethod:: Table.to_parquet
.. automethod:: Table.to_feather
.. automethod:: Table.to_csv

Validation
----------
.. automethod:: Table.validate
.. automethod:: Table.is_valid
		  
Interoperability
----------------

.. automethod:: Table.as_column
.. automethod:: Table.to_structarray
.. automethod:: Table.flattened_table
.. automethod:: Table.to_dataframe
.. automethod:: Table.column

Filtering, Selection, Sorting
-----------------------------
.. automethod:: Table.select
.. automethod:: Table.sort_by
.. automethod:: Table.take
.. automethod:: Table.apply_mask
.. automethod:: Table.where
.. automethod:: Table.__getitem__
.. automethod:: Table.__iter__

Data Internals
--------------
.. automethod:: Table.chunk_counts
.. automethod:: Table.fragmented
.. automethod:: Table.attributes

Miscellaneous
-------------
.. automethod:: Table.__repr__
.. automethod:: Table.__len__
.. automethod:: Table.__eq__

Utility Functions
+++++++++++++++++

.. autofunction:: concatenate

.. autofunction:: defragment
