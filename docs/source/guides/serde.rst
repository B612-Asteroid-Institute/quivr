.. _serde_guide:

Serialization and Deserialization
=================================

.. currentmodule:: quivr

``quivr`` Tables can be loaded from Parquet files, Feather files, and CSVs.


Parquet
+++++++


Renaming Columns
----------------

When you're loading a Parquet file, it might not be one you created, so you
might not have control over its schema. If it has column names that
are not valid Python identifiers, or just if you'd prefer they be
something different, you can supply a column name mapping to the
deserialization functions (:meth:`Table.from_parquet`).

For example, here is the 2022 schema for the famous `New York City taxi data <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>`_:

.. code-block::

   VendorID: int64
   tpep_pickup_datetime: timestamp[us]
   tpep_dropoff_datetime: timestamp[us]
   passenger_count: double
   trip_distance: double
   RatecodeID: double
   store_and_fwd_flag: string
   PULocationID: int64
   DOLocationID: int64
   payment_type: int64
   fare_amount: double
   extra: double
   mta_tax: double
   tip_amount: double
   tolls_amount: double
   improvement_surcharge: double
   total_amount: double
   congestion_surcharge: double
   airport_fee: double

Some of these are fine, but others are a little disagreeable. Without
a column name map, we'd need use matching attribute names in our Table class:

.. literalinclude:: ./snippets/serde/taxi1.py
   :language: python

But we can use a column name map to make the names more pleasant. We
only need to supply the names that are different from the attribute
names:

.. literalinclude:: ./snippets/serde/taxi2.py
   :language: python

To take it one step further, we could encapsulate this in a method for
our ``TaxiData`` class:

.. literalinclude:: ./snippets/serde/taxi3.py
   :language: python

Changing Column Types
---------------------

Sometimes you might want to change the type of a column. For example,
in the preceding example of New York City taxi data, the ``passenger_count``
column is a float, but it should be an integer.

You can do this by just using your desired type in the columns for your table.

.. literalinclude:: ./snippets/serde/taxi4.py
   :language: python

