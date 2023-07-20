API Reference
=============

This is the reference documentation for the entire public API of quivr.

The :mod:`quivr` module is the public API. The submodules of
:mod:`quivr` (for example, :mod:`quivr.tables` or
:mod:`quivr.columns`) are not for direct access.


Table of Contents
-----------------
.. toctree::
   :maxdepth: 3
	     
   tables
   columns
   column_validators
   attributes
   linkages
   errors

   
.. module:: quivr

Primary API Objects
-------------------

.. autosummary::
   :nosignatures:

   Table
   Column
   Linkage

Column Types
------------
   
.. autosummary::
   :nosignatures:

   Column
   SubTableColumn
   Int8Column
   Int16Column
   Int32Column
   Int64Column
   UInt8Column
   UInt16Column
   UInt32Column
   UInt64Column
   Float16Column
   Float32Column
   Float64Column
   Decimal128Column
   Decimal256Column
   StringColumn
   LargeStringColumn
   BinaryColumn
   LargeBinaryColumn
   TimestampColumn
   DurationColumn
   Time32Column
   Time64Column
   Date32Column
   Date64Column
   MonthDayNanoIntervalColumn
   NullColumn
   StructColumn
   ListColumn
   LargeListColumn
   MapColumn
   DictionaryColumn
   RunEndEncodedColumn

Column Validators
-----------------

.. autosummary::
   :nosignatures:

   Validator
   eq
   lt
   le
   gt
   ge
   is_in
   and_


Table Attributes
----------------

.. autosummary::
   :nosignatures:

   StringAttribute
   IntAttribute
   FloatAttribute

Utility Functions
-----------------

.. autosummary::
   :nosignatures:

   concatenate
   defragment
