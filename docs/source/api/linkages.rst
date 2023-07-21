Linkages
========

.. currentmodule:: quivr

Linkages are tools for building relationships that span multiple tables.

You can use linkages to act as indexes to relate the rows from two
different tables which share common values. In fact, linkages are a
bit more general than that: they can relate rows based on computed
values as well.

.. autoclass:: Linkage
   :members:

.. autoclass:: MultiKeyLinkage
   :members:

.. autotypevar:: quivr.linkage.LeftTable

.. autotypevar:: quivr.linkage.RightTable
