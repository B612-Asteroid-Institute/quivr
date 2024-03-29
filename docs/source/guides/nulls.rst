.. _null_guide:

Handling Nulls
==============

One of the core features of Quivr is its support for handling null
values in data.

Data behind a Quivr table can have null values. You can explicitly
permit nulls by passing a ``nullable=True`` parameter when
contructing a :obj:`quivr.Column`. If this is not set, then any
constructors of tables which provide null values will be rejected:

.. code-block:: python

   import quivr as qv

   class MyTable(qv.Table):
       id = qv.StringColumn()

   # This will raise an error:
   t = MyTable.from_kwargs(id=["a", "b", None, "d"])

   class MyTable2(qv.Table):
       id = qv.StringColumn(nullable=True)

   # This is fine
   t = MyTable2.from_kwargs(id=["a", "b", None, "d"])



Using defaults
--------------

Instead of raising an error in the presence of nulls, you can set a
default that will be filled in when nulls are present.

This default value can be either a static literal value, or a callable
which takes no arguments and which returns a value:

.. code-block:: python

   import quivr as qv
   import uuid

   class User(qv.Table):
       id = qv.StringColumn(default=uuid.uuidv4)
       login_attempts = qv.UInt32Column(default=0)

   # This won't raise an error. Instead, it will call uuid.uuidv4()
   # for each null id, and will insert '0' for each null login_attempts.
   t = User.from_kwargs(
       id=["a", "b", None, "d", None],
       login_attempts=[0, 1, None, None, 3],
   )

Default values will also get used if a column is omitted entirely from
a constructor. For example:

.. code-block:: python

   # t.login_attempts will be filled with [0, 0, 0].
   t = User.from_kwargs(
       id=["a", "b", "c"],
   )

Loading Foreign Data
--------------------


