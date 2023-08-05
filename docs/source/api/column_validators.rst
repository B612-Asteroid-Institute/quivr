.. _validators_api:

Column Validators
=================

Validators assess whether input data for a particular column is
valid. Validators provide a framework for describing expectations for
data.

.. currentmodule:: quivr

Validators are implemented in a functional style. There are a bunch of
functions that can create new :class:`Validator` instances to verify
data matches given preconditions.

The validator functions are built on :mod:`pyarrow.compute` calls, so
the data types they can handle are determined by the available kernels
in the `Arrow C++ compute library
<https://arrow.apache.org/docs/cpp/compute.html#available-functions>`_.

The validators can be combined using the :func:`and_` function. This
can combine several validators to make one compound statement. For
example, this code validates the ``size`` column is between 5 and 30:

.. code-block:: py

   import quivr as qv
   from quivr import and_, ge, le

   class Hat(qv.Table):
       size = qv.Int8Column(validator=and_(ge(5), le(30)))
       

.. autofunction:: eq
.. autofunction:: lt
.. autofunction:: le
.. autofunction:: gt
.. autofunction:: ge
.. autofunction:: is_in

.. autofunction:: and_
		  
.. autoclass:: Validator
   :members:

		  
