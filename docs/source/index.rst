.. quivr documentation master file, created by
   sphinx-quickstart on Tue Jul 18 23:08:28 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

quivr: data-oriented objects
============================

.. currentmodule:: quivr
		   
``quivr`` is a library for working with data-oriented objects in
Python.

The central object in ``quivr`` is the :class:`Table`: a precise
description of the schema for arrays of data, combined with namespaced
methods to keep your code well-structured.

To repeat, there are three motivations behind quivr:

1. Use schemas to describe data
2. Use namespaces to organize logic that operates on data
3. Use composition to manage complexity

For more, the best place to start is the :ref:`basic_usage` guide.

.. toctree::
   :maxdepth: 2

   installation
   basics
   guides/index
   examples/index
   api/overview
   development
	     


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
