Development
===========

This is documentation for developers working in ``quivr`` itself.

Quivr's development environment is managed with `Hatch
<https://hatch.pypa.io/>`_. To work on ``quivr``, you'll need
``hatch`` installed on your system, whether in a virtualenv or
globally.

To run the full battery of tests, you also need to install
``hatch-containers`` to allow running tests in containers (to test
against different Python versions).

Development tasks
-----------------

Hatch manages local dependency state and can run tasks. These are
defined in `pyproject.toml
<https://github.com/spenczar/quivr/blob/main/pyproject.toml>`_.

.. list-table:: Development Tasks
   :widths: 40 60

   * - Command
     - Action
   * - ``hatch run dev:format``
     -  auto-format all code to comply with `Black <https://github.com/psf/black>`_.
   * - ``hatch run dev:lint``
     - run `ruff <https://github.com/astral-sh/ruff>`_ to check for common errors.
   * - ``hatch run dev:test``
     - run the primary test suite.
   * - ``hatch run dev:doctest``
     - run the doctests, which are embedded in the docstrings.
   * - ``hatch run dev:typecheck``
     - run the typechecker for internal consistency, and to validate typing tests
   * - ``hatch run dev:benchmark``
     - run the benchmark suite.
   * - ``hatch run test:all``
     - run the full suite of all tests (lint, typecheck, test, and
       doctest) on all target Python versions using containers.


In addition, there are some tasks that are for building documentation:

.. list-table:: Documentation Tasks
   :widths: 40 60

   * - Command
     - Action
   * - ``hatch run docs:make``
     - build the HTML documentation. It will be built in ``docs/build/html``, relative to the repo root.
   * - ``hatch run docs:open``
     - open the HTML documentation in your browser.
   * - ``hatch run docs:clean``
     - remove the built HTML documentation.
   * - ``hatch run docs:rebuild``
     - clean and rebuild the HTML documentation.

Test Suites
-----------

quivr has three sorts of tests:

- Unit tests: These are defined in ``./test`` and use `pytest
  <https://docs.pytest.org/en/stable/>`_.
- Doctests: These are embedded in the docstrings of the code
  itself. They verify that the code examples in the documentation are
  correct.
- Type tests: These are defined in ``./test/typing_tests`` and verify
  that the type annotations in the code pass `mypy
  <https://mypy.readthedocs.io/en/stable/>`_'s type checking.

In addition, there is a benchmark suite, defined in ``./test`` as
well, directly in the relevant Python unit test files. These are
microbenchmarks that measure performance of narrow portions of the
code using `pytest-benchmark
<https://pytest-benchmark.readthedocs.io/en/stable/>`_. These are
not run by default, but can be run with ``hatch run benchmark``.
