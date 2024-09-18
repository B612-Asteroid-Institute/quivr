# Changelog

This file documents notable changes between versions of quivr.

## [0.7.4] - 2024-09-18

### Removed

- `experimental.shmem` has been removed. Users are encouraged to use tools
such as `ray` for parallel processing and shared memory of pyarrow types.

### Added

- `Table.invalid_mask`, `Table.separate_invalid` have been added to allow users to select rows
that fail validation checks.
- `concatenate` now supports passing the `validate` argument, if you want to postpone automatic
validation of a table until after concatenation.
- `Table.drop_duplicates` has been added to remove duplicate rows from a table.
- `Table.unique_indices` has been added to return the indices of the first/last occurrence of each 
unique row or subset of columns.


## [0.7.3] - 2024-05-20

### Fixed

- Concatenating empty quivr tables will no longer raise an error with 
incompatible attributes. Instead, attributes will be taken from the first
non-empty table.


## [0.7.2] - 2023-10-18

### Removed

- The `from_data`, `from_list`, `from_rows`, and `from_pydict`
  constructors, which were deprecated in 0.6.0, have been removed;
  `from_kwargs` is generally preferred when constructing from Python
  values. (#33)

### Fixed

- Quivr tables will now round-trip correctly with all
  types. Previously, FixedLengthLists, LargeStrings, LargeBinary, and
  other unusual types could be incorrectly handled when loading from
  flattened dataframes (#58).

## [0.7.0] - 2023-10-03

### Added

- `quivr.experimental.shmem` provides new utilities for run functions
against quivr Tables with multiple processes in shared memory:
  - `to_shared_memory` and `from_shared_memory` can be used to read and
	write a quivr Table in shared memory. This allows separate
	processes to work off of slices of a Table without a copy of any
	data, or with redundant memory usage.
  - `execute_parallel` is a function that simplifies running a function
    against a Table's data with multiple processes. The Table's data
    will be split up using a configurable partitioning strategy, and
    each partition will be passed to a separate worker. Results are
    returned as they are completed in a streaming iterator.
  - `ChunkedPartitioning` and `GroupedPartitioning` are classes which
    represent two possible partitioning strategies: uniform chunks of
    fixed size, or partitions which share a common particular
    value. Additional partitioning strategies can be provided by
    providing a subclass implementation of the `Partitioning` class.


- Conversion of Tables to and from pandas DataFrames now can preserve
  Table attributes (#56). Three possible approaches are available:
  - "add_columns": Store attribute values by repeating them in every
    row of the dataframe. Each attribute gets a separate column. For
    subtables, attributes are stored under a dot-delimited prefix.
  - "attrs": Use the experimental
    [pandas.DataFrame.attrs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.attrs.html)
    API to store attributes directly on the DataFrame as a dictionary.
  - "drop": remove the attributes entirely. This was the old behavior.
  These are enabled in the `Table.to_dataframe` method. When a
  dataframe is loaded with `from_dataframe` or `from_flat_dataframe`,
  attributes are inferred.

- Column names can be dot-delimited in `Table.select` and
  `Table.column` to reference subtables (#53).

- Column names can be dot-delimited in `Table.sort_by` to reference
  subtables (#54).

## [0.6.6] - 2023-09-27

### Fixed

- Columns which are masked to hide all data now can be accessed (#51).

## [0.6.5] - 2023-08-30

### Fixed

- Concatenating empty tables no longer returns a ValueError.

## [0.6.4] - 2023-08-24

### Added

- `Table.set_column` was added. This is a new method that returns a
  copy of the table, but with a single column replaced.

### Changed

- Accessing columns is now much faster. (#47).

### Removed

- Setting columns through normal Python assignment statements
  (`table.x = ...`) is no longer possible. It was an accident that it
  worked in the first place. Instead, use `Table.set_column`.

## [0.6.3] - 2023-08-16

### Fixed

- Indexing a table with a negative integer (`table[-1]`, for example)
  now works like it does for other Python structures: by pulling from
  the back. (#40)


## [0.6.2] - 2023-08-08

This patch release is an addendum to [0.6.1], which attempted to
resolve issues with nullable subtables with non-nullable fields, but
which had several issues.

### Fixed

- Several more fixes for loading null subtables with non-null columns:
  correctly handling round-tripping, loading from PyArrow tables, and
  more.

## [0.6.1] - 2023-08-08

### Fixed

- When a Table contains a nullable sub-table column, and _that_
  subtable has non-nullable fields, `from_kwargs` would reject a null
  value for the sub-table column. This is now fixed. (#37)

## [0.6.0] - 2023-08-07

This release has several major changes:
 - `Linkage`s are added.
 - Columns are now non-nullable by default.
 - Many Table constructors are deprecated.

### Added

- `combine_linkages` and `combine_multikeylinkages`, utility functions
  for concatenating linkages, were added. (#27)
- Columns now accept default values which are used when null. (#25)
- New `from_pyarrow` constructor for loading data from a PyArrow table. (#24)

### Fixed

- Schemas are more rigorously enforced when loading data, particularly
  from PyArrow Tables. All columns of the source data are checked for
  consistency.

### Changed

- Columns are now non-nullable by default. (#35)
- Several attribute names are now reserved by quivr, and it is an
  error to name columns using the reserved names, which include
  "table", "schema", and a few other more obscure internal names. (#36)
- Column validators are run by default when constructing a Table
  instance. (#32)
- Attributes are now immutable by default. They can optionally be made
  mutable by passing `mutable=True` in their constructors. (#28)

### Deprecated

- The `from_data`, `from_list`, `from_rows`, and `from_pydict`
  constructors are now deprecated; `from_kwargs` is generally
  preferred when constructing from Python values. (#33)

### Removed

- The unadvertised experimental `StringIndex` structure has been
  removed. Linkages do the same job much better.
- The unadvertised experimental `quivr.matrix` module has been
  removed. Use FixedSizeList columns instead.

## [0.5.0] - 2023-07-21

### Added

- `Linkage` and `MultiKeyLinkage`, two constructs for working with
  multiple Table instances with common keys, were added. (#21)
- Documentation is now generated and sent to readthedocs. Find it at
  https://quivr.readthedocs.org/.

## [0.4.3] - 2023-07-18

### Added

- `Table.equals`, which checks for equality of two Table instances,
  has been added. (#17)
- All public modules have full type annotations, now, which are
  verified with mypy.

### Fixed

- Column validators no longer crash on columns with all null values (#13)

## [0.4.2] - 2023-06-05

### Added

- Pandas Series objects can now be passed in to Table constructors
  like `from_kwargs`. (edb7482)

## [0.4.1] - 2023-06-05

### Added

- Table Attributes and Columns can now be accessed as class-level
  attributes. This access the `Column` or `Attribute` _itself_ rather
  than the data it points to.

## [0.4.0] - 2023-06-05

### Changed

- Table "Fields" are renamed to "Columns."

## [0.3.4] - 2023-05-26

### Fixed

- Changes made to support Python 3.9 and 3.10.

## [0.3.3] - 2023-05-26

### Added

- Attributes are added: scalar values that can be attached to an
  entire Table instance. These are serialized in Table metadata so
  they survive encoding and decoding.
- Added `Table.empty()` method which creates a table with length zero.

## [0.3.2] - 2023-05-18

### Added

- Column Validators are added: tools for ensuring that the data in a
  Table passes checks.

## [0.3.1] - 2023-05-18

### Added

- Added a `from_parquet` method to Table.

### Fixed

- Correctly cast inputs to the right schema type when constructing a
  Table instance.

## [0.3.0] - 2023-05-17

### Added
- Added support for instance-level attributes via the `with_table` pattern.

## [0.2.3] - 2023-05-17

Extra release to fix an issue publishing to PyPI.

## [0.2.2] - 2023-05-17

### Added

- Allow nullable columns to be passed in as None via from_kwargs. (#2)

### Fixed

- Import Table at the package level. (#1)

## [0.2.1] - 2023-05-05

### Added

- Added `py.typed` file to package, hooking in to type checkers.
- Make SubTableField a Generic type.

## [0.2.0] - 2023-05-04

### Changed

Instead of naming a `pyarrow.Schema` as a class-level attribute, quivr
now supports an explicit Field type which is used to describe the
fields used in a Table. Implementations of many Fields based on
PyArrow types are provided.

## [0.1.1] - 2023-05-02

### Added

Added a variety of convenience constructors for Table instances.

## [0.1.0] - 2023-05-01

First tagged release. Many, many changes to the core concept.

## [Initial commit] - 2023-04-08

Initial commit of the original idea, implemented via metaclasses.

[0.7.2]: https://github.com/spenczar/quivr/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/spenczar/quivr/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/spenczar/quivr/compare/v0.6.6...v0.7.0
[0.6.6]: https://github.com/spenczar/quivr/compare/v0.6.5...v0.6.6
[0.6.5]: https://github.com/spenczar/quivr/compare/v0.6.4...v0.6.5
[0.6.4]: https://github.com/spenczar/quivr/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/spenczar/quivr/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/spenczar/quivr/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/spenczar/quivr/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/spenczar/quivr/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/spenczar/quivr/compare/v0.4.3...v0.5.0
[0.4.3]: https://github.com/spenczar/quivr/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/spenczar/quivr/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/spenczar/quivr/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/spenczar/quivr/compare/v0.3.4...v0.4.0
[0.3.4]: https://github.com/spenczar/quivr/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/spenczar/quivr/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/spenczar/quivr/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/spenczar/quivr/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/spenczar/quivr/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/spenczar/quivr/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/spenczar/quivr/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/spenczar/quivr/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/spenczar/quivr/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/spenczar/quivr/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/spenczar/quivr/compare/583c456...v0.1.0
[Initial commit]: https://github.com/spenczar/quivr/commit/583c456a2fb4550718cf166ff4330181372f7a1e
