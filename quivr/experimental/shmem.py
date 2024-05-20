from __future__ import annotations

import abc
import concurrent.futures
import dataclasses
import mmap
import sys
from multiprocessing import managers, shared_memory

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from typing import Any, Callable, Concatenate, Generic, Iterator, ParamSpec, TypeVar

import pyarrow as pa
import pyarrow.compute as pc

import quivr as qv


def to_shared_memory(data: qv.Table, mgr: managers.SharedMemoryManager) -> shared_memory.SharedMemory:
    """
    Write a quivr Table instance to a new shared memory object owned by a SharedMemoryManager.

    :param data: The quivr Table instance to write.
    :param mgr: The SharedMemoryManager to own the shared memory object.
    :return: The shared memory object.
    """

    # Write the data as record batches to a buffer
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, data.table.schema) as writer:
        writer.write_table(data.table)
    buf = sink.getvalue()

    shm = mgr.SharedMemory(size=buf.size)
    shm.buf[: buf.size] = buf.to_pybytes()

    return shm


T = TypeVar("T", bound=qv.Table)


def from_shared_memory(shm: shared_memory.SharedMemory, table_class: type[T]) -> T:
    """
    Load a shared memory object as a quivr Table instance.

    :param shm: The shared memory object containing a quivr Table,
        serialized using PyArrow's IPC stream serialization.
    :param table_class: The class of the Table instance to create.
    """
    # HACK: shm.buf.obj is a mmap object, so why not just use that?
    # Because mmap objects use reference counting internally, and the
    # shm.buf.obj mmap's reference count will incorrectly require a
    # ton of management from the caller to avoid leaking
    # memory. Creating a new mmap here ensures that we'll get the
    # reference counting right, even though it's pretty ugly.
    f = mmap.mmap(shm._fd, shm.size)  # type: ignore
    with pa.ipc.open_stream(f) as reader:
        pyarrow_table = reader.read_all()
        instance = table_class.from_pyarrow(pyarrow_table)
    return instance


P = ParamSpec("P")


def _run_on_shared_memory(
    ref: TableReference[T],
    func: Callable[Concatenate[T, P], Any],
    args: P.args,
    kwargs: P.kwargs,
) -> Any:
    """
    Run a function on a table stored in shared memory.
    """
    # Load the shared memory object
    instance = ref.load()

    # Load from args and kwargs
    args = [arg.load() if isinstance(arg, TableReference) else arg for arg in args]
    kwargs = {k: v.load() if isinstance(v, TableReference) else v for k, v in kwargs.items()}
    return func(instance, *args, **kwargs)


class Partitioning(abc.ABC):
    """
    A partitioning strategy for executing a function in parallel on a Table.

    This class is abstract and should be subclassed to implement a particular partitioning strategy.
    """

    @abc.abstractmethod
    def partition(self, table: T) -> Iterator[T]: ...


def partition_func(f: Callable[[T], Iterator[T]]) -> Partitioning:
    """
    Defines a partitioning strategy by providing a function that partitions a
    Table into multiple Tables.
    """

    class PartitioningFunc(Partitioning):
        def partition(self, table: T) -> Iterator[T]:
            return f(table)

    return PartitioningFunc()


class ChunkedPartitioning(Partitioning):
    """
    Partition a Table into chunks of a given fixed size.

    :param chunk_size: The size of each chunk.
    """

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def partition(self, table: T) -> Iterator[T]:
        """
        Partition a Table into chunks of the given fixed size.
        """
        for i in range(0, len(table), self.chunk_size):
            yield table[i : i + self.chunk_size]


class GroupedPartitioning(Partitioning):
    """
    Partition a Table into groups based on the unique values in a given column.

    :param group_column: The name of the column to group by.
    """

    def __init__(self, group_column: str):
        self.group_column = group_column

    def partition(self, table: T) -> Iterator[T]:
        for group in table.column(self.group_column).unique():
            mask = pc.equal(table.column(self.group_column), group)
            yield table.apply_mask(mask)


def execute_parallel(
    table: T,
    func: Callable[..., Any],
    *args: Any,
    max_workers: int = 4,
    partitioning: Partitioning = ChunkedPartitioning(chunk_size=1000),
    **kwargs: Any,
) -> Iterator[Any]:
    """Execute a function in parallel on a Table.

    This function partitions the Table into multiple Tables, and executes the function
    on each partition in parallel. The results are returned as a Python list.

    :param table: The Table to execute the function on.

    :param func: The function to execute. The function takes an
        instance of the Table (a subslice of it) as its first
        argument, and can optionally take additional arguments and
        keyword arguments, which must be passed in to
        ``execute_parallel`` as ``args`` and ``kwargs``.

    :param max_workers: The maximum number of workers to use.

    :param partitioning: The partitioning strategy to use. The default
        is to partition the Table into chunks of 1000 rows. The
        partitioning's ``partition`` method will be called with the
        Table as its only argument, and the values from the iterator
        will be fed to a worker pool one-by-one.

    :param args: Additional arguments to pass to the function. These
        must be pickleable.

    :param kwargs: Additional keyword arguments to pass to the
        function. The values must be pickleable.

    :return: An iterator over the results of the function, as applied
        to each partition of the table. The results are returned in
        an arbitrary order as they are completed.

    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    args_list = list(args)

    # Create a pool of workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a shared memory object
        with managers.SharedMemoryManager() as mgr:
            # If there are tables in the args or kwargs, convert them to
            # shared memory objects.
            for i, arg in enumerate(args):
                if isinstance(arg, qv.Table):
                    args_list[i] = TableReference.from_instance(arg, mgr)

            for key, value in kwargs.items():
                if isinstance(value, qv.Table):
                    kwargs[key] = TableReference.from_instance(value, mgr)

            # Execute the function in parallel
            futures = []
            for partition in partitioning.partition(table):
                ref = TableReference.from_instance(partition, mgr)
                future = executor.submit(
                    _run_on_shared_memory,
                    ref,
                    func,
                    args_list,
                    kwargs,
                )
                futures.append(future)

            # Wait for the results
            for future in concurrent.futures.as_completed(futures):
                yield future.result()


@dataclasses.dataclass
class TableReference(Generic[T]):
    """
    A reference to a Table in shared memory.

    This class is used to pass a reference to a Table to a worker
    process, without actually copying the Table into the worker
    process's memory. The worker process can then load the Table from
    shared memory using the ``load`` method.
    """

    shm_name: str
    table_class: type[T]

    def load(self) -> T:
        """
        Load the Table from shared memory.
        """
        shm = shared_memory.SharedMemory(name=self.shm_name)
        return from_shared_memory(shm, self.table_class)

    @classmethod
    def from_instance(cls, instance: T, mgr: managers.SharedMemoryManager) -> Self:
        """
        Create a TableReference from a Table instance.
        """
        shm = to_shared_memory(instance, mgr)
        return cls(
            shm.name,
            instance.__class__,
        )
