import pyarrow as pa
import numpy as np
import functools


class MatrixExtensionType(pa.PyExtensionType):
    def __init__(self, shape: tuple[int, int], element_dtype: pa.DataType):
        self.shape = shape
        self.element_dtype = element_dtype
        self.dtype = element_dtype
        for dim in reversed(self.shape):
            self.dtype = pa.list_(self.dtype)
        pa.PyExtensionType.__init__(self, self.dtype)

    def __reduce__(self):
        return (
            MatrixExtensionType,
            (self.shape, self.element_dtype),
        )

    def __arrow_ext_class__(self):
        return MatrixArray


class MatrixArray(pa.ExtensionArray):
    def to_numpy(self):
        storage = self.storage
        row_size = 1
        for dim in reversed(self.type.shape):
            storage = storage.flatten()
            row_size *= dim
        np_array = storage.to_numpy()

        n_row = len(storage) // row_size
        return np_array.reshape((n_row, *self.type.shape))


def make_array_working():
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    storage = pa.array(data, pa.list_(pa.list_(pa.float64())))
    typ = MatrixExtensionType((2, 2), pa.float64())
    return MatrixArray.from_storage(typ, storage)


def make_struct_array_not_working():
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    typ = MatrixExtensionType((2, 2), pa.float64())
    struct_typ = pa.struct([("data", typ)])
    return pa.array(data, type=struct_typ)
