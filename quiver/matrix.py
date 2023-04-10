import pyarrow as pa


class MatrixExtensionType(pa.PyExtensionType):
    """This is a custom type for embedding multi-dimensional arrays
    into Table schemas.

    """

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
