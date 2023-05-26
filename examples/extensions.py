import pyarrow as pa
import enum
from quivr import Table, ExtensionField, Float64Field


class ReferenceFrame(enum.Enum):
    heliocentric = 1
    barycentric = 2
    geocentric = 3


class ReferenceFrameScalar(pa.ExtensionScalar):
    def as_py(self) -> ReferenceFrame:
        return ReferenceFrame(self.value.as_py())


class ReferenceFrameArray(pa.ExtensionArray):
    def to_string_list(self) -> list[str]:
        return [v.as_py().name for v in self]

    def all_equal(self):
        return self.storage.all_equal()


class ReferenceFrameType(pa.PyExtensionType):
    def __init__(self):
        pa.PyExtensionType.__init__(self, pa.int8())

    def __reduce__(self):
        return (ReferenceFrameType, ())

    def __arrow_ext_scalar_class__(self):
        return ReferenceFrameScalar

    def __arrow_ext_class__(self):
        return ReferenceFrameArray


class ReferenceFrameField(ExtensionField):
    def __init__(self, nullable=False, validator=None, metadata=None):
        dtype = ReferenceFrameType()
        ExtensionField.__init__(self, dtype, nullable, validator, metadata)

    def __get__(self, obj: Table, objtype: type):
        raw = super().__get__(obj, objtype)
        return raw.to_string_list()


class Position(Table):
    x = Float64Field()
    y = Float64Field()
    z = Float64Field()
    frame = ReferenceFrameField()
