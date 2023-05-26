import struct
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .table import Table


class Attribute:
    """
    An Attribute is an accessor for scalar data in a table.

    It works through the descriptor protocol.

    All attributes must be serializable into bytes.

    Attributes are stored as metadata on the underlying Arrow table's schema.
    """

    def __init__(self, default: Any = None):
        self.default = default
        self.name = "__ERR_UNSET_NAME"

    def __get__(self, instance: "Table", owner):
        if instance.table.schema.metadata is None:
            return self.default
        name = self.name.encode("utf8")
        if name not in instance.table.schema.metadata:
            return self.default
        raw = instance.table.schema.metadata[name]
        return self.from_bytes(raw)

    def __set__(self, instance, value):
        metadata = instance.table.schema.metadata
        if metadata is None:
            metadata = {}
        metadata[self.name.encode("utf8")] = self.to_bytes(value)
        instance.table = instance.table.replace_schema_metadata(metadata)

    def __set_name__(self, owner: type, name: str):
        self.name = name

    def to_bytes(self, value) -> bytes:
        """
        Convert the value to bytes.
        """
        raise NotImplementedError

    def from_bytes(self, raw: bytes):
        """
        Convert the bytes to a value.
        """
        raise NotImplementedError


class StringAttribute(Attribute):
    def __init__(self, default: Optional[str] = None):
        super().__init__(default=default)

    def to_bytes(self, value) -> bytes:
        return value.encode("utf8")

    def from_bytes(self, raw: bytes):
        return raw.decode("utf8")


class IntAttribute(Attribute):
    def __init__(self, default: Optional[int] = None, nbytes: int = 8, signed: bool = True):
        self.nbytes = nbytes
        self.signed = signed
        super().__init__(default=default)

    def to_bytes(self, value: int) -> bytes:
        return value.to_bytes(length=self.nbytes, byteorder="little", signed=self.signed)

    def from_bytes(self, raw: bytes) -> int:
        return int.from_bytes(raw, byteorder="little", signed=self.signed)


class FloatAttribute(Attribute):
    def __init__(self, default: Optional[float] = None, nbytes: int = 8):
        if nbytes == 8:
            self._struct_fmt = "<d"
        elif nbytes == 4:
            self._struct_fmt = "<f"
        elif nbytes == 2:
            self._struct_fmt = "<e"
        else:
            raise ValueError("nbytes must be 2, 4 or 8")
        super().__init__(default=default)

    def to_bytes(self, value: float) -> bytes:
        return struct.pack(self._struct_fmt, value)

    def from_bytes(self, raw: bytes) -> float:
        return struct.unpack(self._struct_fmt, raw)[0]
