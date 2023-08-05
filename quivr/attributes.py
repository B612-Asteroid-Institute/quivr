import struct
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

from . import errors

if TYPE_CHECKING:
    from .tables import Table

T = TypeVar("T")


class Attribute(Generic[T]):
    """
    An Attribute is an accessor for scalar data in a table.

    It works through the descriptor protocol.

    All attributes must be serializable into bytes.

    Attributes are stored as metadata on the underlying Arrow table's schema.
    """

    _type: type[T]

    def __init__(
        self,
        mutable: bool = False,
        default: Optional[T] = None,
    ):
        self.default = default
        self.mutable = mutable
        self.name = "__ERR_UNSET_NAME"

    def __get__(self, instance: "Table", owner: type) -> T:
        if instance is None:
            return self
        name = self.name.encode("utf8")
        if instance.table.schema.metadata is None or name not in instance.table.schema.metadata:
            if self.default is not None:
                return self.default
            raise AttributeError(f"Attribute {self.name} is not set and has no default")
        raw = instance.table.schema.metadata[name]
        return self.from_bytes(raw)

    def __set__(self, instance: "Table", value: T) -> None:
        name = self.name.encode("utf8")
        metadata = instance.table.schema.metadata
        if metadata is None:
            metadata = {}
        if name in metadata and not self.mutable:
            raise errors.AttributeImmutableError(f"Attribute {self.name} is not mutable")
        metadata[self.name.encode("utf8")] = self.to_bytes(value)
        instance.table = instance.table.replace_schema_metadata(metadata)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def to_bytes(self, value: T) -> bytes:
        """
        Convert the value to bytes.
        """
        raise NotImplementedError

    def from_bytes(self, raw: bytes) -> T:
        """
        Convert the bytes to a value.
        """
        raise NotImplementedError

    def to_string(self, value: T) -> str:
        """
        Convert the value to a string. Used for CSV writing.
        """
        raise NotImplementedError

    def from_string(self, raw: str) -> T:
        """
        Convert a string to a value. Used for CSV reading.
        """
        raise NotImplementedError


class StringAttribute(Attribute[str]):
    """StringAttribute represents a string which is stored as UTF-8
    bytes in Table metadata.

    :param default: The default value for this attribute. If no default is provided,
        then the attribute must be set whenever constructing a table that uses it.
    """

    _type = str

    def __init__(self, default: Optional[str] = None, mutable: bool = False):
        super().__init__(default=default, mutable=mutable)

    def to_bytes(self, value: str) -> bytes:
        return value.encode("utf8")

    def from_bytes(self, raw: bytes) -> str:
        return raw.decode("utf8")

    def to_string(self, value: str) -> str:
        return value

    def from_string(self, raw: str) -> str:
        return raw


class IntAttribute(Attribute[int]):
    """IntAttribute represents an integer which is stored as little-endian
    bytes in Table metadata.

    The number of bytes used to store the integer must be specified
    up-front, as well as whether it is signed.

    :param default: The default value for this attribute. If no default is provided,
        then the attribute must be set whenever constructing a table that uses it.
    :param nbytes: The number of bytes to use to store the integer. Must be 1, 2, 4 or 8.
    :param signed: Whether the integer is signed or unsigned.
    """

    _type = int

    def __init__(
        self, default: Optional[int] = None, mutable: bool = False, nbytes: int = 8, signed: bool = True
    ):
        self.nbytes = nbytes
        self.signed = signed
        super().__init__(default=default, mutable=mutable)

    def to_bytes(self, value: int) -> bytes:
        return value.to_bytes(length=self.nbytes, byteorder="little", signed=self.signed)

    def from_bytes(self, raw: bytes) -> int:
        return int.from_bytes(raw, byteorder="little", signed=self.signed)

    def to_string(self, value: int) -> str:
        return str(value)

    def from_string(self, raw: str) -> int:
        return int(raw)


class FloatAttribute(Attribute[float]):
    """
    FloatAttribute represents a floating-point number which is stored as little-endian
    bytes in Table metadata.

    The number of bytes used to store the float must be specified
    up-front.

    :param default: The default value for this attribute. If no default is provided,
        then the attribute must be set whenever constructing a table that uses it.
    :param nbytes: The number of bytes to use to store the float. Must be 2, 4 or 8.
    """

    _type = float

    def __init__(self, default: Optional[float] = None, mutable: bool = False, nbytes: int = 8):
        if nbytes == 8:
            self._struct_fmt = "<d"
        elif nbytes == 4:
            self._struct_fmt = "<f"
        elif nbytes == 2:
            self._struct_fmt = "<e"
        else:
            raise ValueError("nbytes must be 2, 4 or 8")
        super().__init__(default=default, mutable=mutable)

    def to_bytes(self, value: float) -> bytes:
        return struct.pack(self._struct_fmt, value)

    def from_bytes(self, raw: bytes) -> float:
        return float(struct.unpack(self._struct_fmt, raw)[0])

    def to_string(self, value: float) -> str:
        return "{:.17g}".format(value)

    def from_string(self, raw: str) -> float:
        return float(raw)
