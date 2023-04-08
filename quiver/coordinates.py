from .models import BaseModel
from .matrix import MatrixExtensionType, MatrixArray
import pyarrow as pa


class CartesianCoordinates(BaseModel):
    schema = pa.schema(
        [
            pa.field("x", pa.float64()),
            pa.field("y", pa.float64()),
            pa.field("z", pa.float64()),
            pa.field("vx", pa.float64()),
            pa.field("vy", pa.float64()),
            pa.field("vz", pa.float64()),
            pa.field("covariance", MatrixExtensionType((6, 6), pa.float64())),
        ]
    )


class Orbit(BaseModel):
    schema = pa.schema(
        [
            CartesianCoordinates.as_field("coords"),
            pa.field("epoch", pa.float64()),
            pa.field("object_id", pa.string()),
        ]
    )

