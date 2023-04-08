from .models import TableBase
from .matrix import MatrixExtensionType, MatrixArray
import pyarrow as pa
import numpy as np
import numpy.typing as npt


class CartesianCoordinates(TableBase):
    schema = pa.schema(
        [
            pa.field("x", pa.float64()),
            pa.field("y", pa.float64()),
            pa.field("z", pa.float64()),
            pa.field("vx", pa.float64()),
            pa.field("vy", pa.float64()),
            pa.field("vz", pa.float64()),
            pa.field("covariance", pa.list_(pa.float64(), 36)),
        ]
    )

    def covariance_matrix(self) -> npt.NDArray[np.float64]:
        raw = self.column("covariance").to_numpy()
        stacked = np.stack(raw)
        return stacked.reshape((len(stacked), 6, 6))


class Orbit(TableBase):
    schema = pa.schema(
        [
            CartesianCoordinates.as_field("coords"),
            pa.field("epoch", pa.float64()),
            pa.field("object_id", pa.string()),
        ]
    )

