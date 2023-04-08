from quiver import TableBase

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


def create_example_orbits(n_orbits: int):
    data = np.random.random((6, n_orbits))

    xs = pa.array(data[0], pa.float64())
    ys = pa.array(data[1], pa.float64())
    zs = pa.array(data[2], pa.float64())
    vxs = pa.array(data[3], pa.float64())
    vys = pa.array(data[4], pa.float64())
    vzs = pa.array(data[5], pa.float64())
    covs = pa.array(list(np.random.random((n_orbits, 36))), pa.list_(pa.float64(), 36))

    coords = CartesianCoordinates.from_arrays([xs, ys, zs, vxs, vys, vzs, covs])

    coords_sa = coords.to_structarray()
    epochs = pa.array(np.random.random(n_orbits), pa.float64())
    ids = pa.array([f"id{i}" for i in range(n_orbits)], type=pa.string())

    return Orbit.from_arrays([coords_sa, epochs, ids])
