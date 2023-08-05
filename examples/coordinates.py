import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

import quivr as qv


class CartesianCoordinates(qv.Table):
    x = qv.Float64Column()
    y = qv.Float64Column()
    z = qv.Float64Column()
    vx = qv.Float64Column()
    vy = qv.Float64Column()
    vz = qv.Float64Column()
    covariance = qv.ListColumn(pa.float64())

    def covariance_matrix(self) -> npt.NDArray[np.float64]:
        raw = self.column("covariance").to_numpy()
        stacked = np.stack(raw)
        return stacked.reshape((len(stacked), 6, 6))  # type: ignore

    def distance(self) -> pa.Array:
        return pc.sqrt(
            pc.add(
                pc.add(
                    pc.multiply(self.x, self.x),
                    pc.multiply(self.y, self.y),
                ),
                pc.multiply(self.z, self.z),
            )
        )


class Orbit(qv.Table):
    coords = CartesianCoordinates.as_column()
    epoch = qv.Float64Column()
    object_id = qv.StringColumn()


class Etc(qv.Table):
    orbit = Orbit.as_column()
    thing = qv.Float64Column()
