import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

from quivr import Float64Column, ListColumn, StringColumn, Table


class CartesianCoordinates(Table):
    x = Float64Column()
    y = Float64Column()
    z = Float64Column()
    vx = Float64Column()
    vy = Float64Column()
    vz = Float64Column()
    covariance = ListColumn(pa.float64())

    def covariance_matrix(self) -> npt.NDArray[np.float64]:
        raw = self.column("covariance").to_numpy()
        stacked = np.stack(raw)
        return stacked.reshape((len(stacked), 6, 6))

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


class Orbit(Table):
    coords = CartesianCoordinates.as_column()
    epoch = Float64Column()
    object_id = StringColumn()


class Etc(Table):
    orbit = Orbit.as_column()
    thing = Float64Column()
