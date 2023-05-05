import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

from quivr import Table, Float64Field, ListField, StringField


class CartesianCoordinates(Table):
    x = Float64Field()
    y = Float64Field()
    z = Float64Field()
    vx = Float64Field()
    vy = Float64Field()
    vz = Float64Field()
    covariance = ListField(pa.float64())

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
    coords = CartesianCoordinates.as_field()
    epoch = Float64Field()
    object_id = StringField()


class Etc(Table):
    orbit = Orbit.as_field()
    thing = Float64Field()
