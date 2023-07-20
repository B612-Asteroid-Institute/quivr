from quivr import *

class TaxiData(Table):
    vendor_id = Int64Column()
    pickup = TimestampColumn(unit="us")
    dropoff = TimestampColumn(unit="us")
    passenger_count = Float64Column()
    trip_distance = Float64Column()
    rate_code = Float64Column()

    @classmethod
    def from_parquet(cls, path):
        column_name_mapping = {
            "VendorID": "vendor_id",
            "tpep_pickup_datetime": "pickup",
            "tpep_dropoff_datetime": "dropoff",
            "RatecodeID": "rate_code",
        }
        return super().from_parquet(
            path,
            column_name_mapping=column_name_mapping,
        )

taxi_data = TaxiData.from_parquet("./yellow_tripdata_2023-01.parquet")
