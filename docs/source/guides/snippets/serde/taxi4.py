from quivr import *

class TaxiData(Table):
    vendor_id = UInt8Column()
    pickup = TimestampColumn(unit="us")
    dropoff = TimestampColumn(unit="us")
    passenger_count = UInt8Column()
    trip_distance = Float64Column()
    rate_code = UInt8Column()

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
            column_name_map=column_name_mapping,
        )

taxi_data = TaxiData.from_parquet("./yellow__tripdata_2023-01.parquet")
