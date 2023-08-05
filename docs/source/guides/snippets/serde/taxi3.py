import quivr as qv

class TaxiData(qv.Table):
    vendor_id = qv.Int64Column()
    pickup = qv.TimestampColumn(unit="us")
    dropoff = qv.TimestampColumn(unit="us")
    passenger_count = qv.Float64Column()
    trip_distance = qv.Float64Column()
    rate_code = qv.Float64Column()

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
