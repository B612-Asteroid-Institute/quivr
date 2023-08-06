import quivr as qv

class TaxiData(qv.Table):
    vendor_id = qv.Int64Column(nullable=True)
    pickup = qv.TimestampColumn(unit="us", nullable=True)
    dropoff = qv.TimestampColumn(unit="us", nullable=True)
    passenger_count = qv.Float64Column(nullable=True)
    trip_distance = qv.Float64Column(nullable=True)
    rate_code = qv.Float64Column(nullable=True)
    ...

column_name_mapping = {
    "VendorID": "vendor_id",
    "tpep_pickup_datetime": "pickup",
    "tpep_dropoff_datetime": "dropoff",
    "RatecodeID": "rate_code",
}

data = TaxiData.from_parquet(
    "yellow__tripdata_2023-01.parquet",
    column_name_map=column_name_mapping,
)
print(data)
# TaxiData(size=3066766)

print(data.pickup[:2])
# [
#    2023-01-01 00:32:10.000000,
#    2023-01-01 00:55:08.000000
# ]
