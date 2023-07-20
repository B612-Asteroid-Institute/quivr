from quivr import *

class TaxiData(Table):
    vendor_id = Int64Column()
    pickup = TimestampColumn(unit="us")
    dropoff = TimestampColumn(unit="us")
    passenger_count = Float64Column()
    trip_distance = Float64Column()
    rate_code = Float64Column()
    ...

column_name_mapping = {
    "VendorID": "vendor_id",
    "tpep_pickup_datetime": "pickup",
    "tpep_dropoff_datetime": "dropoff",
    "RatecodeID": "rate_code",
}

data = TaxiData.from_parquet(
    "yellow__tripdata_2023-01.parquet",
    column_name_mapping=column_name_mapping,
)
print(data)
# TaxiData(size=3066766)

print(data.pickup[:2])
# [
#    2023-01-01 00:32:10.000000,
#    2023-01-01 00:55:08.000000
# ]
