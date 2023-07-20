from quivr import *

class TaxiData(Table):
    VendorID = Int64Column()
    tpep_pickup_datetime = TimestampColumn(unit="us")
    tpep_dropoff_datetime = TimestampColumn(unit="us")
    passenger_count = Float64Column()
    trip_distance = Float64Column()
    RatecodeID = Float64Column()
    ...


data = TaxiData.from_parquet("yellow__tripdata_2023-01.parquet")
print(data)
# TaxiData(size=3066766)
