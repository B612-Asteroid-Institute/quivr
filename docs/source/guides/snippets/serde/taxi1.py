import quivr as qv

class TaxiData(qv.Table):
    VendorID = qv.Int64Column()
    tpep_pickup_datetime = qv.TimestampColumn(unit="us")
    tpep_dropoff_datetime = qv.TimestampColumn(unit="us")
    passenger_count = qv.Float64Column()
    trip_distance = qv.Float64Column()
    RatecodeID = qv.Float64Column()
    ...


data = TaxiData.from_parquet("yellow__tripdata_2023-01.parquet")
print(data)
# TaxiData(size=3066766)
