import quivr as qv

class TaxiData(qv.Table):
    VendorID = qv.Int64Column(nullable=True)
    tpep_pickup_datetime = qv.TimestampColumn(unit="us", nullable=True)
    tpep_dropoff_datetime = qv.TimestampColumn(unit="us", nullable=True)
    passenger_count = qv.Float64Column(nullable=True)
    trip_distance = qv.Float64Column(nullable=True)
    RatecodeID = qv.Float64Column(nullable=True)
    ...


data = TaxiData.from_parquet("yellow__tripdata_2023-01.parquet")
print(data)
# TaxiData(size=3066766)
