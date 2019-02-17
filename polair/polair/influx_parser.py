import pandas as pd
from influxdb import DataFrameClient, resultset, InfluxDBClient

def get_db_from_txt(file):

    return 

def get_db_client(host_name, data_base) -> DataFrameClient:
    client = DataFrameClient(host=hose_name,
                             port=443,
                             database=data_base)
    return client


def query_influx(query: str) -> dict:
    client = get_db_client()
    result = client.query(query)
    client.close()
    return result