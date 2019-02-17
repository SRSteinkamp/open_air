import pandas as pd
from influxdb import DataFrameClient


def get_db_from_csv(file):
    DF = pd.read_csv(file)

    return DF['host'].item(), DF['database'].item()


def get_db_client(host_name, data_base) -> DataFrameClient:
    client = DataFrameClient(host=host_name,
                             port=443,
                             database=data_base)
    return client


def query_influx(host_name: str, data_base: str, query: str) -> dict:
    client = get_db_client(host_name, data_base)
    result = client.query(query)
    client.close()
    return result
