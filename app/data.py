from dotenv import load_dotenv
import os
import yfinance as yf
import pandas as pd
import boto3

load_dotenv()  # This loads the variables from the .env file

access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")



def get_sp500_tickers():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(sp500_url, header=0)
    sp500_table = html[0]
    tickers = sp500_table['Symbol'].tolist()
    return tickers

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def save_to_s3(dataframe, bucket_name, object_name):
    s3_client = boto3.client('s3')
    dataframe.to_parquet(f'/tmp/{object_name}')
    s3_client.upload_file(f'/tmp/{object_name}', bucket_name, object_name)
    print(f"Successfully uploaded {object_name} to S3 bucket {bucket_name}")