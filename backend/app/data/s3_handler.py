# S3 i/o
import pandas as pd
import boto3
from settings import *
import io

class DataHandler:
    def __init__(self):
        self.s3_client = boto3.client('s3')

    def save_to_s3(self, df, file_path):
        if df is None or df.empty:
            print("DataFrame is empty, skipping S3 upload.")
            return

        print(f"Saving data to S3 at s3://{S3_BUCKET_NAME}/{file_path}...")
        try:

            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, engine='pyarrow')

            self.s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=file_path,
                Body=parquet_buffer.getvalue()
            )
            print("Successfully uploaded to S3.")

        except Exception as e:
            print(f"Error uploading to S3: {e}")

    def get_dfs_from_s3(self, prefix=''):

        if self.s3_client is None:
            self.s3_client = boto3.client('s3')

        list_of_dfs = []

        try:
            response = self.s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)

            if 'Contents' not in response:
                print(f"No objects found in bucket '{S3_BUCKET_NAME}' with prefix '{prefix}'.")
                return list_of_dfs

            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.parquet'):
                    print(f"Reading file: {key}")

                    file_object = self.s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)

                    df = pd.read_parquet(io.BytesIO(file_object['Body'].read()))
                    list_of_dfs.append(df)

        except Exception as e:
            print(f"Error retrieving data from S3: {e}")
            return []

        return list_of_dfs
