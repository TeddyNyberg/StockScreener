from dotenv import load_dotenv
import os

load_dotenv()

DB_NAME = os.getenv("POSTGRES_DB")
DB_USERNAME = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("RDS_HOST")
DB_PORT = os.getenv("PORT")

AWS_ACC_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SCR_ACC_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

S3_BUCKET_NAME = os.getenv("PROD_MODEL_BUCKET")
S3_TRAINING_BUCKET = os.getenv("DEV_TRAINING_BUCKET")