from dotenv import load_dotenv
import os

load_dotenv()

MODEL_ARTIFACTS_PREFIX = 'pytorch-training-2025-10-03-15-34-22-625/output/model.tar.gz'
S3_BUCKET_NAME = "stock-screener-bucker"

# environ.get?
DB_NAME = os.getenv("RDS_DB_NAME")
DB_USERNAME = os.getenv("RDS_USERNAME")
DB_PASSWORD = os.getenv("RDS_PASSWORD")
DB_HOST = os.getenv("RDS_HOST")
DB_PORT = os.getenv("PORT")

AWS_ACC_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SCR_ACC_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")