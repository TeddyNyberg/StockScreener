from dotenv import load_dotenv
import os
from pathlib import Path

current_file_path = Path(__file__).resolve()
backend_dir = current_file_path.parent
root_dir = backend_dir.parent
env_path = root_dir / ".env"
load_dotenv(dotenv_path=env_path)

DB_NAME = os.getenv("POSTGRES_DB")
DB_USERNAME = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("RDS_HOST")
DB_PORT = os.getenv("PORT")

AWS_ACC_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SCR_ACC_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

S3_BUCKET_NAME = os.getenv("PROD_MODEL_BUCKET")
S3_TRAINING_BUCKET = os.getenv("DEV_TRAINING_BUCKET")

TOKEN_SCR_KEY = os.getenv("JWT_SECRET_KEY")