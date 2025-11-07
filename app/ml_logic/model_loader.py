import boto3
import io
import tarfile
import torch
from settings import *

MODEL_CACHE = {}

S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id=AWS_ACC_KEY_ID,
    aws_secret_access_key=AWS_SCR_ACC_KEY
    )

# for static model use model_key=MODEL_ARTIFACTS_PREFIX
def load_model_artifacts(model_key):
    global MODEL_CACHE

    if model_key in MODEL_CACHE:
        print(f"Using cached model artifacts for: {model_key}")
        return MODEL_CACHE[model_key]

    print(f"Loading model artifacts from S3: {model_key} in bucket {S3_TRAINING_BUCKET}...")

    model_buffer = io.BytesIO()

    S3_CLIENT.download_fileobj(S3_TRAINING_BUCKET, model_key, model_buffer)
    model_buffer.seek(0)
    print("Downloaded model archive to memory.")

    with tarfile.open(fileobj=model_buffer, mode='r:gz') as tar:
        with tar.extractfile('model.pth') as f:
            checkpoint = torch.load(io.BytesIO(f.read()), map_location='cpu')

    MODEL_STATE_DICT = checkpoint.get("model_state")
    CONFIG = checkpoint.get("config")

    if MODEL_STATE_DICT is None:
        raise KeyError("Could not find 'model_state' key in the loaded dictionary.")

    print(f"Model state dict and config loaded successfully.")

    MODEL_CACHE[model_key] = (MODEL_STATE_DICT, CONFIG)

    return MODEL_STATE_DICT, CONFIG