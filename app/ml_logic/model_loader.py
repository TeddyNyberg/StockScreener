import boto3
import io
import tarfile
import torch
from settings import *


s3_client = boto3.client(
        's3',
        aws_access_key_id = AWS_ACC_KEY_ID,
        aws_secret_access_key= AWS_SCR_ACC_KEY
    )


_MODEL_STATE_DICT = None
_CONFIG = None
def load_model_artifacts():
    global _MODEL_STATE_DICT, _CONFIG

    if _MODEL_STATE_DICT is not None and _CONFIG is not None:
        print("Using cached model artifacts.")
        return _MODEL_STATE_DICT, _CONFIG

    print("Loading model artifacts from S3...")

    model_buffer = io.BytesIO()

    s3_client.download_fileobj("sagemaker-us-east-1-307926602475", MODEL_ARTIFACTS_PREFIX, model_buffer)
    model_buffer.seek(0)
    print("Downloaded entire model archive to memory.")

    with tarfile.open(fileobj=model_buffer, mode='r:gz') as tar:
        with tar.extractfile('model.pth') as f:
            checkpoint = torch.load(io.BytesIO(f.read()))

    _MODEL_STATE_DICT = checkpoint.get("model_state")
    _CONFIG = checkpoint.get("config")

    if _MODEL_STATE_DICT is None:
        raise KeyError("Could not find 'model_state' key in the loaded dictionary.")

    print("Model state dict and config loaded successfully.")
    return _MODEL_STATE_DICT, _CONFIG