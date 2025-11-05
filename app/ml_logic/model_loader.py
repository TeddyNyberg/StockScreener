import boto3
import io
import tarfile
import torch
from settings import *


S3_CLIENT = boto3.client(
        's3',
        aws_access_key_id = AWS_ACC_KEY_ID,
        aws_secret_access_key= AWS_SCR_ACC_KEY
    )


MODEL_STATE_DICT = None
CONFIG = None
def load_model_artifacts():
    global MODEL_STATE_DICT, CONFIG

    if MODEL_STATE_DICT is not None and CONFIG is not None:
        print("Using cached model artifacts.")
        return MODEL_STATE_DICT, CONFIG

    print("Loading model artifacts from S3...")

    model_buffer = io.BytesIO()

    S3_CLIENT.download_fileobj("sagemaker-us-east-1-307926602475", MODEL_ARTIFACTS_PREFIX, model_buffer)
    model_buffer.seek(0)
    print("Downloaded entire model archive to memory.")

    with tarfile.open(fileobj=model_buffer, mode='r:gz') as tar:
        with tar.extractfile('model.pth') as f:
            checkpoint = torch.load(io.BytesIO(f.read()))

    MODEL_STATE_DICT = checkpoint.get("model_state")
    CONFIG = checkpoint.get("config")

    if MODEL_STATE_DICT is None:
        raise KeyError("Could not find 'model_state' key in the loaded dictionary.")

    print("Model state dict and config loaded successfully.")
    return MODEL_STATE_DICT, CONFIG