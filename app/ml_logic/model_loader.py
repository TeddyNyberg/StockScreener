import boto3
import io
import tarfile
import torch
import gzip
from settings import *
from config import *

MODEL_CACHE = {}

S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id=AWS_ACC_KEY_ID,
    aws_secret_access_key=AWS_SCR_ACC_KEY
    )

# for static model use model_key= MODEL_MAP["A"]["prefix"]
def load_model_artifacts(model_key = MODEL_MAP["A"]["prefix"]):
    global MODEL_CACHE

    if model_key in MODEL_CACHE:
        print(f"Using cached model artifacts for: {model_key}")
        return MODEL_CACHE[model_key]

    print(f"Loading model artifacts from S3: {model_key} in bucket {S3_TRAINING_BUCKET}...")

    model_buffer = io.BytesIO()

    S3_CLIENT.download_fileobj(S3_TRAINING_BUCKET, model_key, model_buffer)
    model_buffer.seek(0)
    print("Downloaded model archive to memory.")

    checkpoint = None
    try:
        with tarfile.open(fileobj=model_buffer, mode="r:gz") as tar:
            with tar.extractfile("model.pth") as f:
                checkpoint = torch.load(io.BytesIO(f.read()), map_location="cpu")
        print("Loaded checkpoint from tar.gz archive.")
    except tarfile.ReadError:
        model_buffer.seek(0)
        try:
            with gzip.GzipFile(fileobj=model_buffer, mode="rb") as gz:
                checkpoint = torch.load(io.BytesIO(gz.read()), map_location="cpu")
            print("Loaded checkpoint from gz file.")
        except OSError:
            model_buffer.seek(0)
            checkpoint = torch.load(model_buffer, map_location="cpu")
            print("Loaded checkpoint from uncompressed file.")

    MODEL_STATE_DICT = checkpoint.get("model_state")
    CONFIG = checkpoint.get("config")

    if MODEL_STATE_DICT is None:
        raise KeyError("Could not find 'model_state' key in the loaded dictionary.")

    print(f"Model state dict and config loaded successfully.")

    MODEL_CACHE[model_key] = (MODEL_STATE_DICT, CONFIG)

    return MODEL_STATE_DICT, CONFIG


def save_model_artifacts(model_state_dict, config, s3_key):

    checkpoint = {
        "model_state": model_state_dict,
        "config": config,
    }

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    try:
        S3_CLIENT.upload_fileobj(
            buffer,
            S3_TRAINING_BUCKET,
            s3_key
        )
        MODEL_CACHE[s3_key] = (model_state_dict, config)
        print(f"Model artifacts successfully saved to s3://{S3_TRAINING_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"Error saving model to S3 at {s3_key}: {e}")


def load_model_artifacts_local(filename):
    global MODEL_CACHE

    if filename in MODEL_CACHE:
        print(f"Using cached model artifacts for: {filename}")
        return MODEL_CACHE[filename]

    loaded_data = torch.load(filename)

    loaded_state_dict = loaded_data.get("model_state")
    loaded_config = loaded_data.get("config")

    if loaded_state_dict is None:
        raise KeyError("Could not find 'model_state' key in the loaded dictionary.")

    MODEL_CACHE[filename] = (loaded_state_dict, loaded_config)

    return loaded_state_dict, loaded_config