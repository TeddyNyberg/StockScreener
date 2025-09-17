import sagemaker
from sagemaker.pytorch import PyTorch
import boto3


boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
sagemaker_role = 'arn:aws:iam::307926602475:role/service-role/AmazonSageMaker-ExecutionRole-20250905T114899'
data_location = 's3://stock-screener-bucker/historical_data/'
print("IN SAGE.PY")

# Define the PyTorch estimator
estimator = PyTorch(
    source_dir='app',
    entry_point='modeltosm.py',
    role=sagemaker_role,
    session=sagemaker_session,
    instance_count=1,
    instance_type='ml.m5.4xlarge',
    framework_version='1.13',
    py_version='py39',
    hyperparameters={'epochs': "10", 'batch_size': "64", "learning_rate": "0.001"}
)

# Start the training job
estimator.fit({'training': data_location})

