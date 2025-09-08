import sagemaker
from sagemaker.pytorch import PyTorch


sagemaker_session = sagemaker.Session()

# Define the S3 path for your data
data_location = 's3://stock-screener-bucker/historical_data/'

# Define the PyTorch estimator
estimator = PyTorch(
    entry_point='model.py',
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    framework_version='1.9',
    py_version='py38',
    hyperparameters={'epochs': 10, 'batch_size': 64}
)

# Start the training job
estimator.fit({'training': data_location})