import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
from sagemaker.tuner import IntegerParameter, ContinuousParameter, CategoricalParameter, HyperparameterTuner

boto_session = boto3.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto_session=boto_session)
sagemaker_role = 'arn:aws:iam::307926602475:role/service-role/AmazonSageMaker-ExecutionRole-20250905T114899'
data_location = 's3://stock-screener-bucker/historical_data/'
print("IN SAGE.PY")



hyperparameter_ranges = {
    'epochs': IntegerParameter(3, 20),
    'batch_size': CategoricalParameter([32, 64, 128]),
    'learning_rate': CategoricalParameter([0.0001, 0.001, 0.01, 0.1])
}

# Define the PyTorch estimator
estimator = PyTorch(
    source_dir='app',
    entry_point='newattemptmodel.py',
    role=sagemaker_role,
    session=sagemaker_session,
    instance_count=1,
    instance_type='ml.m5.4xlarge',
    framework_version='1.13',
    py_version='py39',
    hyperparameters={'epochs': "50", 'batch_size': "64", "learning_rate": "0.001"}
)
estimator.fit({'training': data_location})

"""
objective_metric_name = 'test-loss'
metric_definitions = [
    {'Name': objective_metric_name, 'Regex': 'Average Test Loss: (.*)'}
]

# 3. Create the HyperparameterTuner
tuner = HyperparameterTuner(
    estimator,
    objective_metric_name,
    hyperparameter_ranges,
    metric_definitions,
    max_jobs=9, # Total number of jobs to run
    max_parallel_jobs=3, # Number of jobs to run in parallel
    objective_type='Minimize',
    strategy='Random' # Use 'Random' or 'Bayesian' for more efficient searches
)

# Start the tuning job
tuner.fit({'training': data_location})

"""