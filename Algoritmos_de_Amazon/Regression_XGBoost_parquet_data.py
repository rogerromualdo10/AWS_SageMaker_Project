import os
import boto3
import re
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()
region = boto3.Session().region_name

# S3 bucket for saving code and model artifacts.
# Feel free to specify a different bucket here if you wish.
bucket = sagemaker.Session().default_bucket()
prefix = 'sagemaker/DEMO-xgboost-parquet'
bucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region, bucket)

#Install PyArrow
python -m pip install pyarrow==0.15

import numpy as np
import pandas as pd
import urllib.request
from sklearn.datasets import load_svmlight_file

# Download the dataset and load into a pandas dataframe
FILE_NAME = 'abalone.csv'
urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", FILE_NAME)
feature_names=['Sex', 
               'Length', 
               'Diameter', 
               'Height', 
               'Whole weight', 
               'Shucked weight', 
               'Viscera weight', 
               'Shell weight', 
               'Rings']
data = pd.read_csv(FILE_NAME, 
                   header=None, 
                   names=feature_names)

# SageMaker XGBoost has the convention of label in the first column
data = data[feature_names[-1:] + feature_names[:-1]]
data["Sex"] = data["Sex"].astype("category").cat.codes

# Split the downloaded data into train/test dataframes
train, test = np.split(data.sample(frac=1), [int(.8*len(data))])

# requires PyArrow installed
train.to_parquet('abalone_train.parquet')
test.to_parquet('abalone_test.parquet')

%%time
sagemaker.Session().upload_data('abalone_train.parquet', 
                                bucket=bucket, 
                                key_prefix=prefix+'/'+'train')

sagemaker.Session().upload_data('abalone_test.parquet', 
                                bucket=bucket, 
                                key_prefix=prefix+'/'+'test')

from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(region, 'xgboost', '0.90-1')

%%time
import time
from time import gmtime, strftime

job_name = 'xgboost-parquet-example-training-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("Training job", job_name)

#Ensure that the training and validation data folders generated above are reflected in the "InputDataConfig" parameter below.

create_training_params = {
    "AlgorithmSpecification": {
        "TrainingImage": container,
        "TrainingInputMode": "Pipe"
    },
    "RoleArn": role,
    "OutputDataConfig": {
        "S3OutputPath": bucket_path + "/" + prefix + "/single-xgboost"
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.m5.24xlarge",
        "VolumeSizeInGB": 20
    },
    "TrainingJobName": job_name,
    "HyperParameters": {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "silent":"0",
        "objective":"reg:linear",
        "num_round":"10"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 3600
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": bucket_path + "/" + prefix + "/train",
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "application/x-parquet",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": bucket_path + "/" + prefix + "/test",
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "application/x-parquet",
            "CompressionType": "None"
        }
    ]
}


client = boto3.client('sagemaker', region_name=region)
client.create_training_job(**create_training_params)

status = client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print(status)
while status !='Completed' and status!='Failed':
    time.sleep(60)
    status = client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
    print(status)

%matplotlib inline
from sagemaker.analytics import TrainingJobAnalytics

metric_name = 'validation:rmse'

metrics_dataframe = TrainingJobAnalytics(training_job_name=job_name, metric_names=[metric_name]).dataframe()
plt = metrics_dataframe.plot(kind='line', figsize=(12,5), x='timestamp', y='value', style='b.', legend=False)
plt.set_ylabel(metric_name);