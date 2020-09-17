import os
import boto3
import re
import sagemaker

# Get a SageMaker-compatible role used by this Notebook Instance.
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

### update below values appropriately ###
bucket = sagemaker.Session().default_bucket()
prefix = 'sagemaker/DEMO-xgboost-dist-script'
#### 

print(region)


#Fetching el dataset

import io
import boto3
import random

def data_split(FILE_DATA, DATA_DIR, FILE_TRAIN_BASE, FILE_TRAIN_1, FILE_VALIDATION, FILE_TEST, 
               PERCENT_TRAIN_0, PERCENT_TRAIN_1, PERCENT_VALIDATION, PERCENT_TEST):
    data = [l for l in open(FILE_DATA, 'r')]
    train_file_0 = open(DATA_DIR + "/" + FILE_TRAIN_0, 'w')
    train_file_1 = open(DATA_DIR + "/" + FILE_TRAIN_1, 'w')
    valid_file = open(DATA_DIR + "/" + FILE_VALIDATION, 'w')
    tests_file = open(DATA_DIR + "/" + FILE_TEST, 'w')

    num_of_data = len(data)
    num_train_0 = int((PERCENT_TRAIN_0/100.0)*num_of_data)
    num_train_1 = int((PERCENT_TRAIN_1/100.0)*num_of_data)
    num_valid = int((PERCENT_VALIDATION/100.0)*num_of_data)
    num_tests = int((PERCENT_TEST/100.0)*num_of_data)

    data_fractions = [num_train_0, num_train_1, num_valid, num_tests]
    split_data = [[],[],[],[]]

    rand_data_ind = 0

    for split_ind, fraction in enumerate(data_fractions):
        for i in range(fraction):
            rand_data_ind = random.randint(0, len(data)-1)
            split_data[split_ind].append(data[rand_data_ind])
            data.pop(rand_data_ind)

    for l in split_data[0]:
        train_file_0.write(l)

    for l in split_data[1]:
        train_file_1.write(l)
        
    for l in split_data[2]:
        valid_file.write(l)

    for l in split_data[3]:
        tests_file.write(l)

    train_file_0.close()
    train_file_1.close()
    valid_file.close()
    tests_file.close()

def write_to_s3(fobj, bucket, key):
    return boto3.Session(region_name=region).resource('s3').Bucket(bucket).Object(key).upload_fileobj(fobj)

def upload_to_s3(bucket, channel, filename):
    fobj=open(filename, 'rb')
    key = prefix+'/'+channel
    url = 's3://{}/{}/{}'.format(bucket, key, filename)
    print('Writing to {}'.format(url))
    write_to_s3(fobj, bucket, key)

#Ingesta de la data
import urllib.request

# Load the dataset
FILE_DATA = 'abalone'
urllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone", FILE_DATA)

#split the downloaded data into train/test/validation files
FILE_TRAIN_0 = 'abalone.train_0'
FILE_TRAIN_1 = 'abalone.train_1'
FILE_VALIDATION = 'abalone.validation'
FILE_TEST = 'abalone.test'
PERCENT_TRAIN_0 = 35
PERCENT_TRAIN_1 = 35
PERCENT_VALIDATION = 15
PERCENT_TEST = 15

DATA_DIR = 'data'

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

data_split(FILE_DATA, DATA_DIR, FILE_TRAIN_0, FILE_TRAIN_1, FILE_VALIDATION, FILE_TEST, 
           PERCENT_TRAIN_0, PERCENT_TRAIN_1, PERCENT_VALIDATION, PERCENT_TEST)

#upload the files to the S3 bucket
upload_to_s3(bucket, 'train/train_0.libsvm', DATA_DIR + "/" + FILE_TRAIN_0)
upload_to_s3(bucket, 'train/train_1.libsvm', DATA_DIR + "/" + FILE_TRAIN_1)
upload_to_s3(bucket, 'validation/validation.libsvm', DATA_DIR + "/" + FILE_VALIDATION)
upload_to_s3(bucket, 'test/test.libsvm', DATA_DIR + "/" + FILE_TEST)


import argparse
import json
import logging
import os
import pandas as pd
import pickle as pkl

from sagemaker_containers import entry_point
from sagemaker_xgboost_container.data_utils import get_dmatrix
from sagemaker_xgboost_container import distributed

import xgboost as xgb


def _xgb_train(params, dtrain, evals, num_boost_round, model_dir, is_master):
    """Run xgb train on arguments given with rabit initialized.

    This is our rabit execution function.

    :param args_dict: Argument dictionary used to run xgb.train().
    :param is_master: True if current node is master host in distributed training, or is running single node training job. Note that rabit_run will include this argument.
    """
    booster = xgb.train(params=params, dtrain=dtrain, evals=evals, num_boost_round=num_boost_round)

    if is_master:
        model_location = model_dir + '/xgboost-model'
        pkl.dump(booster, open(model_location, 'wb'))
        logging.info("Stored trained model at {}".format(model_location))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--max_depth', type=int,)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--min_child_weight', type=int)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--verbose', type=int)
    parser.add_argument('--objective', type=str)
    parser.add_argument('--num_round', type=int)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--sm_hosts', type=str, default=os.environ['SM_HOSTS'])
    parser.add_argument('--sm_current_host', type=str, default=os.environ['SM_CURRENT_HOST'])

    args, _ = parser.parse_known_args()

    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(os.environ['SM_HOSTS'])
    sm_current_host = args.sm_current_host

    dtrain = get_dmatrix(args.train, 'libsvm')
    dval = get_dmatrix(args.validation, 'libsvm')
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'verbose': args.verbose,
        'objective': args.objective}

    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        model_dir=args.model_dir)

    if len(sm_hosts) > 1:
        # Wait until all hosts are able to find each other
        entry_point._wait_hostname_resolution()

        # Execute training function after initializing rabit.
        distributed.rabit_run(
            exec_fun=_xgb_train,
            args=xgb_train_args,
            include_in_training=(dtrain is not None),
            hosts=sm_hosts,
            current_host=sm_current_host,
            update_rabit_args=True
        )
    else:
        # If single node training, call training method directly.
        if dtrain:
            xgb_train_args['is_master'] = True
            _xgb_train(**xgb_train_args)
        else:
            raise ValueError("Training channel must have data to train model.")


def model_fn(model_dir):
    """Deserialized and return fitted model.

    Note that this should have the same name as the serialized model in the _xgb_train method
    """
    model_file = 'xgboost-model'
    booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    return booster


#Training the model
hyperparams = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "verbose":"1",
        "objective":"reg:linear",
        "num_round":"50"}

instance_type = "ml.m5.2xlarge"
output_path = 's3://{}/{}/{}/output'.format(bucket, prefix, 'abalone-dist-xgb')
content_type = "libsvm"

# Open Source distributed script mode
from sagemaker.session import s3_input, Session
from sagemaker.xgboost.estimator import XGBoost

boto_session = boto3.Session(region_name=region)
session = Session(boto_session=boto_session)
script_path = 'abalone.py'

xgb_script_mode_estimator = XGBoost(
    entry_point=script_path,
    framework_version='0.90-1', # Note: framework_version is mandatory
    hyperparameters=hyperparams,
    role=role,
    train_instance_count=2, 
    train_instance_type=instance_type,
    output_path=output_path)

train_input = s3_input("s3://{}/{}/{}/".format(bucket, prefix, 'train'), content_type=content_type)
validation_input = s3_input("s3://{}/{}/{}/".format(bucket, prefix, 'validation'), content_type=content_type)

#Training the XGBoost estimator
xgb_script_mode_estimator.fit({'train': train_input, 'validation': validation_input})


#Deploying the model
predictor = xgb_script_mode_estimator.deploy(initial_instance_count=1, 
                                             instance_type="ml.m5.2xlarge")
predictor.serializer = str

test_file = DATA_DIR + "/" + FILE_TEST
with open(test_file, 'r') as f:
    payload = f.read()

runtime_client = boto3.client('runtime.sagemaker', region_name=region)
response = runtime_client.invoke_endpoint(EndpointName=predictor.endpoint, 
                                          ContentType='text/x-libsvm', 
                                          Body=payload)
result = response['Body'].read().decode('ascii')
print('Predicted values are {}.'.format(result))