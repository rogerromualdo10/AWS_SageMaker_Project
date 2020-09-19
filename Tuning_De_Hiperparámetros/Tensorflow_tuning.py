#Enviroment

import sagemaker

bucket = sagemaker.Session().default_bucket() # we are using a default bucket here but you can change it to any bucket in your account
prefix = 'sagemaker/DEMO-hpo-tensorflow-high' # you can customize the prefix (subfolder) here

role = sagemaker.get_execution_role() # we are using the notebook instance role for training in this example


import boto3
from time import gmtime, strftime
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

#Download MNIST dataset
import utils
from tensorflow.contrib.learn.python.learn.datasets import mnist
import tensorflow as tf

data_sets = mnist.read_data_sets('data', dtype=tf.uint8, reshape=False, validation_size=5000)
utils.convert_to(data_sets.train, 'train', 'data')
utils.convert_to(data_sets.validation, 'validation', 'data')
utils.convert_to(data_sets.test, 'test', 'data')

#Upload the data
inputs = sagemaker.Session().upload_data(path='data', bucket=bucket, key_prefix=prefix+'/data/mnist')
print (inputs)


#Script for distributed training
!cat 'mnist.py'

#Hyperparameter job
estimator = TensorFlow(entry_point='mnist.py',
                  role=role,
                  framework_version='1.12.0',
                  training_steps=1000, 
                  evaluation_steps=100,
                  train_instance_count=1,
                  train_instance_type='ml.m4.xlarge',
                  base_job_name='DEMO-hpo-tensorflow')


hyperparameter_ranges = {'learning_rate': ContinuousParameter(0.01, 0.2)}

objective_metric_name = 'loss'
objective_type = 'Minimize'
metric_definitions = [{'Name': 'loss',
                       'Regex': 'loss = ([0-9\\.]+)'}]

tuner = HyperparameterTuner(estimator,
                            objective_metric_name,
                            hyperparameter_ranges,
                            metric_definitions,
                            max_jobs=9,
                            max_parallel_jobs=3,
                            objective_type=objective_type)

#Launch hyperparameter job
tuner.fit(inputs)
boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']
