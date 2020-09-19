%%time
import sagemaker
from sagemaker import get_execution_role
import boto3

role = get_execution_role()
print(role)

sess = sagemaker.Session()
bucket=sess.default_bucket()
prefix = 'ic-fulltraining'

from sagemaker.amazon.amazon_estimator import get_image_uri

training_image = get_image_uri(sess.boto_region_name, 'image-classification', repo_version="1")
print (training_image)

smclient = boto3.Session().client('sagemaker')

import os 
import urllib.request


def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

# caltech-256
download('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec')
download('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec')

# Four channels: train, validation, train_lst, and validation_lst
s3train = 's3://{}/{}/train/'.format(bucket, prefix)
s3validation = 's3://{}/{}/validation/'.format(bucket, prefix)

# upload the lst files to train and validation channels
!aws s3 cp caltech-256-60-train.rec $s3train --quiet
!aws s3 cp caltech-256-60-val.rec $s3validation --quiet

#Set the data type and channels used for training
s3_output_location = 's3://{}/{}/output'.format(bucket, prefix)
s3_input_train = sagemaker.session.s3_input(s3train, distribution='FullyReplicated', 
                        content_type='application/x-recordio', s3_data_type='S3Prefix')
s3_input_validation = sagemaker.session.s3_input(s3validation, distribution='FullyReplicated', 
                             content_type='application/x-recordio', s3_data_type='S3Prefix')

#Hyperparameter tuninng job
sess = sagemaker.Session()

imageclassification = sagemaker.estimator.Estimator(training_image,
                                    role, 
                                    train_instance_count=1, 
                                    train_instance_type='ml.p3.2xlarge',
                                    output_path=s3_output_location,
                                    sagemaker_session=sess)

imageclassification.set_hyperparameters(num_layers=18,
                                        image_shape='3,224,224',
                                        num_classes=257,
                                        num_training_samples=15420,
                                        mini_batch_size=128,
                                        epochs=10,
                                        optimizer='sgd',
                                        top_k='2',
                                        precision_dtype='float32',
                                        augmentation_type='crop')


from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

hyperparameter_ranges = {'learning_rate': ContinuousParameter(0.0001, 0.05),
                         'momentum': ContinuousParameter(0.0, 0.99),
                         'weight_decay': ContinuousParameter(0.0, 0.99)}

objective_metric_name = 'validation:accuracy'

tuner = HyperparameterTuner(imageclassification,
                            objective_metric_name,
                            hyperparameter_ranges,
                            objective_type='Maximize',
                            max_jobs=5,
                            max_parallel_jobs=2)


#Hyperparameter tuning job
tuner.fit({'train': s3_input_train, 'validation': s3_input_validation},include_cls_metadata=False)

tuning_job_name = tuner._current_job_name

tuner_parent_metrics = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)
if not tuner_parent_metrics.dataframe().empty:
    df_parent = tuner_parent_metrics.dataframe().sort_values(['FinalObjectiveValue'], ascending=False)
    
df_parent

import bokeh
import bokeh.io
bokeh.io.output_notebook()
from bokeh.plotting import figure, show
from bokeh.models import HoverTool

import pandas as pd

df_parent_objective_value = df_parent[df_parent['FinalObjectiveValue'] > -float('inf')]

p = figure(plot_width=900, plot_height=400, x_axis_type='datetime',x_axis_label='datetime', y_axis_label=objective_metric_name)
p.circle(source=df_parent_objective_value, x='TrainingStartTime', y='FinalObjectiveValue', color='black')

show(p)

#Tuning using warm start

from sagemaker.tuner import WarmStartConfig, WarmStartTypes

parent_tuning_job_name = tuning_job_name
warm_start_config = WarmStartConfig(WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM, parents={parent_tuning_job_name})

parent_tuning_job_name

tuner_warm_start = HyperparameterTuner(imageclassification,
                            objective_metric_name,
                            hyperparameter_ranges,
                            objective_type='Maximize',
                            max_jobs=5,
                            max_parallel_jobs=2,
                            base_tuning_job_name='warmstart',
                            warm_start_config=warm_start_config)

#Launching tuning
tuner_warm_start.fit({'train': s3_input_train, 'validation': s3_input_validation},include_cls_metadata=False)
warmstart_tuning_job_name = tuner_warm_start._current_job_name
tuner_warm_start_metrics = sagemaker.HyperparameterTuningJobAnalytics(warmstart_tuning_job_name)
if not tuner_warm_start_metrics.dataframe().empty:
    df_warm_start = tuner_warm_start_metrics.dataframe().sort_values(['FinalObjectiveValue'], ascending=False)
df_warm_start

import bokeh
import bokeh.io
bokeh.io.output_notebook()
from bokeh.plotting import figure, show
from bokeh.models import HoverTool

import pandas as pd

df_parent_objective_value = df_parent[df_parent['FinalObjectiveValue'] > -float('inf')]
df_warm_start_objective_value = df_warm_start[df_warm_start['FinalObjectiveValue'] > -float('inf')]

p = figure(plot_width=900, plot_height=400, x_axis_type='datetime',x_axis_label='datetime', y_axis_label=objective_metric_name)
p.circle(source=df_parent_objective_value, x='TrainingStartTime', y='FinalObjectiveValue', color='black')
p.circle(source=df_warm_start_objective_value, x='TrainingStartTime', y='FinalObjectiveValue',color='red')
show(p)


#Get the best model
best_overall_training_job = smclient.describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=warmstart_tuning_job_name)['OverallBestTrainingJob']

best_overall_training_job

#Transfer learning
from sagemaker.tuner import WarmStartConfig, WarmStartTypes
parent_tuning_job_name_2 = warmstart_tuning_job_name
transfer_learning_config = WarmStartConfig(WarmStartTypes.TRANSFER_LEARNING, 
                                    parents={parent_tuning_job_name,parent_tuning_job_name_2})


imageclassification.set_hyperparameters(num_layers=18,
                                        image_shape='3,224,224',
                                        num_classes=257,
                                        num_training_samples=15420,
                                        mini_batch_size=128,
                                        epochs=10,
                                        optimizer='sgd',
                                        top_k='2',
                                        precision_dtype='float32',
                                        augmentation_type='crop_color_transform')

tuner_transfer_learning = HyperparameterTuner(imageclassification,
                            objective_metric_name,
                            hyperparameter_ranges,
                            objective_type='Maximize',
                            max_jobs=5,
                            max_parallel_jobs=2,
                            base_tuning_job_name='transferlearning',
                            warm_start_config=transfer_learning_config)


tuner_transfer_learning.fit({'train': s3_input_train, 'validation': s3_input_validation},include_cls_metadata=False)

transferlearning_tuning_job_name = tuner_transfer_learning._current_job_name

tuner_transferlearning_metrics = sagemaker.HyperparameterTuningJobAnalytics(transferlearning_tuning_job_name)
if not tuner_transferlearning_metrics.dataframe().empty:
    df_transfer_learning = tuner_transferlearning_metrics.dataframe().sort_values(['FinalObjectiveValue'], ascending=False)

df_transfer_learning

import bokeh
import bokeh.io
bokeh.io.output_notebook()
from bokeh.plotting import figure, show
from bokeh.models import HoverTool

import pandas as pd

df_parent_objective_value = df_parent[df_parent['FinalObjectiveValue'] > -float('inf')]
df_warm_start_objective_value = df_warm_start[df_warm_start['FinalObjectiveValue'] > -float('inf')]
df_transfer_learning_objective_value = df_transfer_learning[df_transfer_learning['FinalObjectiveValue'] > -float('inf')]

p = figure(plot_width=900, plot_height=400, x_axis_type='datetime', x_axis_label='datetime', y_axis_label=objective_metric_name)
p.circle(source=df_parent_objective_value, x='TrainingStartTime', y='FinalObjectiveValue', color='black')
p.circle(source=df_warm_start_objective_value, x='TrainingStartTime', y='FinalObjectiveValue',color='red')
p.circle(source=df_transfer_learning_objective_value, x='TrainingStartTime', y='FinalObjectiveValue',color='blue')
show(p)

