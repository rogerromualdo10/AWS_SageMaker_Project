import boto3
from datetime import datetime, timedelta
import re

client = boto3.client('sagemaker')
running_jobs = client.list_training_jobs(CreationTimeAfter=datetime.utcnow() - timedelta(hours=1))

logdir = None
for job in running_jobs['TrainingJobSummaries']:
    tensorboard_job = False
    name = None
    tags = client.list_tags(ResourceArn=job['TrainingJobArn'])
    for tag in tags['Tags']:
        if tag['Key'] == 'TensorBoard':
            name = tag['Value']
        if tag['Key'] == 'Project' and tag['Value'] == 'cifar10':
            desc = client.describe_training_job(TrainingJobName=job['TrainingJobName'])
            job_name = desc['HyperParameters']['sagemaker_job_name'].replace('"', '')
            tensorboard_dir = re.sub(
                'source/sourcedir.tar.gz', 'model', desc['HyperParameters']['sagemaker_submit_directory']
            )
            tensorboard_job = True

    if tensorboard_job:
        if name is None:
            name = job['TrainingJobName']

        if logdir is None:
            logdir = '{}:{}'.format(name, tensorboard_dir)
        else:
            logdir = '{},{}:{}'.format(logdir, name, tensorboard_dir)

if logdir:
    print('AWS_REGION={} tensorboard --logdir {}'.format(boto3.session.Session().region_name, logdir))
else:
    print('No jobs are in progress')