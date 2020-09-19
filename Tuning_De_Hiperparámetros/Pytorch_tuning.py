import sagemaker
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/DEMO-pytorch-mnist'

role = sagemaker.get_execution_role()

#Data
from torchvision import datasets, transforms

datasets.MNIST('data', download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))

inputs = sagemaker_session.upload_data(path='data', bucket=bucket, key_prefix=prefix)
print('input spec (in this case, just an S3 path): {}'.format(inputs))



!pygmentize mnist.py


#Set up the job
from sagemaker.pytorch import PyTorch

estimator = PyTorch(entry_point="mnist.py",
                    role=role,
                    framework_version='1.4.0',
                    train_instance_count=1,
                    train_instance_type='ml.m4.xlarge',
                    hyperparameters={
                        'epochs': 6,
                        'backend': 'gloo'
                    })


hyperparameter_ranges = {'lr': ContinuousParameter(0.001, 0.1),'batch-size': CategoricalParameter([32,64,128,256,512])}


objective_metric_name = 'average test loss'
objective_type = 'Minimize'
metric_definitions = [{'Name': 'average test loss',
                       'Regex': 'Test set: Average loss: ([0-9\\.]+)'}]



tuner = HyperparameterTuner(estimator,
                            objective_metric_name,
                            hyperparameter_ranges,
                            metric_definitions,
                            max_jobs=9,
                            max_parallel_jobs=3,
                            objective_type=objective_type)


#Launching the tuning job
tuner.fit({'training': inputs})

#Creating endpoint
predictor = tuner.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

#Evaluate
from IPython.display import HTML
HTML(open("input.html").read())

import numpy as np

image = np.array([data], dtype=np.float32)
response = predictor.predict(image)
prediction = response.argmax(axis=1)[0]
print(prediction)

#Cleanup
tuner.delete_endpoint()
