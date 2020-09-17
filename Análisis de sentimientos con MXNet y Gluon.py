from sagemaker import s3, session

bucket = session.Session().default_bucket()
inputs = s3.S3Uploader.upload('data', 's3://{}/mxnet-gluon-sentiment-example/data'.format(bucket))


pygmentize 'sentiment.py'


#Se corre un training job de sagemaker

from sagemaker import get_execution_role
from sagemaker.mxnet import MXNet

m = MXNet('sentiment.py',
          role=get_execution_role(),
          train_instance_count=1,
          train_instance_type='ml.c4.xlarge',
          framework_version='1.6.0',
          py_version='py3',
          distributions={'parameter_server': {'enabled': True}},
          hyperparameters={'batch-size': 8,
                           'epochs': 2,
                           'learning-rate': 0.01,
                           'embedding-size': 50, 
                           'log-interval': 1000})


#Fit y generación de resultados
m.fit(inputs)
predictor = m.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')
data = ["this movie was extremely good .",
        "the plot was very boring .",
        "this film is so slick , superficial and trend-hoppy .",
        "i just could not watch it till the end .",
        "the movie was so enthralling !"]

response = predictor.predict(data)
print(response)