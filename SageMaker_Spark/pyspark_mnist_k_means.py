#Setup
import os
import boto3

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

import sagemaker
from sagemaker import get_execution_role
import sagemaker_pyspark

role = get_execution_role()

# Configure Spark to use the SageMaker Spark dependency jars
jars = sagemaker_pyspark.classpath_jars()

classpath = ":".join(sagemaker_pyspark.classpath_jars())

# See the SageMaker Spark Github to learn how to connect to EMR from a notebook instance
spark = SparkSession.builder.config("spark.driver.extraClassPath", classpath)\
    .master("local[*]").getOrCreate()
    
spark

#Loading the data
import boto3

cn_regions = ['cn-north-1', 'cn-northwest-1']
region = boto3.Session().region_name
endpoint_domain = 'com.cn' if region in cn_regions else 'com'
spark._jsc.hadoopConfiguration().set('fs.s3a.endpoint', 's3.{}.amazonaws.{}'.format(region, endpoint_domain))

trainingData = spark.read.format('libsvm')\
    .option('numFeatures', '784')\
    .load('s3a://sagemaker-sample-data-{}/spark/mnist/train/'.format(region))

testData = spark.read.format('libsvm')\
    .option('numFeatures', '784')\
    .load('s3a://sagemaker-sample-data-{}/spark/mnist/test/'.format(region))

trainingData.show()

#Training with K-Means and hosting Model
from sagemaker_pyspark import IAMRole
from sagemaker_pyspark.algorithms import KMeansSageMakerEstimator
from sagemaker_pyspark import RandomNamePolicyFactory

# Create K-Means Estimator
kmeans_estimator = KMeansSageMakerEstimator(
    sagemakerRole = IAMRole(role),
    trainingInstanceType = 'ml.m4.xlarge', # Instance type to train K-means on SageMaker
    trainingInstanceCount = 1,
    endpointInstanceType = 'ml.t2.large', # Instance type to serve model (endpoint) for inference
    endpointInitialInstanceCount = 1,
    namePolicyFactory = RandomNamePolicyFactory("sparksm-1a-")) # All the resources created are prefixed with sparksm-1

# Set parameters for K-Means
kmeans_estimator.setFeatureDim(784)
kmeans_estimator.setK(10)

# Train
initialModel = kmeans_estimator.fit(trainingData)

initialModelEndpointName = initialModel.endpointName
print(initialModelEndpointName)

#Inference
# Run inference on the test data and show some results
transformedData = initialModel.transform(testData)

transformedData.show()

from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt
import numpy as np
import string

# Helper function to display a digit
def showDigit(img, caption='', xlabel='', subplot=None):
    if subplot==None:
        _,(subplot)=plt.subplots(1,1)
    imgr=img.reshape((28,28))
    subplot.axes.get_xaxis().set_ticks([])
    subplot.axes.get_yaxis().set_ticks([])
    plt.title(caption)
    plt.xlabel(xlabel)
    subplot.imshow(imgr, cmap='gray')
    
def displayClusters(data):
    images = np.array(data.select("features").cache().take(250))
    clusters = data.select("closest_cluster").cache().take(250)

    for cluster in range(10):
        print('\n\n\nCluster {}:'.format(string.ascii_uppercase[cluster]))
        digits = [ img for l, img in zip(clusters, images) if int(l.closest_cluster) == cluster ]
        height=((len(digits)-1)//5)+1
        width=5
        plt.rcParams["figure.figsize"] = (width,height)
        _, subplots = plt.subplots(height, width)
        subplots=np.ndarray.flatten(subplots)
        for subplot, image in zip(subplots, digits):
            showDigit(image, subplot=subplot)
        for subplot in subplots[len(digits):]:
            subplot.axis('off')

        plt.show()
        
displayClusters(transformedData)

#Re-using existing endpoints or models to create SageMakerModel
ENDPOINT_NAME = initialModelEndpointName
print(ENDPOINT_NAME)
from sagemaker_pyspark import SageMakerModel
from sagemaker_pyspark import EndpointCreationPolicy
from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer
from sagemaker_pyspark.transformation.deserializers import KMeansProtobufResponseRowDeserializer

attachedModel = SageMakerModel(
    existingEndpointName = ENDPOINT_NAME,
    endpointCreationPolicy = EndpointCreationPolicy.DO_NOT_CREATE,
    endpointInstanceType = None, # Required
    endpointInitialInstanceCount = None, # Required
    requestRowSerializer = ProtobufRequestRowSerializer(featuresColumnName = "features"), # Optional: already default value
    responseRowDeserializer = KMeansProtobufResponseRowDeserializer( # Optional: already default values
      distance_to_cluster_column_name = "distance_to_cluster",
      closest_cluster_column_name = "closest_cluster")
)

transformedData2 = attachedModel.transform(testData)
transformedData2.show()

#Create model and endpoint from model data
from sagemaker_pyspark import S3DataPath

MODEL_S3_PATH = S3DataPath(initialModel.modelPath.bucket, initialModel.modelPath.objectPath)
MODEL_ROLE_ARN = initialModel.modelExecutionRoleARN
MODEL_IMAGE_PATH = initialModel.modelImage

print(MODEL_S3_PATH.bucket + MODEL_S3_PATH.objectPath)
print(MODEL_ROLE_ARN)
print(MODEL_IMAGE_PATH)

from sagemaker_pyspark import RandomNamePolicy

retrievedModel = SageMakerModel(
    modelPath = MODEL_S3_PATH,
    modelExecutionRoleARN = MODEL_ROLE_ARN,
    modelImage = MODEL_IMAGE_PATH,
    endpointInstanceType = "ml.t2.medium",
    endpointInitialInstanceCount = 1,
    requestRowSerializer = ProtobufRequestRowSerializer(), 
    responseRowDeserializer = KMeansProtobufResponseRowDeserializer(),
    namePolicy = RandomNamePolicy("sparksm-1b-"), 
    endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM
)

#Create model and endpoint from job training data

TRAINING_JOB_NAME = "<YOUR_TRAINING_JOB_NAME>"
MODEL_ROLE_ARN = initialModel.modelExecutionRoleARN
MODEL_IMAGE_PATH = initialModel.modelImage

modelFromJob = SageMakerModel.fromTrainingJob(
    trainingJobName = TRAINING_JOB_NAME,
    modelExecutionRoleARN = MODEL_ROLE_ARN,
    modelImage = MODEL_IMAGE_PATH,
    endpointInstanceType = "ml.t2.medium",
    endpointInitialInstanceCount = 1,
    requestRowSerializer = ProtobufRequestRowSerializer(), 
    responseRowDeserializer = KMeansProtobufResponseRowDeserializer(),
    namePolicy = RandomNamePolicy("sparksm-1c-"),
    endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM
)

transformedData4 = modelFromJob.transform(testData)
transformedData4.show()

#Clean-up
# Delete the resources
from sagemaker_pyspark import SageMakerResourceCleanup

def cleanUp(model):
    resource_cleanup = SageMakerResourceCleanup(model.sagemakerClient)
    resource_cleanup.deleteResources(model.getCreatedResources())

# Don't forget to include any models or pipeline models that you created in the notebook
models = [initialModel, retrievedModel, modelFromJob]

# Delete regular SageMakerModels
for m in models:
    cleanUp(m)