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

#Create a custom SageMakerEstimator
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker_pyspark import SageMakerEstimator
from sagemaker_pyspark.transformation.deserializers import KMeansProtobufResponseRowDeserializer
from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer
from sagemaker_pyspark import IAMRole
from sagemaker_pyspark import RandomNamePolicyFactory
from sagemaker_pyspark import EndpointCreationPolicy

# Create an Estimator from scratch
estimator = SageMakerEstimator(
    trainingImage = get_image_uri(region, 'kmeans'), # Training image 
    modelImage = get_image_uri(region, 'kmeans'), # Model image
    requestRowSerializer = ProtobufRequestRowSerializer(),
    responseRowDeserializer = KMeansProtobufResponseRowDeserializer(),
    hyperParameters = {"k": "10", "feature_dim": "784"}, # Set parameters for K-Means
    sagemakerRole = IAMRole(role),
    trainingInstanceType = "ml.m4.xlarge",
    trainingInstanceCount = 1,
    endpointInstanceType = "ml.t2.medium",
    endpointInitialInstanceCount = 1,
    trainingSparkDataFormat = "sagemaker",
    namePolicyFactory = RandomNamePolicyFactory("sparksm-4-"),
    endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM
    )

customModel = estimator.fit(trainingData)

#Inference
transformedData = customModel.transform(testData)
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

# Delete the resources
from sagemaker_pyspark import SageMakerResourceCleanup

def cleanUp(model):
    resource_cleanup = SageMakerResourceCleanup(model.sagemakerClient)
    resource_cleanup.deleteResources(model.getCreatedResources())

cleanUp(customModel)