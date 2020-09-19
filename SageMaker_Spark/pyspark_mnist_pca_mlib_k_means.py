#MLib Setup
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

#Loading data
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

#Create a hybrid pipeline with Spark PCA and SageMaker K-Means
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA

from sagemaker_pyspark.algorithms import KMeansSageMakerEstimator
from sagemaker_pyspark import IAMRole, EndpointCreationPolicy, RandomNamePolicyFactory
from sagemaker_pyspark.transformation.serializers import ProtobufRequestRowSerializer

# ML pipeline with 2 stages: PCA and K-Means

# 1st stage: PCA 
pcaSparkEstimator = PCA(
  inputCol = "features",
  outputCol = "projectedFeatures",
  k = 50)

# 2nd stage: K-Means on SageMaker
kMeansSageMakerEstimator = KMeansSageMakerEstimator(
  sagemakerRole = IAMRole(role),
  trainingSparkDataFormatOptions = {"featuresColumnName": "projectedFeatures"}, # use the output column of PCA
  requestRowSerializer = ProtobufRequestRowSerializer(featuresColumnName = "projectedFeatures"), # use the output column of PCA
  trainingInstanceType = "ml.m4.xlarge",
  trainingInstanceCount = 1,
  endpointInstanceType = "ml.t2.medium",
  endpointInitialInstanceCount = 1,
  namePolicyFactory = RandomNamePolicyFactory("sparksm-2-"),
  endpointCreationPolicy = EndpointCreationPolicy.CREATE_ON_TRANSFORM 
)

# Set parameters for K-Means
kMeansSageMakerEstimator.setFeatureDim(50)
kMeansSageMakerEstimator.setK(10)

# Define the stages of the Pipeline in order
pipelineSparkSM = Pipeline(stages=[pcaSparkEstimator, kMeansSageMakerEstimator])

# Train
pipelineModelSparkSM = pipelineSparkSM.fit(trainingData)

#Inference
# Run predictions
transformedData = pipelineModelSparkSM.transform(testData)
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

#Clean-up
# Delete the resources
from sagemaker_pyspark import SageMakerResourceCleanup
from sagemaker_pyspark import SageMakerModel

def cleanUp(model):
    resource_cleanup = SageMakerResourceCleanup(model.sagemakerClient)
    resource_cleanup.deleteResources(model.getCreatedResources())
    
# Delete the SageMakerModel in pipeline
for m in pipelineModelSparkSM.stages:
    if isinstance(m, SageMakerModel):
        cleanUp(m)
