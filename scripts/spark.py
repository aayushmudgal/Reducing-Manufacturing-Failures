
# coding: utf-8

# ### Open using Databricks Platform/Py-spark. It holds the code for developing the RandomForest Classifier on the chosen subset of important features. 

# In[1]:

import os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef
import pyspark
from numpy import array
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
import gc
from pyspark.sql.functions import col, count, sum
from sklearn.metrics import matthews_corrcoef
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import rand

REPLACE_YOUR_FILE = "/FileStore/tables/e9svdv4y1482386357547/test_numeric.csv"
df0 = sqlContext.read.format("csv").load(REPLACE_YOUR_FILE, header="true", inferSchema="true")
 

df = df0.na.fill(99999)
df = df.na.drop()

df.printSchema()




# In[2]:

feature=['L3_S31_F3846','L1_S24_F1578','L3_S33_F3857','L1_S24_F1406','L3_S29_F3348','L3_S33_F3863',
            'L3_S29_F3427','L3_S37_F3950','L0_S9_F170', 'L3_S29_F3321','L1_S24_F1346','L3_S32_F3850',
            'L3_S30_F3514','L1_S24_F1366','L2_S26_F3036']

assembler = VectorAssembler(
    inputCols=feature,
    outputCol='features')
data = (assembler.transform(df).select("features", df.Response.astype('double')))

(trainingData, testData) = data.randomSplit([0.8, 0.2], seed=451)

data.printSchema()


# In[3]:

cls = RandomForestClassifier(numTrees=60, seed=1111, maxDepth=15, labelCol="Response", featuresCol="features")

pipeline = Pipeline(stages=[cls])
evaluator = MulticlassClassificationEvaluator(
    labelCol="Response", predictionCol="prediction", metricName="accuracy")
trainingData=trainingData.na.drop()
trainingData.printSchema()


# In[4]:

gc.collect()
model = pipeline.fit(trainingData)


# In[5]:

# making predictions
predicted = model.transform(testData)
response = predictions.select("Response").rdd.map(lambda r: r[0]).collect()
predictedValue = predictions.select("probability").rdd.map(lambda r: int(r[0][1])).collect()

mcc = matthews_corrcoef(response, predictedValue)
print (mcc)

