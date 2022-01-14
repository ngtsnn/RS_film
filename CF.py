# Import libraries
from typing import List
from numpy.core.fromnumeric import shape
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
from scipy import sparse, spatial
import pandas as pd


# create an instance
spark = SparkSession.builder.appName("Films RS").getOrCreate()


# pre-processing
np.set_printoptions(suppress=True)
df_movies = spark.read.csv("data/movies_small.csv", header=True, inferSchema=True)
df_ratings = spark.read.csv("data/ratings_small.csv", header=True, inferSchema=True).select(["userId", "movieId", "rating"])
(trainingSet, testSet) = df_ratings.randomSplit([0.8, 0.2])




# cosine_similarity
def cosine_similarity(X, Y):
  res = np.zeros(shape=(X.shape[0], Y.shape[0]))
  for i in range(X.shape[0]):
    for k in range(Y.shape[0]):
      res[i, k] = 1 - spatial.distance.cosine(X[i], Y[k])
  return res

# create class 
class CF (object):
  def __init__(self, k, similarity_function) -> None:
    
    self.k = k
    self.normalizedData = None
    self.simlarity_function = similarity_function

  def fit(self, ratings):
    self.users_count = ratings.select("userId").distinct().count()
    self.items_count = ratings.select("movieId").distinct().count()
    self.users = np.array(ratings.select("userId").distinct().collect())[:, 0]
    self.items = np.array(ratings.select("movieId").distinct().collect())[:, 0]
    ratings = np.array(ratings.collect())
    self.trainY = ratings
    self.normalizedData = ratings.copy()
    self.utility = np.zeros((self.users_count,))
    for i in range(self.users_count):
      user = self.users[i]
      indices = np.where(ratings[:, 0] == user)
      rates = self.trainY[indices, 2]
      self.utility[i] = np.mean(rates) if indices.__len__() > 0 else 0
      self.normalizedData[indices, 2] = rates - self.utility[i] 
      self.normalizedData[indices, 0] = i
    for i in range(self.items_count):
      item = self.items[i]
      indices = np.where(ratings[:, 1] == item)
      self.normalizedData[indices, 1] = i
    self.normalizedData = sparse.coo_matrix((self.normalizedData[:, 2], (self.normalizedData[:, 1], self.normalizedData[:, 0])), (self.items_count, self.users_count)).toarray()
    self.S =  self.simlarity_function(self.normalizedData.T, self.normalizedData.T)

  def pred(self, u, i):
    """ predict the rating of user u for item i"""
    # find item i
    ids = np.where(self.trainY[:, 1] == i)[0].astype(np.int32)
    # all users who rated i
    users_rated_i = (self.trainY[ids, 0]).astype(np.int32)
    # convert id
    for i in range(users_rated_i.__len__()):
      users_rated_i[i] = np.where(self.users == users_rated_i[i])[0][0]
    u = np.where(self.users == u)
    i = np.where(self.items == i)
    if i[0].size == 0: 
      return self.utility[u]
    # similarity of u and users who rated i
    sim = self.S[u, users_rated_i]
    # most k similar users
    nns = np.argsort(sim)[-self.k:]
    nearest_s = sim[0, nns] # and the corresponding similarities
    # the corresponding ratings
    r = self.normalizedData[i, users_rated_i[nns]]
    eps = 1e-8 # a small number to avoid zero division
    return ((r*nearest_s).sum()/(np.abs(nearest_s).sum() + eps) + self.utility[u])[0]

  def predict(self, testSet):
    test = np.array(testSet.collect())
    prediction = np.zeros(shape=(test.shape[0], 4))
    prediction[:, :3] = test
    for testcase in prediction:
      testcase[3] = self.pred(testcase[0], testcase[1])
    columns = ["userId", "movieId", "rating", "prediction"]
    prediction = pd.DataFrame(prediction, columns=columns)
    prediction = spark.createDataFrame(data=prediction)
    return prediction
    



model = CF(20, cosine_similarity)
model.fit(trainingSet)
prediction = model.predict(testSet)
prediction.show()
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(prediction)
print("RMSE = " + str(rmse))

    