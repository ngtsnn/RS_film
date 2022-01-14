# Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, explode

# create an instance
spark = SparkSession.builder.getOrCreate()


# pre-processing
df_movies = spark.read.csv("data/movies_small.csv", header=True, inferSchema=True)
df_ratings = spark.read.csv("data/ratings_small.csv", header=True, inferSchema=True).select(["userId", "movieId", "rating"])
(training, test) = df_ratings.randomSplit([0.8, 0.2])


# fit model ALS
als = ALS(maxIter=10, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", implicitPrefs=False)
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)



# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)
predictions.show()
print("RMSE = " + str(rmse))
model.save("ALS")

userRecs = userRecs.withColumn("rec", explode("recommendations")).select('userId', col("rec.movieId"), col("rec.rating"))
userRecs.join(df_movies, on="movieId").select(["userId", "movieId", "rating", "title", "genres"]).show()