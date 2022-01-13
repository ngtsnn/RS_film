# Import libraries
from pyspark.sql import SparkSession


# create an instance
spark = SparkSession.builder.appName("Films RS").getOrCreate()


# pre processing
df_movies = spark.read.option("header", 'true').csv("movies_small.csv", inferSchema=True)
df_ratings = spark.read.option("header", 'true').csv("ratings_small.csv", inferSchema=True)
df_movies.show(5)
df_movies.printSchema()
df_ratings.show(5)
df_ratings.printSchema()