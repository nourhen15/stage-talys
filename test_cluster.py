from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test-cluster-mode").getOrCreate()
df = spark.range(10)
df.show()
print("Nombre de lignes :", df.count())
spark.stop()