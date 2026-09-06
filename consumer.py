import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType
from pyspark.ml.pipeline import PipelineModel

# Use venv Python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

BASE_PATH = "/home/badawy/projects/HSBD/output"
os.makedirs(BASE_PATH, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("TweetsStreamingWithModel")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2"
    )
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Kafka stream
df = (
    spark.readStream
         .format("kafka")
         .option("kafka.bootstrap.servers", "localhost:9092")
         .option("subscribe", "tweets-topic")
         .option("startingOffsets", "latest")
         .load()
)

# Schema must match producer JSON: {"text": ..., "label": ...}
schema = (
    StructType()
    .add("text", StringType())
    .add("label", StringType())
)

parsed = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.text", "data.label")

# Load trained pipeline model
model_path = "/home/badawy/projects/HSBD/models/tweet_model"
model = PipelineModel.load(model_path)

# Apply model: adds labelIndex, prediction, etc.
predictions = model.transform(parsed)

# Console output for demo
console_query = (
    predictions.select(
        col("text").substr(1, 60).alias("Text"),
        col("label").alias("Label"),
        col("prediction").alias("Prediction")
    )
    .writeStream
    .format("console")
    .option("truncate", False)
    .outputMode("append")
    .start()
)

# Save for offline metrics
file_query = (
    predictions.select("text", "label", "prediction")
    .writeStream
    .format("parquet")
    .option("path", f"{BASE_PATH}/predictions")
    .option("checkpointLocation", f"{BASE_PATH}/checkpoint")
    .outputMode("append")
    .start()
)

spark.streams.awaitAnyTermination()
