from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = (
    SparkSession.builder
    .appName("EvaluateConsumerOutput")
    .getOrCreate()
)

# Path where your consumer writes the results
pred_path = "/home/badawy/projects/HSBD/output/predictions/"

pred_df = (
    spark.read
         .option("mergeSchema", "true")  # in case multiple batches
         .parquet(pred_path)
)

# Index the true string labels to numeric
label_indexer = StringIndexer(
    inputCol="label",
    outputCol="labelIndex",
    handleInvalid="keep"
).fit(pred_df)

eval_df = label_indexer.transform(pred_df)

evaluator = MulticlassClassificationEvaluator(
    labelCol="labelIndex",
    predictionCol="prediction"
)

for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    val = evaluator.setMetricName(metric).evaluate(eval_df)
    print(metric, "=", val)

from pyspark.sql import Row

metrics = []
for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    val = evaluator.setMetricName(metric).evaluate(eval_df)
    print(metric, "=", val)
    metrics.append(Row(metric=metric, value=float(val)))

metrics_df = spark.createDataFrame(metrics)

# Save as Parquet
metrics_df.write.mode("overwrite").parquet(
    "/home/badawy/projects/HSBD/output/eval_metrics_parquet"
)

# Or also save as single CSV for easy viewing
(metrics_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv("/home/badawy/projects/HSBD/output/eval_metrics_csv"))
