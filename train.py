from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = (
    SparkSession.builder
    .appName("TweetsBatchTrain")
    .config("spark.driver.memory", "8g")  # 8g is enough on 32 GB RAM
    .getOrCreate()
)

# 1. Load data
df = (
    spark.read
         .option("header", "true")
         .option("inferSchema", "true")
         .csv("/home/badawy/projects/HSBD/dataset.csv")
)

data = df.select(
    col("Text").alias("text"),
    col("Label").cast("string").alias("label")
).na.drop(subset=["text", "label"])

# 2. Sample to ~200k rows (20% of 1M)
data_small = data.sample(withReplacement=False, fraction=0.2, seed=42)

# 3. Lighter feature space: 2000 features, not 5000
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(
    inputCol="words",
    outputCol="rawFeatures",
    numFeatures=2000
)
idf = IDF(inputCol="rawFeatures", outputCol="features")
label_indexer = StringIndexer(inputCol="label", outputCol="labelIndex",     handleInvalid="keep"  # allow unseen labels in test
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="labelIndex",
    maxIter=20,
    regParam=0.1
)

pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, label_indexer, lr])

train, test = data_small.randomSplit([0.8, 0.2], seed=42)

# 4. Fit model on sampled data
model = pipeline.fit(train)

# 5. Evaluate
preds = model.transform(test)

for metric in ["accuracy", "weightedPrecision", "weightedRecall"]:
    evaluator = MulticlassClassificationEvaluator(
        labelCol="labelIndex",
        predictionCol="prediction",
        metricName=metric,
    )
    print(metric, "=", evaluator.evaluate(preds))

# 6. Save model
model.save("/home/badawy/projects/HSBD/models/tweet_model")
