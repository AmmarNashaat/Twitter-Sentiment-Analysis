# Sentiment Analysis of Financial Tweets

End‑to‑end system for classifying tweets into sentiment categories using Apache Spark MLlib and real‑time streaming with Apache Kafka.

## Table of Contents

- Sentiment Analysis of Financial Tweets  
  - Overview  
  - Challenge Description  
  - Methodology and Architecture  
  - Experimental Results  
  - Project Structure and Tools  
  - Team  

## Overview

This project implements a complete machine learning pipeline to predict the sentiment of financial tweets in near real time. Tweets are classified into a small set of categories such as negative, neutral, positive, and litigious, using a model trained with Apache Spark MLlib and deployed through Apache Kafka and Spark Structured Streaming. The system supports batch training, offline evaluation, and real‑time scoring of new tweets arriving on a Kafka topic.

## Challenge Description

Financial tweets are short, noisy texts that often contain hashtags, user mentions, URLs, emojis, and multiple languages. The goal is to build a robust classifier that can:

- Clean and transform raw tweets into meaningful numerical features.  
- Learn to map each tweet to the correct sentiment class using labeled historical data.  
- Process new tweets continuously from a Kafka topic and assign a sentiment in (near) real time.  

Key requirements:

- Use distributed processing with Apache Spark to handle large datasets efficiently.  
- Use Spark MLlib's multiclass classification tools and evaluation metrics such as accuracy, F1, weighted precision, and weighted recall.  
- Integrate with Apache Kafka to simulate or handle real‑world streaming workloads.  

## Methodology and Architecture

The solution is built around two coordinated pipelines: a batch training pipeline and a streaming inference pipeline. Both use the same Spark MLlib model to ensure consistency between offline and online behavior.

### Batch Training Pipeline

The batch pipeline prepares data, trains the model, and computes offline metrics.

- Data source  
  - Input file: `dataset.csv` containing the columns:  
    - `Text`: raw tweet text.  
    - `Language`: language code (e.g., en, es).  
    - `Label`: sentiment class (e.g., negative, neutral, positive, litigious).  
  - Rows with missing text or label are dropped during preprocessing.

- Training subset  
  - To reduce computation, a random 20% sample of the available labeled data is used for model training and testing:  
    - `data_small = data.sample(withReplacement=False, fraction=0.2, seed=42)`  

- Feature engineering  
  - Tokenization: split `Text` into tokens with Spark's `Tokenizer`.  
  - HashingTF: convert tokens into a fixed‑size sparse vector of term frequencies.  
  - IDF: compute inverse document frequency weights and apply them to term frequencies to obtain TF‑IDF features.  

- Label encoding  
  - `StringIndexer` maps string labels (e.g., negative, neutral, positive, litigious) to a numeric `labelIndex` column suitable for classification algorithms.  
  - The mapping is stored inside the fitted pipeline and reused during streaming inference.

- Classifier  
  - Logistic Regression (multiclass) from Spark MLlib is used as the main classifier, trained on TF‑IDF features with `labelIndex` as the target.  
  - Regularization and maximum iterations are configured to balance accuracy and training time.

- Pipeline  
  - All steps (Tokenizer, HashingTF, IDF, StringIndexer, Logistic Regression) are wrapped in a single `Pipeline` and trained once.  
  - The trained pipeline model is saved under the `models` directory (for example, `models/tweet_model`) and reused by the streaming consumer.

### Streaming Inference Architecture

The online component uses Apache Kafka and Spark Structured Streaming to classify tweets in real time.

- Kafka setup  
  - Broker: Apache Kafka 2.3.  
  - Topic name: `tweets-topic`, created to carry JSON‑encoded tweet messages.  
  - Each message contains at least the tweet text, and for evaluation scenarios, a ground‑truth label.

- Producer  
  - Implemented in `producer.py`.  
  - Reads tweets from `dataset.csv` or a pre‑split streaming file, serializes them as JSON, and sends them to the `tweets-topic` Kafka topic.  
  - Uses a Kafka client compatible with Kafka 2.3 and Python.

- Spark Structured Streaming consumer  
  - Implemented in `consumer.py`.  
  - Spark version: PySpark 3.5.2 running on Java 11.  
  - Subscribes to `tweets-topic` and reads Kafka messages in micro‑batches.  
  - Parses JSON payloads to extract `text` and `label`, cleans the text if necessary, and loads the saved MLlib pipeline model from `models/`.  
  - Applies the model to each batch:  
    - Tokenization, HashingTF, IDF transformation.  
    - Logistic Regression classification producing a numeric `prediction` column.  
    - Mapping from `prediction` indices to human‑readable labels using the stored `StringIndexer` labels (e.g., 0 → negative, 1 → neutral, 2 → positive, 3 → litigious).  

- Output  
  - For debugging and demonstration, a subset of predictions is printed to the console (text excerpt, true label, predicted label).  
  - All predictions are written to disk in Parquet or CSV format under the `output` directory (for example, `output/predictions/`), including `text`, `label`, `prediction`, and optionally a decoded `prediction_label`.  
  - These files are later read by a separate batch evaluation script.

### Evaluation of Streaming Output

A dedicated script `eval.py` loads the predictions written by the consumer and evaluates them with Spark MLlib's `MulticlassClassificationEvaluator`.

- Steps:  
  - Read the Parquet/CSV prediction files from `output/predictions/`.  
  - Use `StringIndexer` to convert the string `label` column into numeric `labelIndex` for evaluation consistency.  
  - Use `MulticlassClassificationEvaluator` with `labelCol="labelIndex"` and `predictionCol="prediction"` to compute:  
    - `accuracy`  
    - `f1`  
    - `weightedPrecision`  
    - `weightedRecall`  

These metrics are saved to a file such as `model eval.txt` or an evaluation CSV for documentation.

## Experimental Results

Two sets of metrics are reported: batch test metrics after training and end‑to‑end streaming evaluation metrics computed on the consumer's output.

### Batch Model Test Metrics (After Training)

On the held‑out test split of the 20% sampled training data, the Logistic Regression model achieves:

- Accuracy: 0.7160  
- Weighted precision: 0.6866  
- Weighted recall: 0.7160  

These metrics are computed using `MulticlassClassificationEvaluator` with `accuracy`, `weightedPrecision`, and `weightedRecall` as metric names.

### Streaming Evaluation Metrics

After deploying the model in the streaming pipeline and evaluating the predictions written by the consumer, the following metrics are obtained:

- Accuracy: 0.8249  
- F1 score: 0.8248  
- Weighted precision: 0.8258  
- Weighted recall: 0.8249  

These values indicate that, on the streaming evaluation dataset, the model achieves higher overall performance compared to the initial batch test split. The improvement may result from differences between the batch test subset and the data used during streaming evaluation (for example, data distribution or sample size). The metrics are again computed with `MulticlassClassificationEvaluator` using accuracy, F1, weighted precision, and weighted recall.

## Project Structure and Tools

Project folder contents:

- `dataset.csv` – labeled financial tweets (Text, Language, Label).  
- `train.py` – batch training script: loads dataset, samples 20% of data, builds and trains the Spark MLlib pipeline, evaluates on a test split, and saves the model.  
- `consumer.py` – Spark Structured Streaming consumer: reads tweets from Kafka topic `tweets-topic`, applies the trained pipeline, and writes predictions to `output`.  
- `producer.py` – Kafka producer: streams tweets from `dataset.csv` or derived files into `tweets-topic`.  
- `eval.py` – evaluation script for consumer output: reads predictions from `output`, indexes labels, and computes accuracy, F1, weighted precision, and weighted recall.  
- `models/` – directory containing the saved Spark ML pipeline model(s) used by the consumer.  
- `output/` – directory where streaming predictions and evaluation files are stored.  
- `Mapping.txt` – file documenting the mapping between numeric prediction indices and sentiment labels (e.g., 0 → negative, 1 → neutral, 2 → positive, 3 → litigious).  
- `model eval.txt` – text file summarizing evaluation results.  

Tools and versions:

- Apache Spark / PySpark: 3.5.2  
- Apache Kafka: 2.3.x  
- Java: 11 (required for running Spark 3.x and Kafka 2.x)  

These versions are compatible and suitable for running Spark Structured Streaming with Kafka integration in a local or small‑scale environment.

## Under supervision of Prof, Giancarlo Sperli

- Ammar Gharaf - a.gharaf@studenti.unina.it
Course: Hardware & Software for Big Data (2025/2026)
