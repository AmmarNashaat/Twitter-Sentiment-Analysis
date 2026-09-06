import pandas as pd
import json
from kafka import KafkaProducer
import time
from pathlib import Path

# 1. Adjust this to your actual CSV path on Linux
CSV_PATH = Path("/home/badawy/projects/HSBD/dataset.csv")  # <-- change 'your_user' and filename

# 2. Topic name must match what you created with kafka-topics.sh
TOPIC = "tweets-topic"

# 3. Use a small sample fraction if you want; set to 1.0 to send everything
SAMPLE_PERCENTAGE = 0.1  # 10%



producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

print("Producer started...")

# Read CSV in chunks to avoid loading everything in RAM
for chunk in pd.read_csv(CSV_PATH, chunksize=1000):
    # Optional: sample a fraction from each chunk
    sample_chunk = chunk.sample(frac=SAMPLE_PERCENTAGE)

    for _, row in sample_chunk.iterrows():
        # Adjust keys/column names to your Kaggle file
        message = {
            "text": str(row["Text"]),     # column with tweet text
            "label": str(row["Label"]),   # column with sentiment label
        }
        producer.send(TOPIC, message)

    # Small delay so the stream looks more "real time"
    time.sleep(0.05)

producer.flush()
producer.close()

print("Producer finished sending sampled data")
