from kafka import KafkaProducer
from kafka.errors import KafkaError
import pandas as pd
import os
from time import sleep
import requests
import json

def create_producer(retries=5, retry_delay=5):
    """Create a Kafka producer with retry logic."""
    for i in range(retries):
        try:
            return KafkaProducer(
                bootstrap_servers='kafka:9092',
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        except KafkaError as e:
            if i == retries - 1:
                raise e
            print(f"Failed to connect to Kafka, retrying in {retry_delay} seconds...")
            sleep(retry_delay)

def get_sentiment_prediction(text):
    """Send text to Flask server for sentiment analysis and return the prediction."""
    response = requests.post('http://flask_server:5000/analyze', json={'text': text})
    if response.status_code == 200:
        return response.json()['sentiment_score']
    else:
        raise Exception(f"Error from Flask server: {response.text}")

def main():
    # Load CSV file
    csv_file = 'sentiment_analysis_dataset_ANG.csv'
    
    # Verify file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Read CSV and convert column names to lowercase
    df = pd.read_csv(csv_file, encoding='utf-8')
    df.columns = df.columns.str.lower()  # Convert all column names to lowercase
    
    # Create Kafka producer
    producer = create_producer()
    
    # Process each row and send to Kafka
    for index, row in df.iterrows():
        text = row['text']  # Assuming the text column is named 'text'
        sentiment_score = get_sentiment_prediction(text)
        row['sentiment_score'] = sentiment_score
        producer.send('sentiment-analysis-results', value=row.to_dict())
        print(f"Sent message: {row.to_dict()}")
    
    producer.flush()

if __name__ == "__main__":
    main()