from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
from time import sleep

def create_consumer(retries=5, retry_delay=5):
    """Create a Kafka consumer with retry logic"""
    for i in range(retries):
        try:
            return KafkaConsumer(
                'sentiment-analysis-results',
                bootstrap_servers='kafka:9092',  
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id='text-consumer-group',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
        except KafkaError as e:
            if i == retries - 1:
                raise e
            print(f"Failed to connect to Kafka, retrying in {retry_delay} seconds...")
            sleep(retry_delay)

def map_sentiment_label(score):
    """Map sentiment score to sentiment label"""
    if score == 0:
        return 'negatif'
    elif score == 1:
        return 'positif'
    else:
        return 'inconnu'

def main():
    print("Consuming messages from Kafka...")
    consumer = create_consumer()
    try:
        for message in consumer:
            value = message.value
            sentiment_score = value.get('sentiment_score')
            sentiment_label = map_sentiment_label(sentiment_score)
            value['sentiment_score'] = sentiment_label
            print(f"Received message from partition {message.partition}:")
            print(f"Offset: {message.offset}")
            print(f"Key: {message.key}")
            print(f"Value: {value}")
            print("-" * 50)
    except KeyboardInterrupt:
        print("Consumer interrupted by user")
    except Exception as e:
        print(f"Error consuming message: {e}")
    finally:
        # Close consumer
        consumer.close()
        print("Consumer closed.")

if __name__ == "__main__":
    main()