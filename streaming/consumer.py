from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
from time import sleep
from collections import defaultdict
from pprint import pprint

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
    
    total_messages = 0
    sentiment_counts = defaultdict(lambda: defaultdict(int))
    country_counts = defaultdict(lambda: defaultdict(int))
    platform_counts = defaultdict(lambda: defaultdict(int))
    sex_counts = defaultdict(lambda: defaultdict(int))
    
    try:
        for message in consumer:
            value = message.value
            sentiment_score = value.get('sentiment_score')
            sentiment_label = map_sentiment_label(sentiment_score)
            value['sentiment_score'] = sentiment_label
            
            country = value.get('country', 'Unknown')
            platform = value.get('platform', 'Unknown')
            sex = value.get('sex', 'Unknown')
            
            # Update statistics
            total_messages += 1
            sentiment_counts[sentiment_label]['total'] += 1
            country_counts[country][sentiment_label] += 1
            platform_counts[platform][sentiment_label] += 1
            sex_counts[sex][sentiment_label] += 1
            
            print(f"Received message from partition {message.partition}:")
            print(f"Offset: {message.offset}")
            print(f"Key: {message.key}")
            print("Value:")
            pprint(value)
            print("-" * 50)
            
            # Print statistics
            print(f"Total messages: {total_messages}")
            print("Sentiment counts:")
            pprint(dict(sentiment_counts))
            print("Country counts:")
            pprint(dict(country_counts))
            print("Platform counts:")
            pprint(dict(platform_counts))
            print("Sex counts:")
            pprint(dict(sex_counts))
            print("=" * 50)
            
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