import torch
from flask import Flask, request, jsonify
import torch.nn as nn
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType
from fuzzy_logic import fuzzy_sentiment_analysis  # Importer la logique floue

app = Flask(__name__)

# Définir le modèle GRUNet
class GRUNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, max_len):
        super(GRUNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        output = self.fc(hidden[-1])
        return output

# Hyperparamètres
vocab_size = 5000
embed_dim = 128
hidden_dim = 128
output_dim = 3
max_len = 10

# Initialiser le modèle
model = GRUNet(vocab_size, embed_dim, hidden_dim, output_dim, max_len)
model.load_state_dict(torch.load('gru_model.pth'))
model.eval()  # Le mettre en mode évaluation

# Charger le tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Enlever les URLs
    text = re.sub(r'@\w+', '', text)  # Enlever les mentions d'utilisateurs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Enlever les caractères spéciaux et chiffres
    text = re.sub(r'\s+', ' ', text).strip()  # Enlever les espaces multiples
    return text

def predict(model, tokenizer, texts, max_len):
    texts_cleaned = [clean_text(text) for text in texts]
    texts_seq = tokenizer.texts_to_sequences(texts_cleaned)
    texts_padded = pad_sequences(texts_seq, maxlen=max_len)
    texts_tensor = torch.tensor(texts_padded, dtype=torch.long)

    with torch.no_grad():
        outputs = model(texts_tensor)
        _, predicted = torch.max(outputs, 1)

    return predicted

# Initialiser Spark
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# UDF pour nettoyer le texte
clean_text_udf = udf(clean_text, StringType()) 

# UDF pour prédire le sentiment
def predict_udf(text):
    sentiment_score = predict(model, tokenizer, [text], max_len).item()
    return sentiment_score

predict_udf = udf(predict_udf, IntegerType())

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Créer un DataFrame Spark
    df = spark.createDataFrame([(text,)], ["text"])
    
    # Nettoyer le texte et prédire le sentiment
    df = df.withColumn("cleaned_text", clean_text_udf(df.text))
    df = df.withColumn("sentiment_score", predict_udf(df.cleaned_text))
    
    # Récupérer le résultat
    result = df.select("sentiment_score").collect()[0]["sentiment_score"]
    
    # Utiliser la logique floue pour obtenir le label de sentiment
    fuzzy_label = fuzzy_sentiment_analysis(result)
    
    return jsonify({'sentiment_score': result, 'fuzzy_label': fuzzy_label})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)