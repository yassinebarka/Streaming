import torch
from flask import Flask, request, jsonify

import torch.nn as nn
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

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
max_len = 100

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

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    sentiment_score = predict(model, tokenizer, [text], max_len).item()
    return jsonify({'sentiment_score': sentiment_score})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)