# Utiliser l'image de base Python
FROM python:3.9-slim

# Mettre à jour pip et installer les packages requis
RUN pip install Flask torch tensorflow

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application
COPY flask_server.py /app/
COPY gru_model.pth /app/
COPY tokenizer.pkl /app/

# Commande par défaut
CMD ["python", "/app/flask_server.py"]