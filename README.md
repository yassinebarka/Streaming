# Projet d'Analyse de Sentiments avec Kafka, Flask, PySpark et Logique Floue

## Description

Ce projet implémente une application d'analyse de sentiments en flux continu. Utilisant Docker pour l'orchestration, l'application repose sur Apache Kafka, Flask, PySpark et des réseaux de neurones récurrents (GRU). La logique floue enrichit l'interprétation des résultats pour une analyse plus précise.

---

## Fonctionnalités

### 1. Serveur Flask
- Analyse les textes via une API REST.
- Nettoie les données et utilise un modèle GRU pour prédire les sentiments.
- Renvoie un score de sentiment et un label flou.

### 2. Producteur Kafka
- Lit des textes depuis un fichier CSV.
- Envoie les textes au serveur Flask pour analyse.
- Publie les résultats sur un topic Kafka.

### 3. Consommateur Kafka
- Lit les messages des sentiments depuis Kafka.
- Compte les occurrences des scores et labels flous.
- Affiche les statistiques en temps réel.

---

## Prérequis

- **Docker** : Assurez-vous que Docker et Docker Compose sont installés sur votre système.  
  [Télécharger Docker](https://www.docker.com/products/docker-desktop)

---

## Installation et Démarrage

### 1. Cloner le dépôt
```bash
git clone https://github.com/your-username/sentiment-analysis-kafka.git
cd sentiment-analysis-kafka
```

### 2. Construire et démarrer les services
Utilisez Docker Compose pour construire les images et démarrer les conteneurs :
```bash
docker-compose up --build
```

---

## Structure des Conteneurs

Le projet est orchestré à l'aide de `docker-compose`. Voici les services inclus :  
- **flask_server** : Fournit l'API REST pour l'analyse des sentiments.  
- **kafka** : Gère les flux de données en temps réel.  
- **zookeeper** : Nécessaire pour Kafka.  
- **producer** : Envoie les textes depuis un fichier CSV à Kafka.  
- **consumer** : Traite les messages publiés sur Kafka.  

---

## Utilisation

### 1. Ajouter des Textes
Ajoutez vos textes au fichier `data/input_texts.csv`.

### 2. Vérifier les Résultats
Les résultats des analyses seront publiés et visibles dans les logs du consommateur Kafka ou peuvent être intégrés dans un tableau de bord externe.

---

## Configuration Personnalisée

- **Modèle GRU** : Remplacez le modèle dans `model/sentiment_model.pt` si vous souhaitez utiliser votre propre réseau neuronal.
- **Topic Kafka** : Modifiez le topic dans le fichier `docker-compose.yml` si nécessaire.
- **Logique Floue** : Les règles de logique floue peuvent être ajustées dans le code du serveur Flask.

---

## Résultats

### Exemple de Résultat
- **Texte** : "Le produit est incroyable !"  
- **Score de Sentiment** : 0.92  
- **Label Flou** : "Très positif"  

---

## Arrêter les Services
Pour arrêter tous les conteneurs, utilisez :  
```bash
docker-compose down
```

---



## Licence

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.
