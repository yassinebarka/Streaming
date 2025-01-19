# Projet d'Analyse de Sentiments avec Kafka, Flask, PySpark et Logique Floue

## Description

Ce projet est une application de traitement de texte et d'analyse de sentiments utilisant Apache Kafka, Flask, PySpark, et un modèle de réseau de neurones récurrent (GRU) pour l'analyse des sentiments. La logique floue est intégrée pour améliorer l'interprétation des résultats de l'analyse des sentiments.

## Fonctionnalités

- **Flask Server** : Un serveur Flask qui reçoit des requêtes HTTP POST avec du texte, nettoie le texte, utilise un modèle GRU pour prédire le sentiment du texte, et renvoie le score de sentiment et le label flou.
- **Producteur Kafka** : Lit un fichier CSV contenant des textes, envoie chaque texte au serveur Flask pour obtenir un score de sentiment et un label flou, et envoie les résultats à un topic Kafka.
- **Consommateur Kafka** : Lit les messages du topic Kafka, compte les occurrences de différents scores de sentiment et labels flous, et affiche les résultats.

## Technologies Utilisées

- **Apache Kafka** : Pour la messagerie et le traitement des flux de données.
- **Flask** : Pour créer un serveur web léger pour l'analyse des sentiments.
- **PySpark** : Pour le traitement des données et les transformations.
- **Torch** : Pour le modèle de réseau de neurones GRU.
- **Pandas** : Pour la manipulation des données CSV.
- **scikit-fuzzy** : Pour la logique floue.

## Structure du Projet
