import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Définir les variables floues
sentiment_score = ctrl.Antecedent(np.arange(0, 11, 1), 'sentiment_score')
sentiment_label = ctrl.Consequent(np.arange(0, 11, 1), 'sentiment_label')

# Définir les ensembles flous pour le score de sentiment
sentiment_score['low'] = fuzz.trimf(sentiment_score.universe, [0, 0, 5])
sentiment_score['medium'] = fuzz.trimf(sentiment_score.universe, [0, 5, 10])
sentiment_score['high'] = fuzz.trimf(sentiment_score.universe, [5, 10, 10])

# Définir les ensembles flous pour le label de sentiment
sentiment_label['negative'] = fuzz.trimf(sentiment_label.universe, [0, 0, 5])
sentiment_label['neutral'] = fuzz.trimf(sentiment_label.universe, [0, 5, 10])
sentiment_label['positive'] = fuzz.trimf(sentiment_label.universe, [5, 10, 10])

# Définir les règles floues
rule1 = ctrl.Rule(sentiment_score['low'], sentiment_label['negative'])
rule2 = ctrl.Rule(sentiment_score['medium'], sentiment_label['neutral'])
rule3 = ctrl.Rule(sentiment_score['high'], sentiment_label['positive'])

# Créer le système de contrôle flou
sentiment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
sentiment_sim = ctrl.ControlSystemSimulation(sentiment_ctrl)

def fuzzy_sentiment_analysis(score):
    sentiment_sim.input['sentiment_score'] = score
    sentiment_sim.compute()
    return sentiment_sim.output['sentiment_label']