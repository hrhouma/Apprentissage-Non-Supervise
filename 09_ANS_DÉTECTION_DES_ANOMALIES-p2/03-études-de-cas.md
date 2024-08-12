# voir les 4 études de cas : 
- étude de cas 1 : Voir drive
- étude de cas 2 : Voir drive
- étude de cas 3 : Voir drive
- étude de cas 4 : Voir drive (https://towardsdatascience.com/time-series-anomaly-detection-with-pycaret-706a6e2b2427)

- https://towardsdatascience.com/time-series-anomaly-detection-with-pycaret-706a6e2b2427

----


# Toutes les étapes mentionnées dans le tutoriel "Détection d'anomalies sur les séries temporelles avec PyCaret" :

```
+---------------------------------------------------------------+
| 1. Introduction à la Détection d'Anomalies sur Séries Temporelles |
+---------------------------------------------------------------+
| - Présentation de PyCaret : bibliothèque open-source de       |
|   machine learning en Python pour automatiser les workflows.  |
| - Objectifs du tutoriel :                                     |
|   * Comprendre la détection d'anomalies et ses types.         |
|   * Utiliser PyCaret pour entraîner et évaluer un modèle de   |
|     détection d'anomalies.                                    |
|   * Identifier et analyser les anomalies détectées.           |
+---------------------------------------------------------------+
                       |
                       v
+---------------------------------------------------------------+
| 2. Installation de PyCaret                                    |
+---------------------------------------------------------------+
| - Installation de PyCaret dans un environnement virtuel :     |
|   * Version slim : `pip install pycaret`                      |
|   * Version complète : `pip install pycaret[full]`            |
| - Vérification des dépendances nécessaires et optionnelles.   |
+---------------------------------------------------------------+
                       |
                       v
+---------------------------------------------------------------+
| 3. Compréhension de la Détection d'Anomalies                  |
+---------------------------------------------------------------+
| - Définition : Détecter des événements rares ou des valeurs   |
|   aberrantes qui diffèrent significativement de la majorité   |
|   des données.                                                |
| - Types d'algorithmes :                                       |
|   * Supervisé : nécessite des étiquettes pour les anomalies.  |
|   * Non supervisé : apprentissage à partir de données non     |
|     étiquetées.                                               |
|   * Semi-supervisé : entraînement sur données normales, puis  |
|     détection sur nouvelles données.                          |
| - Cas d'utilisation : détection de fraude, problèmes          |
|   structurels, diagnostics médicaux, etc.                     |
+---------------------------------------------------------------+
                       |
                       v
+---------------------------------------------------------------+
| 4. Chargement et Préparation des Données                      |
+---------------------------------------------------------------+
| - Jeu de données utilisé : nombre de passagers de taxis à NYC.|
| - Chargement des données et conversion du format 'timestamp'  |
|   en datetime.                                                |
|   * `data = pd.read_csv('url_du_dataset')`                    |
|   * `data['timestamp'] = pd.to_datetime(data['timestamp'])`   |
| - Création de moyennes mobiles pour les analyses temporelles. |
|   * `data['MA48'] = data['value'].rolling(48).mean()`         |
|   * `data['MA336'] = data['value'].rolling(336).mean()`       |
| - Visualisation des séries temporelles et des moyennes mobiles|
|   avec `plotly`.                                              |
+---------------------------------------------------------------+
                       |
                       v
+---------------------------------------------------------------+
| 5. Préparation des Caractéristiques                           |
+---------------------------------------------------------------+
| - Suppression des colonnes inutiles :                         |
|   * `data.drop(['MA48', 'MA336'], axis=1, inplace=True)`      |
| - Définir 'timestamp' comme index des données.                |
|   * `data.set_index('timestamp', drop=True, inplace=True)`    |
| - Rééchantillonnage des séries temporelles en intervalles     |
|   horaires :                                                  |
|   * `data = data.resample('H').sum()`                         |
| - Extraction des caractéristiques temporelles :               |
|   * Jour, nom du jour, jour de l'année, semaine, heure, etc.  |
|   * `data['day'] = [i.day for i in data.index]`               |
|   * `data['hour'] = [i.hour for i in data.index]`             |
|   * ... et ainsi de suite pour chaque caractéristique.        |
+---------------------------------------------------------------+
                       |
                       v
+---------------------------------------------------------------+
| 6. Configuration de l'Expérience avec PyCaret                 |
+---------------------------------------------------------------+
| - Initialisation de la configuration avec `setup` :           |
|   * `from pycaret.anomaly import *`                           |
|   * `s = setup(data, session_id=123)`                         |
| - PyCaret profile automatiquement les données et infère les   |
|   types des caractéristiques (catégorielles ou numériques).   |
+---------------------------------------------------------------+
                       |
                       v
+---------------------------------------------------------------+
| 7. Entraînement du Modèle de Détection d'Anomalies            |
+---------------------------------------------------------------+
| - Vérification des modèles disponibles avec `models()`.       |
| - Création du modèle Isolation Forest pour la détection d'anomalies :|
|   * `iforest = create_model('iforest', fraction=0.1)`         |
| - Assignation des résultats au modèle :                       |
|   * `iforest_results = assign_model(iforest)`                 |
| - Ajout des colonnes 'Anomaly' (1 pour les anomalies, 0 pour  |
|   les points normaux) et 'Anomaly_Score' (score de décision). |
| - Exemple de filtrage des anomalies détectées :               |
|   * `iforest_results[iforest_results['Anomaly'] == 1].head()` |
+---------------------------------------------------------------+
                       |
                       v
+---------------------------------------------------------------+
| 8. Visualisation et Analyse des Anomalies                     |
+---------------------------------------------------------------+
| - Visualisation des anomalies sur un graphique temporel avec  |
|   `plotly` :                                                  |
|   * Tracer les valeurs normales et les anomalies en rouge.    |
|   * `fig.add_trace(go.Scatter(x=outlier_dates, y=y_values,    |
|      mode='markers', name='Anomaly', marker=dict(color='red', |
|      size=10)))`                                              |
| - Analyse des anomalies détectées :                           |
|   * Par exemple, des anomalies détectées autour du 1er janvier|
|     (Réveillon du Nouvel An) ou lors de la tempête de neige   |
|     nord-américaine en janvier.                               |
+---------------------------------------------------------------+
                       |
                       v
+---------------------------------------------------------------+
| 9. Conclusion et Étapes Suivantes                             |
+---------------------------------------------------------------+
| - Résumé de la simplicité d'utilisation et de l'efficacité de |
|   PyCaret pour la détection d'anomalies.                      |
| - Annonce des prochains tutoriels sur l'utilisation de PyCaret|
|   pour d'autres modules (ex. régression).                     |
+---------------------------------------------------------------+
                       |
                       v
+---------------------------------------------------------------+
| 10. Références et Ressources Supplémentaires                  |
+---------------------------------------------------------------+
| - Documentation de PyCaret, tutoriels, GitHub, etc.           |
| - Lien pour rejoindre la communauté PyCaret sur Slack.        |
| - Tutoriels recommandés pour approfondir l'utilisation de     |
|   PyCaret dans différents contextes (déploiement, AutoML, etc.)|
+---------------------------------------------------------------+
```





---

# Configuration et Exécution de la Détection d'Anomalies avec PyCaret

Ce guide vous guidera à travers l'installation, la configuration et l'exécution d'un script de détection d'anomalies en utilisant PyCaret. Le script analyse un jeu de données de trajets en taxi à New York.

## Prérequis

- **Python** : Assurez-vous d'avoir Python (version 3.7 à 3.10) installé sur votre machine.
- **pip** : Assurez-vous que `pip` est installé et à jour.

## Instructions d'Installation

### Étape 1 : Créer un Environnement Virtuel

Il est recommandé d'utiliser un environnement virtuel pour éviter les conflits de dépendances.

```bash
python -m venv pycaret-env
```

Activez l'environnement virtuel :

- **Sur Windows** :
  ```bash
  pycaret-env\Scripts\activate
  ```

- **Sur macOS/Linux** :
  ```bash
  source pycaret-env/bin/activate
  ```

### Étape 2 : Installer PyCaret et ses Dépendances

Installez PyCaret dans l'environnement virtuel. Vous pouvez choisir d'installer la version de base ou la version complète avec toutes les dépendances optionnelles.

```bash
pip install pycaret
```

Pour la version complète avec toutes les dépendances :

```bash
pip install pycaret[full]
```

### Étape 3 : Vérifier l'Installation

Vérifiez que PyCaret est installé correctement en important le module dans un shell Python.

```python
from pycaret.anomaly import *
```

## Exécution du Script de Détection d'Anomalies

Copiez le code suivant dans un fichier Python (par exemple, `anomaly_detection.py`) et exécutez-le.

```python
# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import plotly.express as px
from pycaret.anomaly import *

# Charger le jeu de données des passagers de taxi à NYC
data = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv')

# Conversion de la colonne 'timestamp' en format datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Préparation des données pour la modélisation
data.set_index('timestamp', drop=True, inplace=True)
data = data.resample('H').sum()

# Initialisation de l'environnement PyCaret pour la détection d'anomalies
s = setup(data, session_id=123, use_gpu=False)

# Vérification des modèles disponibles pour la détection d'anomalies
print(models())

# Création et entraînement du modèle Isolation Forest
iforest = create_model('iforest', fraction=0.1)

# Assigner les résultats du modèle aux données d'origine
iforest_results = assign_model(iforest)

# Afficher les anomalies détectées
anomalies = iforest_results[iforest_results['Anomaly'] == 1]
print(anomalies.head())

# Visualisation des anomalies
fig = px.line(iforest_results, x=iforest_results.index, y="value", title='NYC TAXI TRIPS - UNSUPERVISED ANOMALY DETECTION', template='plotly_dark')
outlier_dates = anomalies.index
y_values = [iforest_results.loc[i]['value'] for i in outlier_dates]
fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode='markers', name='Anomaly', marker=dict(color='red', size=10)))
fig.show()
```

## Dépannage

- **ModuleNotFoundError** : Assurez-vous que PyCaret est installé dans l'environnement virtuel actif.
- **Problèmes de dépendances** : Mettez à jour `pip` et `setuptools` :
  ```bash
  pip install --upgrade pip setuptools
  ```

## Ressources

- [Documentation de PyCaret](https://pycaret.gitbook.io/docs/)
- [Guide d'utilisation de Plotly](https://plotly.com/python/)

