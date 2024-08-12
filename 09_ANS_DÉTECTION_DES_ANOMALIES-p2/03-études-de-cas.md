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

- https://towardsdatascience.com/time-series-anomaly-detection-with-pycaret-706a6e2b2427

# Application de Détection d'Anomalies avec PyCaret et Tkinter

Ce guide fournit des instructions étape par étape pour installer, configurer et exécuter une application de détection d'anomalies utilisant PyCaret pour l'analyse et Tkinter pour l'interface graphique.

## Prérequis

- **Python** : Assurez-vous d'avoir Python installé (version 3.7 à 3.10 recommandée).
- **pip** : Assurez-vous que `pip` est installé et à jour.

## Installation

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
  
![image](https://github.com/user-attachments/assets/50d945c9-3895-471f-9689-4ef71cd689cc)


- **Sur macOS/Linux** :
  ```bash
  source pycaret-env/bin/activate
  ```

### Étape 2 : Installer les Dépendances

Installez PyCaret et les autres bibliothèques nécessaires dans l'environnement virtuel.

```bash
pip install pycaret[full]
pip install pandas numpy plotly
```

**Note** : Tkinter est inclus avec Python par défaut, donc aucune installation supplémentaire n'est nécessaire pour Tkinter.

## Exécution du Script

### Étape 3 : Créer le Script Python

Créez un fichier Python (par exemple, `anomaly_detection_tkinter.py`) et copiez le code suivant :

```python
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pycaret.anomaly import *

def run_anomaly_detection():
    try:
        # Load the NYC taxi passengers dataset
        data = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv')

        # Convert 'timestamp' to datetime format
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Prepare data for modeling
        data.set_index('timestamp', drop=True, inplace=True)
        data = data.resample('H').sum()

        # Initialize PyCaret environment for anomaly detection
        s = setup(data, session_id=123, use_gpu=False)

        # Create and train the Isolation Forest model
        iforest = create_model('iforest', fraction=0.1)
        iforest_results = assign_model(iforest)

        # Display detected anomalies
        anomalies = iforest_results[iforest_results['Anomaly'] == 1]
        print(anomalies.head())

        # Visualize detected anomalies
        fig = px.line(iforest_results, x=iforest_results.index, y="value", title='NYC TAXI TRIPS - UNSUPERVISED ANOMALY DETECTION', template='plotly_dark')
        outlier_dates = anomalies.index
        y_values = [iforest_results.loc[i]['value'] for i in outlier_dates]
        fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode='markers', name='Anomaly', marker=dict(color='red', size=10)))
        fig.show()

        messagebox.showinfo("Success", "Anomaly detection completed successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create the main window
root = tk.Tk()
root.title("Anomaly Detection with PyCaret")

# Create a button to run the anomaly detection
run_button = tk.Button(root, text="Run Anomaly Detection", command=run_anomaly_detection)
run_button.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()

```

### Étape 4 : Exécuter le Script

1. **Assurez-vous que l'environnement virtuel est activé**.
2. **Exécutez le script** avec la commande suivante :

   ```bash
   python anomaly_detection_tkinter.py
   ```

3. **Interagissez avec l'application** : Cliquez sur le bouton "Run Anomaly Detection" dans la fenêtre Tkinter pour exécuter l'analyse et visualiser les anomalies.

## Dépannage

- **ModuleNotFoundError** : Assurez-vous que toutes les bibliothèques requises sont installées dans l'environnement virtuel actif.
- **Problèmes de dépendances** : Mettez à jour `pip` et `setuptools` si nécessaire :
  ```bash
  pip install --upgrade pip setuptools
  ```

## Ressources

- [Documentation de PyCaret](https://pycaret.gitbook.io/docs/)
- [Guide d'utilisation de Plotly](https://plotly.com/python/)
- [Documentation Tkinter](https://docs.python.org/3/library/tkinter.html)


