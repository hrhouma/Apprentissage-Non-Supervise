# voir les 4 études de cas dans le drive: 
- étude de cas 1 : Voir drive (Exécution sur COLAB)
- étude de cas 2 : Voir drive (Exécution sur COLAB)
- étude de cas 3 : Voir drive (Exécution sur COLAB)
- étude de cas 4 : Voir drive (https://towardsdatascience.com/time-series-anomaly-detection-with-pycaret-706a6e2b2427) (Exécution en local)

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




# Étude de cas 4 
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



---
# Annexe  :
----



### 1. **Importation des modules**

```python
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pycaret.anomaly import *
```

- **Pourquoi importer ces modules ?**
  - **tkinter** : C'est une bibliothèque standard en Python pour créer des interfaces graphiques. Elle permet de créer des fenêtres, des boutons, etc.
  - **pandas** : C'est une bibliothèque pour manipuler et analyser des données sous forme de tableaux. Pensez à des feuilles Excel.
  - **numpy** : C'est une bibliothèque pour faire des calculs mathématiques rapides, souvent avec des tableaux de nombres.
  - **plotly.express et plotly.graph_objects** : Ces bibliothèques permettent de créer des visualisations interactives, comme des graphiques.
  - **pycaret.anomaly** : PyCaret est une bibliothèque pour simplifier le machine learning. Ici, on l'utilise pour la détection d'anomalies.

### 2. **Définition de la fonction principale**

```python
def run_anomaly_detection():
    try:
```

- **Pourquoi une fonction ?**
  - Une fonction permet de regrouper du code pour le réutiliser facilement. Ici, la fonction `run_anomaly_detection` regroupe tout ce qui est nécessaire pour détecter les anomalies dans les données.

### 3. **Chargement des données**

```python
data = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv')
```

- **Pourquoi charger des données ?**
  - Pour faire de la détection d'anomalies, il faut avoir des données à analyser. Ici, on utilise des données de trajets en taxi à New York.

### 4. **Conversion et préparation des données**

```python
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', drop=True, inplace=True)
data = data.resample('H').sum()
```

- **Pourquoi convertir et préparer les données ?**
  - **Conversion** : On transforme la colonne 'timestamp' en format date/heure, pour que Python comprenne qu'il s'agit de temps.
  - **Mise à l'index** : On utilise 'timestamp' comme index, c'est-à-dire comme référence pour nos lignes.
  - **Resampling** : On regroupe les données par heure. C'est comme si on regroupait toutes les courses de taxi faites chaque heure.

### 5. **Initialisation de l'environnement PyCaret**

```python
s = setup(data, session_id=123, use_gpu=False)
```

- **Pourquoi initialiser PyCaret ?**
  - PyCaret simplifie le processus de machine learning. Ici, il prépare les données pour la détection d'anomalies.

### 6. **Création et entraînement du modèle Isolation Forest**

```python
iforest = create_model('iforest', fraction=0.1)
iforest_results = assign_model(iforest)
```

- **Pourquoi utiliser Isolation Forest ?**
  - **Isolation Forest** est un algorithme qui détecte les anomalies en isolant les données étranges. On l'entraîne ici sur nos données de trajets de taxi.

### 7. **Affichage des anomalies détectées**

```python
anomalies = iforest_results[iforest_results['Anomaly'] == 1]
print(anomalies.head())
```

- **Pourquoi afficher les anomalies ?**
  - Pour voir quelles données sont considérées comme inhabituelles ou "anormales". Ici, on montre les premières anomalies détectées.

### 8. **Visualisation des anomalies**

```python
fig = px.line(iforest_results, x=iforest_results.index, y="value", title='NYC TAXI TRIPS - UNSUPERVISED ANOMALY DETECTION', template='plotly_dark')
outlier_dates = anomalies.index
y_values = [iforest_results.loc[i]['value'] for i in outlier_dates]
fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode='markers', name='Anomaly', marker=dict(color='red', size=10)))
fig.show()
```

- **Pourquoi visualiser les anomalies ?**
  - Visualiser permet de mieux comprendre où se trouvent les anomalies sur un graphique. Ici, on voit les anomalies comme des points rouges.

### 9. **Messages d'information ou d'erreur**

```python
messagebox.showinfo("Success", "Anomaly detection completed successfully!")
```

- **Pourquoi afficher des messages ?**
  - Pour informer l'utilisateur si tout s'est bien passé ou s'il y a eu un problème.

### 10. **Création de la fenêtre principale**

```python
root = tk.Tk()
root.title("Anomaly Detection with PyCaret")
```

- **Pourquoi une fenêtre principale ?**
  - C'est l'interface utilisateur où l'on va cliquer pour lancer la détection d'anomalies.

### 11. **Création du bouton pour lancer la détection**

```python
run_button = tk.Button(root, text="Run Anomaly Detection", command=run_anomaly_detection)
run_button.pack(pady=20)
```

- **Pourquoi un bouton ?**
  - Pour que l'utilisateur puisse lancer la détection d'anomalies en cliquant sur un bouton.

### 12. **Boucle d'événements Tkinter**

```python
root.mainloop()
```

- **Pourquoi une boucle d'événements ?**
  - Cette boucle permet à la fenêtre de rester ouverte et d'attendre que l'utilisateur interagisse, comme cliquer sur le bouton.

### Conclusion

Ce code est une petite application qui permet à un utilisateur de charger des données, de détecter des anomalies à l'aide de l'algorithme Isolation Forest, et de visualiser ces anomalies. Chaque étape est conçue pour rendre le processus aussi simple que possible pour l'utilisateur final, tout en fournissant des retours visuels et des messages pour indiquer le succès ou l'échec des opérations.
