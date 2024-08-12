- https://towardsdatascience.com/time-series-anomaly-detection-with-pycaret-706a6e2b2427

```bash
pip install pycaret[full]
pip install pandas numpy plotly
```

# Anomaly Detection with PyCaret

## 1. Introduction

Ce projet est une application simple développée en Python qui permet de détecter des anomalies dans un ensemble de données, en utilisant la bibliothèque PyCaret pour le machine learning. L'application propose une interface graphique (GUI) créée avec Tkinter, où l'utilisateur peut sélectionner un modèle d'anomalie parmi ceux disponibles dans PyCaret, et visualiser les anomalies détectées.

## 2. Prérequis

Avant de commencer, assurez-vous d'avoir installé les bibliothèques Python nécessaires. Vous pouvez installer toutes les dépendances en utilisant `pip` :

```bash
pip install pandas numpy plotly pycaret tk
```

## 3. Explication du code

Le code est organisé de manière à ce que l'utilisateur puisse facilement charger les données, sélectionner un modèle d'anomalie, exécuter la détection, et visualiser les résultats. Voici une explication détaillée des différentes parties du code.

### 3.1 Importation des modules

```python
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pycaret.anomaly import *
```

- **tkinter** : Utilisé pour créer l'interface graphique de l'application.
- **pandas** : Utilisé pour manipuler les données sous forme de DataFrames.
- **numpy** : Utilisé pour les opérations numériques sur les tableaux.
- **plotly.express et plotly.graph_objects** : Utilisés pour la création de visualisations interactives.
- **pycaret.anomaly** : Importé pour simplifier la détection d'anomalies en utilisant différents modèles de machine learning.

### 3.2 Chargement et préparation des données

```python
data = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', drop=True, inplace=True)
data = data.resample('H').sum()
```

- **Chargement des données** : Les données sur les trajets de taxi à New York sont chargées depuis un fichier CSV.
- **Conversion en datetime** : La colonne `timestamp` est convertie en format date/heure pour faciliter les opérations temporelles.
- **Mise à l'index** : La colonne `timestamp` est utilisée comme index pour le DataFrame.
- **Resampling** : Les données sont regroupées par heure pour une analyse plus précise.

### 3.3 Initialisation de l'environnement PyCaret

```python
s = setup(data, session_id=123, use_gpu=False)
```

- **setup()** : Cette fonction initialise l'environnement PyCaret pour la détection d'anomalies. Elle prépare les données pour le modèle, en s'assurant que tous les paramètres sont configurés correctement.

### 3.4 Liste des modèles disponibles

```python
models_list = models().index.tolist()
```

- **models()** : Cette fonction récupère la liste des modèles d'anomalies disponibles dans PyCaret. Ces modèles peuvent ensuite être sélectionnés via une liste déroulante dans l'interface utilisateur.

### 3.5 Création de l'interface graphique

```python
root = tk.Tk()
root.title("Anomaly Detection with PyCaret")
```

- **tk.Tk()** : Crée la fenêtre principale de l'application.
- **title()** : Définit le titre de la fenêtre.

### 3.6 Menu déroulant pour la sélection du modèle

```python
model_var = tk.StringVar(root)
model_var.set(models_list[0])
model_menu = tk.OptionMenu(root, model_var, *models_list)
model_menu.pack(pady=20)
```

- **tk.StringVar()** : Crée une variable pour stocker la sélection de l'utilisateur dans la liste déroulante.
- **tk.OptionMenu()** : Crée une liste déroulante avec tous les modèles disponibles pour la sélection.

### 3.7 Bouton pour lancer la détection d'anomalies

```python
run_button = tk.Button(root, text="Run Anomaly Detection", command=run_anomaly_detection)
run_button.pack(pady=20)
```

- **tk.Button()** : Crée un bouton que l'utilisateur peut cliquer pour lancer la détection d'anomalies en utilisant le modèle sélectionné.

### 3.8 Détection et visualisation des anomalies

```python
selected_model = model_var.get()
model = create_model(selected_model, fraction=0.1)
model_results = assign_model(model)
```

- **get()** : Récupère le modèle sélectionné par l'utilisateur dans la liste déroulante.
- **create_model()** : Crée et entraîne le modèle de détection d'anomalies sélectionné.
- **assign_model()** : Identifie et marque les anomalies dans les données.

### 3.9 Visualisation des résultats

```python
fig = px.line(model_results, x=model_results.index, y="value", title=f'NYC TAXI TRIPS - {selected_model.upper()} ANOMALY DETECTION', template='plotly_dark')
outlier_dates = anomalies.index
y_values = [model_results.loc[i]['value'] for i in outlier_dates]
fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode='markers', name='Anomaly', marker=dict(color='red', size=10)))
fig.show()
```

- **Visualisation** : Un graphique interactif est généré pour montrer les anomalies détectées dans les données. Les anomalies sont marquées en rouge.

### 3.10 Boucle d'événements Tkinter

```python
root.mainloop()
```

- **mainloop()** : Démarre la boucle d'événements Tkinter, maintenant la fenêtre ouverte jusqu'à ce que l'utilisateur la ferme.

## 4. Comment utiliser l'application

1. **Lancer le script** : Exécutez le script Python (`t.py`) pour ouvrir l'application.
2. **Sélection du modèle** : Dans la liste déroulante, sélectionnez un modèle d'anomalie.
3. **Exécution de la détection** : Cliquez sur le bouton "Run Anomaly Detection" pour exécuter la détection d'anomalies.
4. **Visualisation des résultats** : Les anomalies détectées seront affichées dans un graphique interactif.

## 5. Conclusion

Cette application montre comment utiliser PyCaret pour la détection d'anomalies avec une interface graphique simple en Python. Elle permet aux utilisateurs de choisir parmi différents modèles, d'exécuter une détection d'anomalies, et de visualiser les résultats de manière intuitive.

# Tous le code :

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

        # Get the selected model from the dropdown
        selected_model = model_var.get()

        # Create and train the selected model
        model = create_model(selected_model, fraction=0.1)
        model_results = assign_model(model)

        # Display detected anomalies
        anomalies = model_results[model_results['Anomaly'] == 1]
        print(anomalies.head())

        # Visualize detected anomalies
        fig = px.line(model_results, x=model_results.index, y="value", title=f'NYC TAXI TRIPS - {selected_model.upper()} ANOMALY DETECTION', template='plotly_dark')
        outlier_dates = anomalies.index
        y_values = [model_results.loc[i]['value'] for i in outlier_dates]
        fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode='markers', name='Anomaly', marker=dict(color='red', size=10)))
        fig.show()

        messagebox.showinfo("Success", "Anomaly detection completed successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create the main window
root = tk.Tk()
root.title("Anomaly Detection with PyCaret")

# Load the NYC taxi passengers dataset for initializing the setup
data = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', drop=True, inplace=True)
data = data.resample('H').sum()

# Initialize PyCaret environment to get the list of models
s = setup(data, session_id=123, use_gpu=False)
models_list = models().index.tolist()

# Create a StringVar to store the selected model
model_var = tk.StringVar(root)
model_var.set(models_list[0])  # Set the default model to the first one in the list

# Create a dropdown menu for selecting the model
model_menu = tk.OptionMenu(root, model_var, *models_list)
model_menu.pack(pady=20)

# Create a button to run the anomaly detection
run_button = tk.Button(root, text="Run Anomaly Detection", command=run_anomaly_detection)
run_button.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()

```


---
# Intreprétation :
---

## 6. Explication du Graphique de Détection d'Anomalies

Le graphique ci-dessous représente une analyse des anomalies détectées dans les trajets de taxi à New York entre juillet 2014 et février 2015.


![NYC Taxi Trips - Anomaly Detection](https://github.com/user-attachments/assets/f3a0528d-347f-45aa-be4e-125723610953)


### Contexte Général

Ce graphique montre le nombre de trajets en taxi à chaque heure durant la période analysée. L'axe horizontal représente le temps (de juillet 2014 à février 2015), et l'axe vertical représente le nombre de trajets en taxi effectués.

### Les Éléments du Graphique

- **Ligne bleue** : Représente le nombre de trajets en taxi à chaque heure. Cette ligne fluctue au fil du temps, reflétant les variations normales d'activité.
- **Points rouges** : Représentent les anomalies détectées. Ce sont des moments où le nombre de trajets est nettement différent de la norme attendue.

### Pourquoi y a-t-il des Anomalies ?

Les anomalies se produisent lorsque le nombre de trajets en taxi s'écarte considérablement du comportement attendu. Voici deux types de scénarios qui peuvent conduire à des anomalies :

1. **Augmentation soudaine** : Par exemple, un grand événement, comme le Nouvel An ou un grand concert, pourrait faire augmenter le nombre de trajets en taxi bien au-delà de la normale. Ces pics sont marqués comme des anomalies (points rouges situés en haut de la ligne bleue).

2. **Diminution soudaine** : Inversement, un événement inattendu, comme une tempête de neige ou une panne générale, pourrait réduire drastiquement le nombre de trajets. Ces baisses sont également considérées comme des anomalies (points rouges situés en bas de la ligne bleue).

### Conclusion

Ce graphique est une illustration visuelle des anomalies dans les trajets de taxi à New York. Les points rouges indiquent des moments où quelque chose d'inhabituel s'est produit, conduisant à un nombre de trajets en taxi très différent de ce qui est attendu. Ces anomalies sont importantes à détecter car elles peuvent révéler des événements ou des comportements exceptionnels dans les données.
