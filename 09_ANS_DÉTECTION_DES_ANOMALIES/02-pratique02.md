# Références :

- https://drive.google.com/drive/folders/1HrEu6KmgmiTs3QkqI9s0-mRs8JvnX1xr?usp=sharing

# Détection d'Anomalies avec DBSCAN

Ce document explique comment détecter des anomalies dans un ensemble de données en utilisant l'algorithme DBSCAN (Density-Based Spatial Clustering of Applications with Noise). Le script utilise Google Colab pour accéder aux fichiers, charger les données, les prétraiter, appliquer le clustering DBSCAN et visualiser les résultats.

### Approches

#### Algorithme DBSCAN

##### Description
DBSCAN est un algorithme de clustering basé sur la densité. Il forme des clusters en regroupant des points proches les uns des autres, en fonction de deux paramètres :
- **eps** : La distance maximale entre deux points pour les considérer comme voisins.
- **min_samples** : Le nombre minimum de points pour former un cluster dense.

Voici comment fonctionne DBSCAN :
1. Commencer avec un point non visité.
2. Récupérer tous les points densément accessibles à partir de ce point (en utilisant `eps` et `min_samples`).
3. Si le point est une core point (suffisamment de voisins), un cluster est formé.
4. Sinon, le point est considéré comme du bruit.
5. Répéter jusqu'à ce que tous les points soient visités.

##### Détection d'Anomalies
Les anomalies sont détectées comme des points marqués par le label `-1`, indiquant qu'ils ne font partie d'aucun cluster dense :
- **Anomalie** : Un point est considéré comme une anomalie s'il est marqué comme du bruit par DBSCAN.

### Explication du Code

#### Importation des Bibliothèques et Montage de Google Drive

```python
from google.colab import drive

# Monter Google Drive pour accéder aux fichiers
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
```
- `drive.mount('/content/drive')` : Cette ligne monte Google Drive pour permettre l'accès aux fichiers stockés dans Google Drive depuis Colab.

#### Chargement et Prétraitement des Données

```python
# Changer le répertoire de travail vers le chemin spécifié dans Google Drive
os.chdir('drive/My Drive/datacolab')

# Charger le dataset
df = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv')
# Afficher le nombre de lignes dans le dataset
print(len(df))
# Afficher les premières lignes du dataset
df.head()
```
- `os.chdir('drive/My Drive/datacolab')` : Change le répertoire de travail pour accéder au dossier contenant les données.
- `df = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv')` : Charge le dataset des prix de l'immobilier à Melbourne.

```python
# Sélectionner uniquement les colonnes numériques
df_num = df.select_dtypes(include=["float64", "int64"])

# Remplir les valeurs manquantes avec la médiane de chaque colonne
df_num.fillna(df_num.median(), inplace=True)
```
- `df.select_dtypes(include=["float64", "int64"])` : Sélectionne uniquement les colonnes numériques du dataset.
- `df_num.fillna(df_num.median(), inplace=True)` : Remplit les valeurs manquantes avec la médiane de chaque colonne.

```python
# Normaliser les données numériques pour le clustering
X = StandardScaler().fit_transform(df_num)
# Créer un DataFrame avec les données normalisées
X1 = pd.DataFrame(X, columns=df_num.columns)
# Afficher les premières lignes des données normalisées
X1.head()
```
- `StandardScaler().fit_transform(df_num)` : Normalise les données pour avoir une moyenne de 0 et un écart-type de 1.
- `pd.DataFrame(X, columns=df_num.columns)` : Crée un DataFrame avec les données normalisées.

#### Fonction pour Tracer les Résultats

```python
# Fonction pour tracer les résultats du modèle DBSCAN
def plot_model(labels, alg_name, plot_index):
    fig = plt.figure(figsize=(15, 15))  # Créer une figure pour le tracé
    ax = fig.add_subplot(1, 1, plot_index)  # Ajouter un sous-graphique
    color_code = {'anomaly': 'red', 'normal': 'green'}  # Définir les couleurs pour les anomalies et les points normaux
    colors = [color_code[x] for x in labels]  # Assigner des couleurs basées sur les étiquettes

    # Tracé en nuage de points des données avec des couleurs basées sur leurs étiquettes
    ax.scatter(X1.iloc[:, 0], X1.iloc[:, 1], color=colors, marker='.', label='red = anomaly')
    ax.legend(loc="lower right")  # Ajouter une légende au graphique
    ax.set_title(alg_name)  # Définir le titre du graphique
    plt.show()  # Afficher le graphique
```
- `fig = plt.figure(figsize=(15, 15))` : Crée une figure pour le tracé avec une taille de 15x15 pouces.
- `ax = fig.add_subplot(1, 1, plot_index)` : Ajoute un sous-graphique à la figure.
- `color_code = {'anomaly': 'red', 'normal': 'green'}` : Définit les couleurs pour les anomalies (rouge) et les points normaux (vert).
- `colors = [color_code[x] for x in labels]` : Assigne des couleurs aux points en fonction de leurs étiquettes.
- `ax.scatter(X1.iloc[:, 0], X1.iloc[:, 1], color=colors, marker='.', label='red = anomaly')` : Trace les données en nuage de points avec des couleurs basées sur leurs étiquettes.
- `ax.legend(loc="lower right")` : Ajoute une légende au graphique, placée en bas à droite.
- `ax.set_title(alg_name)` : Définit le titre du graphique.
- `plt.show()` : Affiche le graphique.

#### Application de l'Algorithme DBSCAN et Visualisation des Résultats

```python
# Appliquer l'algorithme de clustering DBSCAN
model = DBSCAN(eps=0.63).fit(X1)
# Obtenir les étiquettes assignées par DBSCAN
labels = model.labels_

# Étiqueter les anomalies et les points normaux
labels = [('anomaly' if x == -1 else 'normal') for x in labels]

# Tracer les résultats
plot_model(labels, 'DBSCAN', 1)
```
- `model = DBSCAN(eps=0.63).fit(X1)` : Applique l'algorithme DBSCAN aux données normalisées avec un `eps` de 0.63.
- `labels = model.labels_` : Obtient les étiquettes assignées par DBSCAN. Les points de bruit sont marqués avec le label `-1`.
- `labels = [('anomaly' if x == -1 else 'normal') for x in labels]` : Étiquette les points comme anomalies ou normaux en fonction de leurs étiquettes.
- `plot_model(labels, 'DBSCAN', 1)` : Trace les résultats du modèle DBSCAN en utilisant la fonction `plot_model`.

## Annexes

### Annexe : Code Complet

```python
from google.colab import drive

# Monter Google Drive pour accéder aux fichiers
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os

# Changer le répertoire de travail vers le chemin spécifié dans Google Drive
os.chdir('drive/My Drive/datacolab')

# Charger le dataset
df = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv')
# Afficher le nombre de lignes dans le dataset
print(len(df))
# Afficher les premières lignes du dataset
df.head()

# Sélectionner uniquement les colonnes numériques
df_num = df.select_dtypes(include=["float64", "int64"])

# Remplir les valeurs manquantes avec la médiane de chaque colonne
df_num.fillna(df_num.median(), inplace=True)

# Normaliser les données numériques pour le clustering
X = StandardScaler().fit_transform(df_num)
# Créer un DataFrame avec les données normalisées
X1 = pd.DataFrame(X, columns=df_num.columns)
# Afficher les premières lignes des données normalisées
X1.head()

# Fonction pour tracer les résultats du modèle DBSCAN
def plot_model(labels, alg_name, plot_index):
    fig = plt.figure(figsize=(15, 15))  # Créer une figure pour le tracé
    ax = fig.add_subplot(1, 1, plot_index)  # Ajouter un sous-graphique
    color_code = {'anomaly': 'red', 'normal': 'green'}  # Définir les couleurs pour les anomalies et les points normaux
    colors = [color_code[x] for x in labels]  # Assigner des couleurs basées sur les étiquettes

    # Tracé en nuage de points des données avec des couleurs basées sur leurs étiquettes
    ax.scatter(X1.iloc[:, 0], X1.iloc[:, 1], color=colors, marker='.', label='red = anomaly')
    ax.legend(loc="lower right")  # Ajouter une légende au graphique
    ax.set_title(alg_name)  # Définir le titre du graphique
   

 plt.show()  # Afficher le graphique

# Appliquer l'algorithme de clustering DBSCAN
model = DBSCAN(eps=0.63).fit(X1)
# Obtenir les étiquettes assignées par DBSCAN
labels = model.labels_

# Étiqueter les anomalies et les points normaux
labels = [('anomaly' if x == -1 else 'normal') for x in labels]

# Tracer les résultats
plot_model(labels, 'DBSCAN', 1)
```

Pour plus de détails sur les implémentations, voir les annexes ci-dessus.
