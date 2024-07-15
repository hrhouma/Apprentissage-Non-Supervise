# Exemple de code

## Lien pour les données : 
- https://drive.google.com/drive/folders/13OXLFmbp29cuEH3gTZhDSjb697iivN1J?usp=sharing
  
Ce README présente un exemple d'analyse comparative de différentes méthodes de clustering pour la détection de chutes à partir de données brutes. Cet exemple utilise les algorithmes K-means, DBSCAN et le clustering hiérarchique pour analyser les données des sujets.

> **Remarque :** Cet exemple est fourni à titre d'illustration et peut nécessiter des ajustements et des corrections. Il est de votre responsabilité de vérifier, corriger et interpréter les résultats obtenus.

### Étapes de l'Analyse

#### 1. Chargement et Prétraitement des Données

**Objectif :** Charger les fichiers CSV de chaque dossier et les combiner en un seul DataFrame pour chaque type de données (`raw`, `features`, `raw-all`).

```python
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler

# Définir les chemins vers les dossiers
raw_all_path = 'path/to/fall-dataset-all/*.csv'
features_path = 'path/to/fall-dataset-features/*.csv'
raw_path = 'path/to/fall-dataset-raw/*.csv'

# Charger et combiner les fichiers CSV
raw_all_files = glob.glob(raw_all_path)
features_files = glob.glob(features_path)
raw_files = glob.glob(raw_path)

# Fonction pour charger et combiner les fichiers
def load_and_combine(files):
    dataframes = [pd.read_csv(file) for file in files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# Combiner les données
combined_raw_all_data = load_and_combine(raw_all_files)
combined_features_data = load_and_combine(features_files)
combined_raw_data = load_and_combine(raw_files)

# Afficher les premières lignes pour vérification
print("Combined Raw All Data")
print(combined_raw_all_data.head())
print("\nCombined Features Data")
print(combined_features_data.head())
print("\nCombined Raw Data")
print(combined_raw_data.head())
```

# 2. Normalisation des Données

**Objectif :** Normaliser les données pour garantir une échelle uniforme pour toutes les caractéristiques.

```python
# Normalisation des données combinées
scaler = StandardScaler()
normalized_raw_all_data = scaler.fit_transform(combined_raw_all_data.drop(columns=['Timestamp']))
normalized_features_data = scaler.fit_transform(combined_features_data.drop(columns=['Timestamp']))
normalized_raw_data = scaler.fit_transform(combined_raw_data.drop(columns=['Timestamp', 'Fall']))
```

# 3. Sélection des Caractéristiques Pertinentes

**Objectif :** Sélectionner les caractéristiques les plus pertinentes pour l'analyse de clustering.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Calculer la matrice de corrélation
correlation_matrix = combined_features_data.corr()

# Tracer la carte de chaleur
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Sélectionner les caractéristiques ayant une forte corrélation avec la variable cible (Fall)
correlation_threshold = 0.5  # Exemple de seuil
relevant_features = correlation_matrix['Fall'][abs(correlation_matrix['Fall']) > correlation_threshold].index.tolist()

print(f"Relevant Features: {relevant_features}")
```

# 4. Application des Algorithmes de Clustering

##### K-means

**Objectif :** Appliquer l'algorithme K-means et déterminer le nombre optimal de clusters.

```python
from sklearn.cluster import KMeans

# Utiliser uniquement les caractéristiques pertinentes
relevant_data = combined_features_data[relevant_features].drop(columns=['Fall'])

# Déterminer le nombre optimal de clusters
def plot_elbow_method(data):
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    plt.plot(range(1, 11), sse)
    plt.xlabel('Nombre de clusters')
    plt.ylabel('SSE')
    plt.show()

plot_elbow_method(relevant_data)

# Appliquer K-means avec le nombre optimal de clusters (supposons k=4)
optimal_k = 4  # À ajuster selon la méthode du coude
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(relevant_data)
```

##### DBSCAN

**Objectif :** Appliquer l'algorithme DBSCAN avec différents paramètres.

```python
from sklearn.cluster import DBSCAN

# Appliquer DBSCAN aux données pertinentes
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(relevant_data)
```

##### Clustering Hiérarchique

**Objectif :** Appliquer la méthode de liaison agglomérative et visualiser les dendrogrammes.

```python
from scipy.cluster.hierarchy import dendrogram, linkage

# Appliquer le clustering hiérarchique aux données pertinentes
Z = linkage(relevant_data, 'ward')
dendrogram(Z)
plt.show()
```

# 5. Évaluation de la Qualité du Clustering

**Objectif :** Calculer et comparer les scores de silhouette et l'indice de Davies-Bouldin pour chaque algorithme.

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Scores de silhouette
silhouette_kmeans = silhouette_score(relevant_data, kmeans_labels)
silhouette_dbscan = silhouette_score(relevant_data, dbscan_labels)

# Indice de Davies-Bouldin
dbi_kmeans = davies_bouldin_score(relevant_data, kmeans_labels)
dbi_dbscan = davies_bouldin_score(relevant_data, dbscan_labels)

print(f"Silhouette Score K-means: {silhouette_kmeans}")
print(f"Davies-Bouldin Index K-means: {dbi_kmeans}")
print(f"Silhouette Score DBSCAN: {silhouette_dbscan}")
print(f"Davies-Bouldin Index DBSCAN: {dbi_dbscan}")
```

# Conclusion

Ce README fournit un exemple complet d'analyse de clustering pour la détection de chutes. Cet exemple n'est pas garanti d'être correct à 100% et peut nécessiter des ajustements. Il est de votre responsabilité de vérifier, corriger et interpréter les résultats obtenus. Utilisez cet exemple comme point de départ pour votre propre analyse et adaptation aux spécificités de votre dataset.
