# Table des matières

1. Introduction au clustering
2. Bases du clustering
3. Clustering K-means
4. DÉMO _ Clustering K-means en Python
5. Example - Création d'un nouveau carnet pour le clustering
6. Assignment _ Clustering K-means
7. Solution _ Clustering K-means
8. Clustering Hiérarchique
9. DÉMO _ Clustering Hiérarchique en Python
10. Example _ Clustering hiérarchique en Python
11. Assignment _ Clustering Hiérarchique
12. Solution _ Clustering Hiérarchique
13. DBSCAN
14. DBSCAN en Python
15. DÉMO _ DBSCAN en Python
16. Assignment _ DBSCAN
17. Solution _ DBSCAN
18. Score de silhouette
19. DÉMO _ Score de silhouette en Python
20. Assignment _ Score de silhouette
21. Solution _ Score de silhouette
22. Comparaison des algorithmes de clustering
23. Étapes suivantes du clustering

---

# 1. Introduction au clustering

Le clustering est une méthode d'apprentissage automatique non supervisé qui consiste à regrouper des objets similaires dans des groupes appelés clusters. Cette technique est utilisée pour identifier des structures cachées dans des données sans étiquettes prédéfinies. Le clustering est largement utilisé dans diverses applications telles que la segmentation de marché, la détection de fraudes, l'analyse d'images, et plus encore.

---

# 2. Bases du clustering

## Qu'est-ce que le clustering ?

Le clustering est le processus de division d'un ensemble de données en sous-ensembles significatifs appelés clusters, où les objets dans chaque cluster sont plus similaires les uns aux autres qu'aux objets des autres clusters. Les techniques de clustering les plus courantes incluent K-means, le clustering hiérarchique et DBSCAN.

## Applications du clustering

Le clustering est utilisé dans de nombreux domaines pour des applications diverses, notamment :
- **Segmentation de marché** : Identifier des groupes de clients ayant des comportements similaires.
- **Détection de fraudes** : Identifier des transactions anormales.
- **Analyse de texte** : Grouper des documents similaires.
- **Analyse d'images** : Identifier des motifs ou des objets dans des images.

---

# 3. Clustering K-means

## Introduction

Le clustering K-means est l'une des méthodes de clustering les plus populaires. Il vise à partitionner n observations en k clusters où chaque observation appartient au cluster avec la moyenne la plus proche (centreide ou centre de cluster).

## Comment ça marche ?

1. **Initialisation** : Choisir k centres de clusters initiaux (aléatoirement ou en utilisant des heuristiques).
2. **Assignment** : Attribuer chaque point de données au cluster le plus proche en fonction de la distance euclidienne.
3. **Update** : Calculer le nouveau centroïde de chaque cluster.
4. **Répéter** : Répéter les étapes d'assignation et de mise à jour jusqu'à convergence (les centres de clusters ne changent plus).

---

# 4. DÉMO _ Clustering K-means en Python

## Étape 1 : Importer les bibliothèques nécessaires

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

## Étape 2 : Charger et préparer le jeu de données

```python
# Charger le jeu de données
data = pd.read_csv('cereal.csv')

# Sélectionner les caractéristiques numériques pertinentes
numeric_data = data[['Calories', 'Protein', 'Fat', 'Sodium', 'Fiber']]
```

## Étape 3 : Normaliser les données

```python
# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
```

## Étape 4 : Appliquer K-means

```python
# Appliquer K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_

# Ajouter les étiquettes de clusters au DataFrame
data['Cluster'] = labels
```

## Étape 5 : Visualiser les clusters

```python
# Visualiser les clusters
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels, cmap='viridis')
plt.title("Clustering K-means")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

---

# 5. Example - Création d'un nouveau carnet pour le clustering

Dans cette section, nous allons créer un nouveau carnet pour appliquer K-means sur un jeu de données spécifique.

## Étape 1 : Charger les données

```python
import pandas as pd

# Charger le jeu de données
data = pd.read_csv('Entertainment_Clean.csv')
```

## Étape 2 : Vérifier les données

```python
# Afficher les premières lignes du DataFrame
print(data.head())

# Vérifier les types de données
print(data.info())
```

## Étape 3 : Normaliser les données

```python
from sklearn.preprocessing import StandardScaler

# Sélectionner les caractéristiques numériques
numeric_data = data.select_dtypes(include=[np.number])

# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
```

## Étape 4 : Appliquer K-means

```python
from sklearn.cluster import KMeans

# Appliquer K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_

# Ajouter les étiquettes de clusters au DataFrame
data['Cluster'] = labels
```

## Étape 5 : Visualiser les clusters

```python
import matplotlib.pyplot as plt

# Visualiser les clusters
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels, cmap='viridis')
plt.title("Clustering K-means sur le jeu de données Entertainment")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

---

# 6. Assignment _ Clustering K-means

## Objectif

Votre mission est d'appliquer l'algorithme de clustering K-means sur un nouveau jeu de données, de déterminer le nombre optimal de clusters, et d'interpréter les résultats.

## Étapes

1. **Charger le jeu de données** : Utilisez le fichier `new_data.csv`.
2. **Préparer les données** : Normalisez les caractéristiques numériques.
3. **Appliquer K-means** : Choisissez le nombre optimal de clusters en utilisant la méthode du coude.
4. **Interpréter les résultats** : Analysez les clusters formés.

## Critères d'évaluation

- Préparation adéquate des données.
- Sélection correcte du nombre de clusters.
- Interprétation claire et précise des résultats.

---

# 7. Solution _ Clustering K-means

## Étape 1 : Charger le jeu de données

```python
import pandas as pd

# Charger le jeu de données
data = pd.read_csv('new_data.csv')
```

## Étape 2 : Préparer les données

```python
from sklearn.preprocessing import StandardScaler

# Sélectionner les caractéristiques numériques
numeric_data = data.select_dtypes(include=[np.number])

# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
```

## Étape 3 : Appliquer K-means et choisir le nombre optimal de clusters

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Méthode du coude pour choisir le nombre optimal de clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

# Tracer la courbe de la méthode du coude
plt.plot(range(1, 11), sse, marker='o')
plt.title('Méthode du coude')
plt.xlabel('Nombre de clusters')
plt.ylabel('SSE')
plt.show()
```

## Étape 4 : Appliquer K-means avec le nombre optimal de clusters

```python
# Appliquer K-means avec le nombre optimal de clusters (ex. 3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_

# Ajouter les étiquettes de clusters au DataFrame
data['Cluster'] = labels
```

## Étape 5 : Interpréter les résultats

```python
# Analyser les clusters formés
print(data.groupby('Cluster').mean())

# Visualiser les clusters
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels, cmap='viridis')
plt.title("Clustering K-means")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

---

# 8. Clustering Hiérarchique

## Introduction

Le clustering hiérarchique est une méthode de clustering qui cherche à créer une hiérarchie de clusters. Il existe deux approches principales : le clustering hiérarchique agglomératif (ascendant) et le clustering hiérarchique divisif (descendant).

## Comment ça marche ?

1. **Agglomératif** :
    - Chaque point commence comme un cluster unique.
    - À chaque étape, les deux clusters les plus similaires sont fusionnés.
    - Ce processus continue

 jusqu'à ce qu'il ne reste qu'un seul cluster.
    
2. **Divisif** :
    - Tous les points commencent dans un seul cluster.
    - À chaque étape, un cluster est divisé en deux clusters.
    - Ce processus continue jusqu'à ce que chaque point soit dans son propre cluster.

## Types de liaisons

- **Single linkage** : La distance entre deux clusters est la distance minimale entre deux points, un de chaque cluster.
- **Complete linkage** : La distance entre deux clusters est la distance maximale entre deux points, un de chaque cluster.
- **Average linkage** : La distance entre deux clusters est la distance moyenne entre tous les points des deux clusters.

---

# 9. DÉMO _ Clustering Hiérarchique en Python

## Étape 1 : Importer les bibliothèques nécessaires

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
```

## Étape 2 : Charger et préparer le jeu de données

```python
# Charger le jeu de données
data = pd.read_csv('cereal.csv')

# Sélectionner les caractéristiques numériques pertinentes
numeric_data = data[['Calories', 'Protein', 'Fat', 'Sodium', 'Fiber']]
```

## Étape 3 : Normaliser les données

```python
# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
```

## Étape 4 : Créer les liens hiérarchiques

```python
# Créer les liens hiérarchiques en utilisant la méthode de Ward
Z = linkage(scaled_data, method='ward')

# Créer le dendrogramme
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrogramme des Données Normalisées")
plt.xlabel("Points de Données")
plt.ylabel("Distance Euclidienne")
plt.show()
```

## Étape 5 : Appliquer le clustering hiérarchique agglomératif

```python
# Appliquer le clustering hiérarchique agglomératif
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
model.fit(scaled_data)
labels = model.labels_

# Ajouter les étiquettes de clusters au DataFrame
data['Cluster'] = labels
```

## Étape 6 : Visualiser les clusters

```python
# Visualiser les clusters
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels, cmap='viridis')
plt.title("Clustering Hiérarchique Agglomératif")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

---

# 10. Example _ Clustering hiérarchique en Python

## Étape 1 : Charger les données

```python
import pandas as pd

# Charger le jeu de données
data = pd.read_csv('Entertainment_Clean.csv')
```

## Étape 2 : Vérifier les données

```python
# Afficher les premières lignes du DataFrame
print(data.head())

# Vérifier les types de données
print(data.info())
```

## Étape 3 : Normaliser les données

```python
from sklearn.preprocessing import StandardScaler

# Sélectionner les caractéristiques numériques
numeric_data = data.select_dtypes(include=[np.number])

# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
```

## Étape 4 : Créer les liens hiérarchiques

```python
from scipy.cluster.hierarchy import dendrogram, linkage

# Créer les liens hiérarchiques en utilisant la méthode de Ward
Z = linkage(scaled_data, method='ward')

# Créer le dendrogramme
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrogramme des Données Normalisées")
plt.xlabel("Points de Données")
plt.ylabel("Distance Euclidienne")
plt.show()
```

## Étape 5 : Appliquer le clustering hiérarchique agglomératif

```python
from sklearn.cluster import AgglomerativeClustering

# Appliquer le clustering hiérarchique agglomératif
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
model.fit(scaled_data)
labels = model.labels_

# Ajouter les étiquettes de clusters au DataFrame
data['Cluster'] = labels
```

## Étape 6 : Visualiser les clusters

```python
import matplotlib.pyplot as plt

# Visualiser les clusters
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels, cmap='viridis')
plt.title("Clustering Hiérarchique Agglomératif sur le jeu de données Entertainment")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

---

# 11. Assignment _ Clustering Hiérarchique

## Objectif

Votre mission est d'appliquer l'algorithme de clustering hiérarchique sur un nouveau jeu de données, de déterminer le nombre optimal de clusters en utilisant un dendrogramme, et d'interpréter les résultats.

## Étapes

1. **Charger le jeu de données** : Utilisez le fichier `new_data.csv`.
2. **Préparer les données** : Normalisez les caractéristiques numériques.
3. **Créer un dendrogramme** : Identifiez le nombre optimal de clusters.
4. **Appliquer le clustering hiérarchique** : Utilisez le nombre de clusters identifié.
5. **Interpréter les résultats** : Analysez les clusters formés.

## Critères d'évaluation

- Préparation adéquate des données.
- Identification correcte du nombre de clusters.
- Interprétation claire et précise des résultats.

---

# 12. Solution _ Clustering Hiérarchique

## Étape 1 : Charger le jeu de données

```python
import pandas as pd

# Charger le jeu de données
data = pd.read_csv('new_data.csv')
```

## Étape 2 : Préparer les données

```python
from sklearn.preprocessing import StandardScaler

# Sélectionner les caractéristiques numériques
numeric_data = data.select_dtypes(include=[np.number])

# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
```

## Étape 3 : Créer un dendrogramme

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Créer les liens hiérarchiques en utilisant la méthode de Ward
Z = linkage(scaled_data, method='ward')

# Créer le dendrogramme
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrogramme des Données Normalisées")
plt.xlabel("Points de Données")
plt.ylabel("Distance Euclidienne")
plt.show()
```

## Étape 4 : Identifier le nombre optimal de clusters

En regardant le dendrogramme, nous pouvons identifier le nombre optimal de clusters et ajuster le seuil de couleur pour afficher ce nombre de clusters.

```python
# Ajuster le seuil de couleur pour visualiser les clusters
plt.figure(figsize=(10, 7))
dendrogram(Z, color_threshold=7.5)
plt.title("Dendrogramme avec Seuil de Couleur Ajusté")
plt.xlabel("Points de Données")
plt.ylabel("Distance Euclidienne")
plt.show()
```

## Étape 5 : Appliquer le clustering hiérarchique agglomératif

```python
from sklearn.cluster import AgglomerativeClustering

# Appliquer le clustering hiérarchique agglomératif avec le nombre optimal de clusters (ex. 3)
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
model.fit(scaled_data)
labels = model.labels_

# Ajouter les étiquettes de clusters au DataFrame
data['Cluster'] = labels
```

## Étape 6 : Interpréter les résultats

```python
# Analyser les clusters formés
print(data.groupby('Cluster').mean())

# Visualiser les clusters
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=labels, cmap='viridis')
plt.title("Clustering Hiérarchique Agglomératif")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

---

# 13. DBSCAN

## Introduction

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) est une méthode de clustering basée sur la densité des points de données. Contrairement à K-means et au clustering hiérarchique, DBSCAN peut identifier des clusters de formes irrégulières et détecter les outliers.

## Comment ça marche ?

1. **Sélection de deux paramètres** :
    - **Epsilon (ε)** : Le rayon de voisinage autour d'un point.
    - **MinPts (min_samples)** : Le nombre minimum de points requis dans un rayon ε pour qu'un point soit considéré comme un point noyau.
    
2. **Classification des points** :
    - **Points noyau** : Points ayant au moins MinPts voisins dans leur rayon ε.
    - **Points frontière** : Points ayant moins de MinPts voisins dans leur rayon ε mais étant voisins d'un point noyau.
    - **Points de bruit** : Points ne remplissant ni les conditions de points noyau ni celles de points frontière.

---

# 14. DBSCAN en Python

## Étape 1 : Importer les bibliothèques

 nécessaires

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
```

## Étape 2 : Charger et préparer le jeu de données

```python
# Charger le jeu de données
data = pd.read_csv('cereal.csv')

# Sélectionner les caractéristiques numériques pertinentes
numeric_data = data[['Calories', 'Protein', 'Sodium', 'Fiber']]
```

## Étape 3 : Normaliser les données

```python
# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
```

## Étape 4 : Appliquer DBSCAN

```python
# Appliquer DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(scaled_data)
labels = dbscan.labels_

# Ajouter les étiquettes de clusters au DataFrame
data['Cluster'] = labels
```

## Étape 5 : Visualiser les clusters

```python
import seaborn as sns

# Visualiser les clusters avec une carte de clusters
sns.clustermap(data[['Calories', 'Protein', 'Sodium', 'Fiber', 'Cluster']].sort_values(by='Cluster'),
               cmap='viridis', figsize=(10, 7), row_cluster=False, col_cluster=False)
plt.show()
```

---

# 15. DÉMO _ DBSCAN en Python

## Étape 1 : Importer les bibliothèques nécessaires

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
```

## Étape 2 : Générer des données synthétiques

```python
# Générer des données synthétiques
X, _ = make_blobs(n_samples=300, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4, random_state=0)
```

## Étape 3 : Appliquer DBSCAN

```python
# Appliquer DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10)
dbscan.fit(X)
labels = dbscan.labels_

# Tracer les clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Clustering avec DBSCAN")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

---

# 16. Assignment _ DBSCAN

## Objectif

Votre mission est d'appliquer l'algorithme DBSCAN sur un nouveau jeu de données, de déterminer les meilleurs paramètres epsilon et min_samples, et d'interpréter les résultats.

## Étapes

1. **Charger le jeu de données** : Utilisez le fichier `new_data.csv`.
2. **Préparer les données** : Normalisez les caractéristiques numériques.
3. **Appliquer DBSCAN** : Choisissez les meilleurs paramètres epsilon et min_samples.
4. **Interpréter les résultats** : Analysez les clusters formés et les outliers détectés.

## Critères d'évaluation

- Préparation adéquate des données.
- Sélection correcte des paramètres.
- Interprétation claire et précise des résultats.

---

# 17. Solution _ DBSCAN

## Étape 1 : Charger le jeu de données

```python
import pandas as pd

# Charger le jeu de données
data = pd.read_csv('new_data.csv')
```

## Étape 2 : Préparer les données

```python
from sklearn.preprocessing import StandardScaler

# Sélectionner les caractéristiques numériques
numeric_data = data.select_dtypes(include=[np.number])

# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
```

## Étape 3 : Appliquer DBSCAN et choisir les meilleurs paramètres

```python
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Fonction pour tester plusieurs combinaisons de paramètres
def tune_dbscan(data, eps_values, min_samples_values):
    results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(data)
            labels = dbscan.labels_
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters > 1:
                score = silhouette_score(data, labels)
            else:
                score = -1
            
            results.append((eps, min_samples, n_clusters, n_noise, score))
    
    results_df = pd.DataFrame(results, columns=['eps', 'min_samples', 'n_clusters', 'n_noise', 'silhouette_score'])
    return results_df

# Définir les plages de valeurs pour epsilon et min_samples
eps_values = np.arange(0.1, 2.1, 0.1)
min_samples_values = range(2, 11)

# Appliquer la fonction aux données normalisées
results = tune_dbscan(scaled_data, eps_values, min_samples_values)

# Trouver les meilleures combinaisons de paramètres
best_params = results.sort_values(by='silhouette_score', ascending=False).iloc[0]
print(f"Meilleure combinaison: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
```

## Étape 4 : Appliquer DBSCAN avec les meilleurs paramètres

```python
# Appliquer DBSCAN avec les meilleurs paramètres
dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
dbscan.fit(scaled_data)
labels = dbscan.labels_

# Ajouter les étiquettes de clusters au DataFrame
data['Cluster'] = labels
```

## Étape 5 : Interpréter les résultats

```python
# Analyser les clusters formés
print(data.groupby('Cluster').mean())

# Visualiser les clusters avec une carte de clusters
import seaborn as sns

sns.clustermap(data[['Feature1', 'Feature2', 'Cluster']].sort_values(by='Cluster'),
               cmap='viridis', figsize=(10, 7), row_cluster=False, col_cluster=False)
plt.show()

# Afficher le nombre de points dans chaque cluster
print(data['Cluster'].value_counts())
```

---

# 18. Score de silhouette

## Introduction

Le score de silhouette est une métrique utilisée pour évaluer la qualité des clusters créés par un modèle de clustering. Il mesure dans quelle mesure les points de données sont correctement assignés à leurs propres clusters par rapport à d'autres clusters.

## Calcul

Le score de silhouette pour un point de données est défini comme suit :
- **a** : La distance moyenne entre ce point et tous les autres points de son cluster.
- **b** : La distance moyenne entre ce point et tous les points du cluster le plus proche auquel il n'appartient pas.

Le score de silhouette s(i) pour un point i est donné par :

$$
s(i) = \frac{b - a}{\max(a, b)} 
$$

---

# 19. DÉMO _ Score de silhouette en Python

## Étape 1 : Importer les bibliothèques nécessaires

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```

## Étape 2 : Charger et préparer le jeu de données

```python
# Charger le jeu de données
data = pd.read_csv('cereal.csv')

# Sélectionner les caractéristiques numériques pertinentes
numeric_data = data[['Calories', 'Protein', 'Sodium', 'Fiber']]
```

## Étape 3 : Normaliser les données

```python
# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
```

## Étape 4 : Appliquer K-means et calculer le score de silhouette

```python
# Appliquer K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_

# Calculer le score de silhouette
score = silhouette_score(scaled_data, labels)
print(f"Score de Silhouette pour K-means : {score}")
```

## Étape 5 : Appliquer DBSCAN et calculer le score de silhouette

```python
# Appliquer DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(scaled_data)
labels = dbscan.labels_

# Calculer le score de silhouette
score = silhouette_score(scaled_data, labels)
print(f"Score de Silhouette pour DBSCAN : {score}")
```

---

# 20. Assignment _ Score de silhouette

## Objectif

Votre mission est de calculer le score de silhouette pour différents modèles de clustering (K-means et DBSCAN) sur un nouveau jeu de données et d'interpréter les résultats.

## Étapes

1. **Charger le jeu de données** : Utilisez le fichier `new_data.csv`.
2. **Préparer les données** : Normalisez les caractéristiques numériques.
3. **Appliquer K-means et DBSCAN** : Choisissez les meilleurs paramètres.
4. **Calculer le score de silhouette** : Comparez les scores pour les deux modèles.
5. **Interpréter les résultats** : Analysez les scores obtenus.

## Critères d'évaluation

- Préparation adéquate des données.
- Sélection correcte des paramètres.
- Calcul et comparaison précis des scores de silhouette.


- Interprétation claire et précise des résultats.

---

# 21. Solution _ Score de silhouette

## Étape 1 : Charger le jeu de données

```python
import pandas as pd

# Charger le jeu de données
data = pd.read_csv('new_data.csv')
```

## Étape 2 : Préparer les données

```python
from sklearn.preprocessing import StandardScaler

# Sélectionner les caractéristiques numériques
numeric_data = data.select_dtypes(include=[np.number])

# Normaliser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
```

## Étape 3 : Appliquer K-means et calculer le score de silhouette

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Appliquer K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)
labels = kmeans.labels_

# Calculer le score de silhouette
score = silhouette_score(scaled_data, labels)
print(f"Score de Silhouette pour K-means : {score}")
```

## Étape 4 : Appliquer DBSCAN et calculer le score de silhouette

```python
from sklearn.cluster import DBSCAN

# Appliquer DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(scaled_data)
labels = dbscan.labels_

# Calculer le score de silhouette
score = silhouette_score(scaled_data, labels)
print(f"Score de Silhouette pour DBSCAN : {score}")
```

## Étape 5 : Comparer et interpréter les résultats

```python
# Comparer les scores de silhouette
kmeans_score = silhouette_score(scaled_data, kmeans.labels_)
dbscan_score = silhouette_score(scaled_data, dbscan.labels_)

print(f"Score de Silhouette pour K-means : {kmeans_score}")
print(f"Score de Silhouette pour DBSCAN : {dbscan_score}")

# Interprétation des résultats
if kmeans_score > dbscan_score:
    print("K-means a produit de meilleurs clusters que DBSCAN.")
else:
    print("DBSCAN a produit de meilleurs clusters que K-means.")
```

---

# 22. Comparaison des algorithmes de clustering

## Introduction

Dans cette section, nous allons comparer différents algorithmes de clustering que nous avons appliqués jusqu'à présent : K-means, le clustering hiérarchique (agglomératif), et DBSCAN. Chaque méthode a ses propres avantages et inconvénients, et l'objectif est de comprendre comment ces algorithmes se comportent sur différents ensembles de données.

## Avantages et inconvénients

| Algorithme                     | Avantages                                                                                         | Inconvénients                                                                               |
|--------------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| **K-means**                    | Simple à comprendre et à implémenter, rapide pour des grands ensembles de données.                | Doit spécifier le nombre de clusters à l'avance, sensible aux valeurs aberrantes et au bruit. |
| **Clustering Hiérarchique**    | Pas besoin de spécifier le nombre de clusters à l'avance, génère un dendrogramme pour la visualisation. | Plus lent et inefficace pour des grands ensembles de données, sensible au bruit.            |
| **DBSCAN**                     | Identifie des clusters de formes arbitraires, capable de gérer les valeurs aberrantes et le bruit. | Les résultats dépendent fortement des paramètres epsilon et min_samples.                    |

## Comparaison visuelle

Pour montrer visuellement comment ces modèles se comparent côte à côte, nous allons utiliser des visualisations pour comparer nos modèles sur différents jeux de données.

### Clusters sphériques

- **K-means** : Excellent pour des clusters sphériques bien séparés.
- **Clustering Hiérarchique** : Identifie correctement les clusters.
- **DBSCAN** : Identifie correctement les clusters.

### Clusters en forme de longues chaînes

- **K-means** : Essaye de trouver des clusters sphériques même s'ils n'existent pas.
- **Clustering Hiérarchique** : Fait un meilleur travail en trouvant des clusters.
- **DBSCAN** : Identifie correctement les clusters.

### Clusters en forme de cercle

- **K-means** : Essaye de trouver des clusters sphériques, ce qui n'est pas adapté ici.
- **Clustering Hiérarchique** : Identifie correctement les clusters.
- **DBSCAN** : Le plus performant, identifie les clusters irréguliers et les points de bruit.

### Clusters de formes aléatoires

- **K-means** : Essaye de trouver des clusters sphériques, ce qui n'est pas adapté ici.
- **Clustering Hiérarchique** : Identifie correctement les clusters.
- **DBSCAN** : Le plus performant, identifie les clusters irréguliers et les points de bruit.

### Données aléatoires


- **K-means** : Essaye de trouver des clusters là où il n'y en a pas.
- **Clustering Hiérarchique** : Identifie un cluster unique dans une zone légèrement différente.
- **DBSCAN** : Reconnaît que tout est du bruit.

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/bef6719e-871e-413a-a1ba-6207d2601bd9)
![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/0289bba2-232e-4aa8-8731-d4be27314df3)
![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/677dc78c-6cd3-4b6f-a73f-d932d5cb7c06)
![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/444d5636-c915-400b-95f7-44881079fcbc)




```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs, make_moons, make_circles

# Comparaison visuelle
# Pour montrer visuellement comment ces modèles se comparent côte à côte, nous allons utiliser des visualisations pour comparer nos modèles sur différents jeux de données.

# Clusters sphériques
# Clustering sphériques
# K-means : Excellent pour des clusters sphériques bien séparés.
# Clustering Hiérarchique : Identifie correctement les clusters.
# DBSCAN : Identifie correctement les clusters.

# Clusters en forme de longues chaînes
# Clustering longues chaînes
# K-means : Essaye de trouver des clusters sphériques même s'ils n'existent pas.
# Clustering Hiérarchique : Fait un meilleur travail en trouvant des clusters.
# DBSCAN : Identifie correctement les clusters.

# Clusters en forme de cercle
# Clustering cercles
# K-means : Essaye de trouver des clusters sphériques, ce qui n'est pas adapté ici.
# Clustering Hiérarchique : Identifie correctement les clusters.
# DBSCAN : Le plus performant, identifie les clusters irréguliers et les points de bruit.

# Clusters de formes aléatoires
# Clustering formes aléatoires
# K-means : Essaye de trouver des clusters sphériques, ce qui n'est pas adapté ici.
# Clustering Hiérarchique : Identifie correctement les clusters.
# DBSCAN : Le plus performant, identifie les clusters irréguliers et les points de bruit.

# Données aléatoires
# Clustering données aléatoires
# K-means : Essaye de trouver des clusters là où il n'y en a pas.
# Clustering Hiérarchique : Identifie un cluster unique dans une zone légèrement différente.
# DBSCAN : Reconnaît que tout est du bruit.

# Conclusion
# En utilisant ces différentes méthodes de clustering, nous avons pu voir comment chaque algorithme se comporte avec différents jeux de données. En comparant les scores de silhouette, nous pouvons déterminer quel algorithme a produit les clusters les plus distincts et les plus appropriés pour un jeu de données spécifique.

# Fonction pour tracer les clusters
def plot_clusters(X, labels, title, ax):
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    ax.set_title(title)

# Générer les données
spherical_data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
chain_data, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
circle_data, _ = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=0)
random_data = np.random.rand(300, 2)

# Configurer les modèles de clustering
kmeans = KMeans(n_clusters=4)
hierarchical = AgglomerativeClustering(n_clusters=4)
dbscan = DBSCAN(eps=0.3, min_samples=5)

datasets = [
    ('Clusters sphériques', spherical_data),
    ('Clusters en forme de longues chaînes', chain_data),
    ('Clusters en forme de cercle', circle_data),
    ('Données aléatoires', random_data)
]

fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 20))
for idx, (title, data) in enumerate(datasets):
    kmeans_labels = kmeans.fit_predict(data)
    hierarchical_labels = hierarchical.fit_predict(data)
    dbscan_labels = dbscan.fit_predict(data)
    
    plot_clusters(data, kmeans_labels, f'{title}\nK-means : Excellent pour des clusters sphériques bien séparés.', axes[idx, 0])
    plot_clusters(data, hierarchical_labels, f'{title}\nClustering Hiérarchique : Identifie correctement les clusters.', axes[idx, 1])
    plot_clusters(data, dbscan_labels, f'{title}\nDBSCAN : Identifie correctement les clusters.', axes[idx, 2])

plt.tight_layout()
plt.show()


## Conclusion

En utilisant ces différentes méthodes de clustering, nous avons pu voir comment chaque algorithme se comporte avec différents jeux de données. En comparant les scores de silhouette, nous pouvons déterminer quel algorithme a produit les clusters les plus distincts et les plus appropriés pour un jeu de données spécifique.
```
---

# 23. Étapes suivantes du clustering

## Introduction

Maintenant que nous avons parcouru plusieurs modèles de clustering en détail, nous allons récapituler et examiner les étapes suivantes pour approfondir vos connaissances en clustering.

## Avantages des algorithmes de clustering

| Algorithme                    | Avantages                                                                                         |
|-------------------------------|---------------------------------------------------------------------------------------------------|
| **K-means**                   | Clusters faciles à comprendre et à interpréter, évolue bien avec les grands ensembles de données. |
| **Clustering Hiérarchique**   | Pas besoin de pré-définir k à l'avance, peut travailler avec des ensembles de données complexes.  |
| **DBSCAN**                    | Identifie des clusters de formes arbitraires, gère les valeurs aberrantes et le bruit.           |

## Inconvénients des algorithmes de clustering

| Algorithme                    | Inconvénients                                                                               |
|-------------------------------|---------------------------------------------------------------------------------------------|
| **K-means**                   | Doit spécifier le nombre de clusters à l'avance, sensible aux valeurs aberrantes.           |
| **Clustering Hiérarchique**   | Ne s'adapte pas bien aux grands ensembles de données, sensible aux valeurs aberrantes.      |
| **DBSCAN**                    | Ne s'adapte pas bien aux grands ensembles de données, réglage des hyperparamètres difficile. |

## Quand utiliser chaque modèle

| Algorithme                    | Quand l'utiliser                                                                             |
|-------------------------------|---------------------------------------------------------------------------------------------|
| **K-means**                   | Premier choix pour commencer un projet de clustering, clusters interprétables.              |
| **Clustering Hiérarchique**   | Utilisé principalement pour la visualisation et l'exploration des relations entre les points. |
| **DBSCAN**                    | Utilisé pour les jeux de données avec des valeurs aberrantes et des clusters de formes irrégulières. |

## Conclusion

En résumé, aucun modèle de clustering n'est le meilleur tout le temps. Cela dépend vraiment de l'apparence de votre jeu de données. Pour K-means, il trouvera des clusters sphériques. Pour le clustering hiérarchique, il se base sur les calculs de distance. Pour DBSCAN, il se base sur la densité des points. En choisissant le bon algorithme pour vos données, vous pouvez obtenir des clusters plus significatifs et interprétables.

# Références : 
- https://r.qcbs.ca/workshop09/book-fr/regroupement.html
