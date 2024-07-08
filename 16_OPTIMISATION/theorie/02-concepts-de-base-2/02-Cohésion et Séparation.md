### Partie 1: Cohésion et Séparation

La cohésion et la séparation sont deux critères essentiels pour évaluer la qualité des clusters :

- **Cohésion (Intra-cluster distance)** : Mesure à quel point les points de données dans un même cluster sont proches les uns des autres. Une bonne cohésion signifie que les membres d'un cluster sont très similaires entre eux.
- **Séparation (Inter-cluster distance)** : Mesure la distance entre les différents clusters. Une bonne séparation signifie que les clusters sont bien distincts les uns des autres.

Une bonne cohésion et une bonne séparation indiquent que les clusters sont bien formés, ce qui est crucial pour des analyses significatives et interprétables.

#### Exemple avec K-Means

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.datasets import make_blobs

# Génération de données
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, random_state=42)

# Clustering avec k-means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Calcul de la cohésion (intra-cluster distance)
centers = kmeans.cluster_centers_
_, distances = pairwise_distances_argmin_min(X, centers)
cohesion = np.sum(distances)
print(f"La cohésion (intra-cluster distance) est : {cohesion}")

# Calcul de la séparation (inter-cluster distance)
separation = np.sum([np.linalg.norm(centers[i] - centers[j]) for i in range(len(centers)) for j in range(i+1, len(centers))])
print(f"La séparation (inter-cluster distance) est : {separation}")

# Visualisation des clusters
plt.figure(figsize=(10, 6))
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    cluster_points = X[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

# Ajouter les centres des clusters
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

# Ajouter les annotations
plt.title(f'Visualisation des Clusters K-Means\n(Cohésion : {cohesion:.2f}, Séparation : {separation:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
```

#### Analyse Visuelle avec Différents Niveaux de Dispersion

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.datasets import make_blobs

def plot_kmeans_cohesion_separation(X, n_clusters, title_suffix=""):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    _, distances = pairwise_distances_argmin_min(X, centers)
    
    cohesion = np.sum(distances)
    separation = np.sum([np.linalg.norm(centers[i] - centers[j]) for i in range(len(centers)) for j in range(i+1, len(centers))])

    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    for label in unique_labels:
        cluster_points = X[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

    plt.title(f'Clusters K-Means (k={n_clusters})\nCohésion : {cohesion:.2f}, Séparation : {separation:.2f} {title_suffix}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f'Cohésion pour k={n_clusters} est : {cohesion:.2f}')
    print(f'Séparation pour k={n_clusters} est : {separation:.2f}')

# Génération de données avec une grande largeur de cluster
X_large, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=0.3, random_state=42)

# Génération de données avec une petite largeur de cluster
X_small, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.5, random_state=42)

# Génération de données avec une très grande largeur de cluster
X_extremely_dispersed, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=5.5, random_state=42)

# Visualisation avec une bonne cohésion et une bonne séparation
plot_kmeans_cohesion_separation(X_large, n_clusters=4, title_suffix="(grande largeur de cluster)")

# Visualisation avec une mauvaise cohésion et une mauvaise séparation
plot_kmeans_cohesion_separation(X_small, n_clusters=4, title_suffix="(petite largeur de cluster)")

# Visualisation avec une très mauvaise cohésion et une mauvaise séparation
plot_kmeans_cohesion_separation(X_extremely_dispersed, n_clusters=4, title_suffix="(très extrêmement dispersé)")
```

### Partie 2: Détection d'Anomalies

La détection des anomalies peut également être analysée en étudiant les points de données qui sont loin des centres de clusters, ce qui indique une mauvaise cohésion.

#### Exemple avec K-Means et Détection d'Anomalies

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.datasets import make_blobs

# Génération de données avec une dispersion plus élevée
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=2, random_state=42)

# Clustering avec k-means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Calcul de la cohésion (intra-cluster distance)
centers = kmeans.cluster_centers_
_, distances = pairwise_distances_argmin_min(X, centers)
cohesion = np.sum(distances)

# Calcul de la séparation (inter-cluster distance)
separation = np.sum([np.linalg.norm(centers[i] - centers[j]) for i in range(len(centers)) for j in range(i+1, len(centers))])

# Visualisation des clusters
plt.figure(figsize=(10, 6))
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    cluster_points = X[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

# Ajouter les centres des clusters
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

# Ajouter les anomalies (points éloignés des centres de clusters)
anomalies = X[distances > np.percentile(distances, 90)] # Utilisation d'un seuil pour identifier les anomalies
plt.scatter(anomalies[:, 0], anomalies[:, 1], s=100, c='black', marker='o', edgecolor='red', label='Anomalies')

# Ajouter les annotations
plt.title(f'Clusters K-Means avec Détection d\'Anomalies\n(Cohésion : {cohesion:.2f}, Séparation : {separation:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Affichage de la cohésion et de la séparation
print(f'Cohésion : {cohesion:.2f}')
print(f'Séparation : {separation:.2f}')
print(f'Nombre d\'anomalies détectées : {len(anomalies)}')
```

#### Exemple avec DBSCAN et Détection d'Anomalies

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.datasets import make_blobs

# Génération de données avec une dispersion plus élevée
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=2, random_state=42)

# Utiliser DBSCAN pour le clustering
dbscan = DBSCAN(eps=0.9, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

# Calcul de la cohésion (intra-cluster distance) pour DBSCAN
unique_labels

_dbscan = np.unique(labels_dbscan)
cohesion_dbscan = sum([np.sum(pairwise_distances_argmin_min(X[labels_dbscan == label], [X[labels_dbscan == label].mean(axis=0)])[1]) for label in unique_labels_dbscan if label != -1])

# Calcul de la séparation (inter-cluster distance) pour DBSCAN
centers_dbscan = np.array([X[labels_dbscan == label].mean(axis=0) for label in unique_labels_dbscan if label != -1])
separation_dbscan = np.sum([np.linalg.norm(centers_dbscan[i] - centers_dbscan[j]) for i in range(len(centers_dbscan)) for j in range(i+1, len(centers_dbscan))])

# Visualisation des clusters avec DBSCAN
plt.figure(figsize=(10, 6))
colors = plt.cm.get_cmap("tab10", len(unique_labels_dbscan))

for label in unique_labels_dbscan:
    if label == -1:
        # Anomalies (bruit) dans DBSCAN
        cluster_points = X[labels_dbscan == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=100, c='black', marker='o', edgecolor='red', label='Anomalies')
    else:
        cluster_points = X[labels_dbscan == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

# Ajouter les annotations
plt.title(f'Clusters DBSCAN avec Détection d\'Anomalies\n(Cohésion : {cohesion_dbscan:.2f}, Séparation : {separation_dbscan:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Affichage de la cohésion et de la séparation
print(f'Cohésion pour DBSCAN : {cohesion_dbscan:.2f}')
print(f'Séparation pour DBSCAN : {separation_dbscan:.2f}')
print(f'Nombre d\'anomalies détectées : {len(X[labels_dbscan == -1])}')
```
