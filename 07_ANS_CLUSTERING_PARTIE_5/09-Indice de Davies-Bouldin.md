### Partie 1: L'Indice de Davies-Bouldin

L'indice de Davies-Bouldin (DBI) évalue la qualité du clustering en comparant la moyenne des dispersions intra-cluster à la séparation inter-cluster. 

- **Dispersion intra-cluster** : Mesure la distance moyenne entre les points de données et le centre de leur cluster respectif.
- **Séparation inter-cluster** : Mesure la distance entre les centres de clusters.

Un indice de Davies-Bouldin faible indique que les clusters sont compacts et bien séparés les uns des autres, suggérant un bon clustering. Cette mesure aide à identifier si les clusters sont bien définis.

#### Exemple avec K-Means

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.datasets import make_blobs

# Génération de données
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, random_state=42)

# Clustering avec k-means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Calcul de l'indice de Davies-Bouldin
dbi = davies_bouldin_score(X, labels)
print(f"L'indice de Davies-Bouldin est : {dbi}")

# Visualisation des clusters
plt.figure(figsize=(10, 6))
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    cluster_points = X[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

# Ajouter les centres des clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

# Ajouter les annotations
plt.title(f'Visualisation des Clusters K-Means (Davies-Bouldin Index : {dbi:.2f})')
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
from sklearn.metrics import davies_bouldin_score
from sklearn.datasets import make_blobs

def plot_kmeans_dbi(X, n_clusters, title_suffix=""):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    dbi = davies_bouldin_score(X, labels)

    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    for label in unique_labels:
        cluster_points = X[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

    plt.title(f'Clusters K-Means (k={n_clusters}, Davies-Bouldin Index : {dbi:.2f}) {title_suffix}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"L'indice de Davies-Bouldin pour k={n_clusters} est : {dbi:.2f}")

# Génération de données avec une grande largeur de cluster
X_large, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=0.3, random_state=42)

# Génération de données avec une petite largeur de cluster
X_small, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.5, random_state=42)

# Génération de données avec une très grande largeur de cluster
X_extremely_dispersed, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=5.5, random_state=42)

# Visualisation avec un bon indice de Davies-Bouldin
plot_kmeans_dbi(X_large, n_clusters=4, title_suffix="(grande largeur de cluster)")

# Visualisation avec un mauvais indice de Davies-Bouldin
plot_kmeans_dbi(X_small, n_clusters=4, title_suffix="(petite largeur de cluster)")

# Visualisation avec un très mauvais indice de Davies-Bouldin
plot_kmeans_dbi(X_extremely_dispersed, n_clusters=4, title_suffix="(très extrêmement dispersé)")
```

### Partie 2: Détection d'Anomalies

Les anomalies peuvent également être détectées en utilisant l'indice de Davies-Bouldin pour identifier les points de données mal clusterisés.

#### Exemple avec K-Means et Détection d'Anomalies

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.datasets import make_blobs

# Génération de données avec une dispersion plus élevée
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=2, random_state=42)

# Clustering avec k-means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Calcul de l'indice de Davies-Bouldin
dbi = davies_bouldin_score(X, labels)

# Visualisation des clusters
plt.figure(figsize=(10, 6))
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    cluster_points = X[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

# Ajouter les centres des clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

# Ajouter les anomalies (points éloignés des centres de clusters)
anomalies = X[labels == -1]
plt.scatter(anomalies[:, 0], anomalies[:, 1], s=100, c='black', marker='o', edgecolor='red', label='Anomalies')

# Ajouter les annotations
plt.title(f'Clusters K-Means avec Détection d\'Anomalies (Davies-Bouldin Index : {dbi:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Affichage de l'indice de Davies-Bouldin
print(f"L'indice de Davies-Bouldin est : {dbi:.2f}")
print(f"Nombre d'anomalies détectées : {len(anomalies)}")
```

#### Exemple avec DBSCAN et Détection d'Anomalies

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score
from sklearn.datasets import make_blobs

# Génération de données avec une dispersion plus élevée
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=2, random_state=42)

# Utiliser DBSCAN pour le clustering
dbscan = DBSCAN(eps=0.9, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

# Calcul de l'indice de Davies-Bouldin pour DBSCAN
dbi_dbscan = davies_bouldin_score(X, labels_dbscan)

# Visualisation des clusters avec DBSCAN
plt.figure(figsize=(10, 6))
unique_labels_dbscan = np.unique(labels_dbscan)
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
plt.title(f'Clusters DBSCAN avec Détection d\'Anomalies (Davies-Bouldin Index : {dbi_dbscan:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Affichage de l'indice de Davies-Bouldin
print(f"L'indice de Davies-Bouldin pour DBSCAN est : {dbi_dbscan:.2f}")
print(f"Nombre d'anomalies détectées : {len(X[labels_db

scan == -1])}")
```
