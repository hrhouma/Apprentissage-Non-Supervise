### Partie 1: Le Score de Silhouette

Le score de silhouette est une métrique utilisée pour évaluer la qualité des clusters formés par un algorithme de clustering. Il est basé sur deux critères :
1. **Cohésion** : La proximité d'un point de données avec les autres points dans le même cluster.
2. **Séparation** : La distance d'un point de données avec les points dans les clusters voisins.

Un score de silhouette varie de -1 à 1 :
- **Proche de 1** : Les points sont bien groupés et bien séparés des autres clusters.
- **Proche de 0** : Les points sont à la frontière entre deux clusters.
- **Négatif** : Les points sont probablement mal assignés à un cluster.

#### Exemple avec K-Means

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Génération de données
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, random_state=42)

# Clustering avec k-means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Calcul de la largeur de silhouette
silhouette_avg = silhouette_score(X, labels)
print(f"La largeur de silhouette moyenne est : {silhouette_avg}")

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
plt.title(f'Visualisation des Clusters K-Means (Silhouette moyenne : {silhouette_avg:.2f})')
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
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

def plot_kmeans(X, n_clusters, title_suffix=""):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)

    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    for label in unique_labels:
        cluster_points = X[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

    plt.title(f'Clusters K-Means (k={n_clusters}, Silhouette moyenne : {silhouette_avg:.2f}) {title_suffix}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f'La largeur de silhouette moyenne pour k={n_clusters} est : {silhouette_avg:.2f}')

# Génération de données avec une grande largeur de cluster
X_large, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=0.3, random_state=42)

# Génération de données avec une petite largeur de cluster
X_small, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.5, random_state=42)

# Génération de données avec une très grande largeur de cluster
X_extremely_dispersed, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=5.5, random_state=42)

# Visualisation avec un bon score de silhouette
plot_kmeans(X_large, n_clusters=4, title_suffix="(grande largeur de cluster)")

# Visualisation avec un mauvais score de silhouette
plot_kmeans(X_small, n_clusters=4, title_suffix="(petite largeur de cluster)")

# Visualisation avec une silhouette très petite
plot_kmeans(X_extremely_dispersed, n_clusters=4, title_suffix="(très extrêmement dispersé)")
```

### Partie 2: Détection d'Anomalies

Les anomalies peuvent être détectées en utilisant des métriques telles que la distance au centre du cluster ou le score de silhouette.

#### Exemple avec K-Means et Détection d'Anomalies

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs

# Génération de données avec une dispersion plus élevée
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=2, random_state=42)

# Clustering avec k-means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Calcul de la largeur de silhouette
silhouette_avg = silhouette_score(X, labels)
silhouette_values = silhouette_samples(X, labels)

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

# Ajouter les anomalies (points avec silhouette négative)
anomalies = X[silhouette_values < 0]
plt.scatter(anomalies[:, 0], anomalies[:, 1], s=100, c='black', marker='o', edgecolor='red', label='Anomalies')

# Ajouter les annotations
plt.title(f'Clusters K-Means avec Détection d\'Anomalies (Silhouette moyenne : {silhouette_avg:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Affichage de la largeur de silhouette moyenne
print(f'La largeur de silhouette moyenne est : {silhouette_avg:.2f}')
print(f'Nombre d\'anomalies détectées : {len(anomalies)}')
```

Dans ce graphique, les points marqués comme "Anomalies" sont identifiés par des cercles rouges. Une anomalie, dans le contexte du clustering, est un point de données qui se situe loin des centres des clusters et des autres points de données du même cluster.

#### Exemple avec DBSCAN et Détection d'Anomalies

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs

# Génération de données avec une dispersion plus élevée
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=2, random_state=42)

# Utiliser DBSCAN pour le clustering
dbscan = DBSCAN(eps=0.9, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

# Calcul de la largeur de silhouette pour DBSCAN
silhouette_avg_dbscan = silhouette_score(X, labels_dbscan)
silhouette_values_dbscan = silhouette_samples(X, labels_dbscan)

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
plt.title(f'Clusters DBSCAN avec Détection d\'Anomalies (Silhouette moyenne : {silhouette_avg_dbscan:.2f})')
plt.xlabel('Feature 1')


plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Affichage de la largeur de silhouette moyenne
print(f'La largeur de silhouette moyenne pour DBSCAN est : {silhouette_avg_dbscan:.2f}')
print(f'Nombre d\'anomalies détectées : {len(X[labels_dbscan == -1])}')
```
