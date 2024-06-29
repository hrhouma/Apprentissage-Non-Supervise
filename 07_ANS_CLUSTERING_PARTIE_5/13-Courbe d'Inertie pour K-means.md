### Partie 1: Courbe d'Inertie pour K-means

La courbe d'inertie est un outil graphique utilisé pour déterminer le nombre optimal de clusters dans l'algorithme K-means. 

- **Inertie** mesure la somme des distances au carré entre chaque point de données et le centre de son cluster.
- En traçant l'inertie en fonction du nombre de clusters, on peut observer comment l'inertie diminue à mesure que le nombre de clusters augmente.
- Le "coude" de la courbe indique le point optimal où ajouter plus de clusters n'améliore plus significativement l'inertie, suggérant ainsi le nombre de clusters à utiliser.

#### Exemple avec K-Means

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Génération de données
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, random_state=42)

# Calcul de l'inertie pour différents nombres de clusters
inertias = []
range_n_clusters = range(1, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Tracé de la courbe d'inertie
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, inertias, marker='o')
plt.title('Courbe d\'Inertie pour déterminer le nombre optimal de clusters')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.grid(True)
plt.show()
```

#### Interprétation de la Courbe d'Inertie

La courbe d'inertie aide à identifier le "coude", c'est-à-dire le point où l'inertie commence à diminuer moins rapidement avec l'augmentation du nombre de clusters. Ce point suggère le nombre optimal de clusters.

### Partie 2: Exemple de Clustering avec le Nombre Optimal de Clusters

Après avoir identifié le nombre optimal de clusters à l'aide de la courbe d'inertie, nous pouvons appliquer K-Means avec ce nombre de clusters.

#### Exemple avec K-Means et le Nombre Optimal de Clusters

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Génération de données
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, random_state=42)

# Nombre optimal de clusters (identifié à partir de la courbe d'inertie)
optimal_n_clusters = 4

# Clustering avec K-Means
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

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
plt.title(f'Visualisation des Clusters K-Means avec {optimal_n_clusters} clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
```

### Partie 3: Détection d'Anomalies

L'analyse de l'inertie peut également être utilisée pour identifier des anomalies, car des points éloignés des centres des clusters peuvent être considérés comme des anomalies.

#### Exemple avec K-Means et Détection d'Anomalies

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Génération de données avec une dispersion plus élevée
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=2, random_state=42)

# Clustering avec K-Means
optimal_n_clusters = 4
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# Visualisation des clusters et des anomalies
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
distances = np.min(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
threshold = np.percentile(distances, 95)  # Seuil pour identifier les anomalies
anomalies = X[distances > threshold]
plt.scatter(anomalies[:, 0], anomalies[:, 1], s=100, c='black', marker='o', edgecolor='red', label='Anomalies')

# Ajouter les annotations
plt.title(f'Clusters K-Means avec Détection d\'Anomalies\n(Nombre de clusters optimal : {optimal_n_clusters})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Affichage des anomalies
print(f"Nombre d'anomalies détectées : {len(anomalies)}")
```
