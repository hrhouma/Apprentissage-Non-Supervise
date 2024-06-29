### Partie 1: Normalized Mutual Information (NMI)

La Normalized Mutual Information (NMI) est une mesure utilisée pour comparer deux partitions d'un ensemble de données en termes d'information partagée. Elle évalue combien d'information sur l'une des partitions est contenue dans l'autre, et vice versa.

- Une **NMI élevée** indique que les deux partitions partagent beaucoup d'information, ce qui signifie qu'elles sont similaires.
- **NMI** est particulièrement utile pour évaluer des algorithmes de clustering où les partitions peuvent avoir des tailles différentes et où l'information partagée doit être normalisée pour fournir une comparaison équitable.

#### Exemple avec K-Means

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.datasets import make_blobs

# Génération de données avec étiquettes réelles
X, true_labels = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, random_state=42)

# Clustering avec k-means
kmeans = KMeans(n_clusters=4, random_state=42)
pred_labels = kmeans.fit_predict(X)

# Calcul de la Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(true_labels, pred_labels)
print(f"La Normalized Mutual Information (NMI) est : {nmi}")

# Visualisation des clusters
plt.figure(figsize=(10, 6))
unique_labels = np.unique(pred_labels)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    cluster_points = X[pred_labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

# Ajouter les centres des clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

# Ajouter les annotations
plt.title(f'Visualisation des Clusters K-Means (NMI : {nmi:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
```

#### Analyse Visuelle avec Différents Scénarios

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.datasets import make_blobs

def plot_kmeans_nmi(X, true_labels, n_clusters, title_suffix=""):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pred_labels = kmeans.fit_predict(X)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(pred_labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    for label in unique_labels:
        cluster_points = X[pred_labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

    plt.title(f'Clusters K-Means (k={n_clusters}, NMI : {nmi:.2f}) {title_suffix}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f'NMI pour k={n_clusters} est : {nmi:.2f}')

# Génération de données avec étiquettes réelles
X, true_labels = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, random_state=42)

# Scénario 1: Bon clustering avec k=4
plot_kmeans_nmi(X, true_labels, n_clusters=4, title_suffix="(bon clustering)")

# Scénario 2: Mauvais clustering avec k=3
plot_kmeans_nmi(X, true_labels, n_clusters=3, title_suffix="(mauvais clustering)")

# Scénario 3: Très mauvais clustering avec k=5
plot_kmeans_nmi(X, true_labels, n_clusters=5, title_suffix="(très mauvais clustering)")
```

### Partie 2: Détection d'Anomalies

Bien que la NMI ne soit pas directement utilisée pour la détection d'anomalies, elle peut aider à évaluer la qualité du clustering, ce qui est crucial pour la détection d'anomalies.

#### Exemple avec K-Means et Détection d'Anomalies

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.datasets import make_blobs

# Génération de données avec étiquettes réelles et une dispersion plus élevée
X, true_labels = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=2, random_state=42)

# Clustering avec k-means
kmeans = KMeans(n_clusters=4, random_state=42)
pred_labels = kmeans.fit_predict(X)

# Calcul de la Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(true_labels, pred_labels)

# Visualisation des clusters
plt.figure(figsize=(10, 6))
unique_labels = np.unique(pred_labels)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    cluster_points = X[pred_labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

# Ajouter les centres des clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

# Ajouter les anomalies (points éloignés des centres de clusters)
anomalies = X[pred_labels == -1]
plt.scatter(anomalies[:, 0], anomalies[:, 1], s=100, c='black', marker='o', edgecolor='red', label='Anomalies')

# Ajouter les annotations
plt.title(f'Clusters K-Means avec Détection d\'Anomalies (NMI : {nmi:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Affichage de la Normalized Mutual Information (NMI)
print(f"La Normalized Mutual Information (NMI) est : {nmi:.2f}")
print(f"Nombre d'anomalies détectées : {len(anomalies)}")
```

#### Exemple avec DBSCAN et Détection d'Anomalies

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score
from sklearn.datasets import make_blobs

# Génération de données avec étiquettes réelles et une dispersion plus élevée
X, true_labels = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=2, random_state=42)

# Utiliser DBSCAN pour le clustering
dbscan = DBSCAN(eps=0.9, min_samples=5)
pred_labels_dbscan = dbscan.fit_predict(X)

# Calcul de la Normalized Mutual Information (NMI) pour DBSCAN
nmi_dbscan = normalized_mutual_info_score(true_labels, pred_labels_dbscan)

# Visualisation des clusters avec DBSCAN
plt.figure(figsize=(10, 6))
unique_labels_dbscan = np.unique(pred_labels_dbscan)
colors = plt.cm.get_cmap("tab10", len(unique_labels_dbscan))

for label in unique_labels_dbscan:
    if label == -1:
        # Anomalies (bruit) dans DBSCAN
        cluster_points = X[pred_labels_dbscan == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=100, c='black', marker='o', edgecolor='red', label='Anomalies')
    else:
        cluster_points = X[pred_labels_dbscan == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

# Ajouter les annotations
plt.title(f'Clusters DBSCAN avec Détection d\'Anomalies (NMI : {nmi_dbscan:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Affichage de la Normalized Mutual Information (NMI)
print(f"La Normalized Mutual Information (NMI) pour DBSCAN est : {nmi_dbscan:.2f}")
print(f"Nombre d'anomalies détectées : {len(X[pred_labels_dbscan == -1])}")
```
