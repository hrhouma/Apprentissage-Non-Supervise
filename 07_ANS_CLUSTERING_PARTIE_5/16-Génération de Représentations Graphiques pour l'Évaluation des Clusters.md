# Représentations Graphiques pour l'Évaluation des Clusters

#### Ensemble 1: Bon Clustering

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.datasets import make_blobs

# Génération de données avec étiquettes réelles
X1, true_labels1 = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=0.5, random_state=42)

# Clustering avec K-Means
kmeans1 = KMeans(n_clusters=4, random_state=42)
pred_labels1 = kmeans1.fit_predict(X1)

# Calcul des métriques
silhouette_avg1 = silhouette_score(X1, pred_labels1)
dbi1 = davies_bouldin_score(X1, pred_labels1)
ari1 = adjusted_rand_score(true_labels1, pred_labels1)
nmi1 = normalized_mutual_info_score(true_labels1, pred_labels1)
centers1 = kmeans1.cluster_centers_
_, distances1 = pairwise_distances_argmin_min(X1, centers1)
cohesion1 = np.sum(distances1)
separation1 = np.sum([np.linalg.norm(centers1[i] - centers1[j]) for i in range(len(centers1)) for j in range(i+1, len(centers1))])

# Affichage des résultats
print("Ensemble 1: Bon Clustering")
print(f"Score de Silhouette : {silhouette_avg1:.2f}")
print(f"Indice de Davies-Bouldin : {dbi1:.2f}")
print(f"Indice de Rand Ajusté (ARI) : {ari1:.2f}")
print(f"Normalized Mutual Information (NMI) : {nmi1:.2f}")
print(f"Cohésion : {cohesion1:.2f}")
print(f"Séparation : {separation1:.2f}")

# Visualisation des clusters
plt.figure(figsize=(10, 6))
unique_labels = np.unique(pred_labels1)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    cluster_points = X1[pred_labels1 == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

# Ajouter les centres des clusters
plt.scatter(centers1[:, 0], centers1[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

# Ajouter les annotations
plt.title(f'Visualisation des Clusters K-Means (Bon Clustering)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
```

![Bon Clustering](https://via.placeholder.com/800x400.png?text=Ensemble+1%3A+Bon+Clustering)

#### Ensemble 2: Clustering Moyennement Bon

```python
# Génération de données avec étiquettes réelles
X2, true_labels2 = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.5, random_state=42)

# Clustering avec K-Means
kmeans2 = KMeans(n_clusters=4, random_state=42)
pred_labels2 = kmeans2.fit_predict(X2)

# Calcul des métriques
silhouette_avg2 = silhouette_score(X2, pred_labels2)
dbi2 = davies_bouldin_score(X2, pred_labels2)
ari2 = adjusted_rand_score(true_labels2, pred_labels2)
nmi2 = normalized_mutual_info_score(true_labels2, pred_labels2)
centers2 = kmeans2.cluster_centers_
_, distances2 = pairwise_distances_argmin_min(X2, centers2)
cohesion2 = np.sum(distances2)
separation2 = np.sum([np.linalg.norm(centers2[i] - centers2[j]) for i in range(len(centers2)) for j in range(i+1, len(centers2))])

# Affichage des résultats
print("Ensemble 2: Clustering Moyennement Bon")
print(f"Score de Silhouette : {silhouette_avg2:.2f}")
print(f"Indice de Davies-Bouldin : {dbi2:.2f}")
print(f"Indice de Rand Ajusté (ARI) : {ari2:.2f}")
print(f"Normalized Mutual Information (NMI) : {nmi2:.2f}")
print(f"Cohésion : {cohesion2:.2f}")
print(f"Séparation : {separation2:.2f}")

# Visualisation des clusters
plt.figure(figsize=(10, 6))
unique_labels = np.unique(pred_labels2)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    cluster_points = X2[pred_labels2 == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

# Ajouter les centres des clusters
plt.scatter(centers2[:, 0], centers2[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

# Ajouter les annotations
plt.title(f'Visualisation des Clusters K-Means (Clustering Moyennement Bon)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
```

![Clustering Moyennement Bon](https://via.placeholder.com/800x400.png?text=Ensemble+2%3A+Clustering+Moyennement+Bon)

#### Ensemble 3: Mauvais Clustering

```python
# Génération de données avec étiquettes réelles
X3, true_labels3 = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=3.0, random_state=42)

# Clustering avec K-Means
kmeans3 = KMeans(n_clusters=4, random_state=42)
pred_labels3 = kmeans3.fit_predict(X3)

# Calcul des métriques
silhouette_avg3 = silhouette_score(X3, pred_labels3)
dbi3 = davies_bouldin_score(X3, pred_labels3)
ari3 = adjusted_rand_score(true_labels3, pred_labels3)
nmi3 = normalized_mutual_info_score(true_labels3, pred_labels3)
centers3 = kmeans3.cluster_centers_
_, distances3 = pairwise_distances_argmin_min(X3, centers3)
cohesion3 = np.sum(distances3)
separation3 = np.sum([np.linalg.norm(centers3[i] - centers3[j]) for i in range(len(centers3)) for j in range(i+1, len(centers3))])

# Affichage des résultats
print("Ensemble 3: Mauvais Clustering")
print(f"Score de Silhouette : {silhouette_avg3:.2f}")
print(f"Indice de Davies-Bouldin : {dbi3:.2f}")
print(f"Indice de Rand Ajusté (ARI) : {ari3:.2f}")
print(f"Normalized Mutual Information (NMI) : {nmi3:.2f}")
print(f"Cohésion : {cohesion3:.2f}")
print(f"Séparation : {separation3:.2f}")

# Visualisation des clusters
plt.figure(figsize=(10, 6))
unique_labels = np.unique(pred_labels3)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    cluster_points = X3[pred_labels3 == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

# Ajouter les centres des clusters
plt.scatter(centers3[:, 0], centers3[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

# Ajouter les annotations
plt.title(f'Visualisation des Clusters K-Means (Mauvais Clustering)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
```

![Mauvais Clustering](https://via.placeholder.com/800x400.png?text=Ensemble+3%3A+Mauvais+Clustering)

### Questions 

1. **Ensemble 1: Bon Clustering**
   - Les clusters sont-ils bien formés, moyennement bien formés, ou mal formés ? Justifiez votre réponse en utilisant les valeurs des métriques suivantes :
     - Score de Silhouette : `0.65`
     - Indice de Davies-Bouldin : `0.75`
     - Indice de Rand Ajusté (ARI) : `0.85`
     - Normalized Mutual Information (NMI) : `0.90

`
     - Cohésion : `250.0`
     - Séparation : `1500.0`

2. **Ensemble 2: Clustering Moyennement Bon**
   - Les clusters sont-ils bien formés, moyennement bien formés, ou mal formés ? Justifiez votre réponse en utilisant les valeurs des métriques suivantes :
     - Score de Silhouette : `0.45`
     - Indice de Davies-Bouldin : `1.25`
     - Indice de Rand Ajusté (ARI) : `0.60`
     - Normalized Mutual Information (NMI) : `0.70`
     - Cohésion : `400.0`
     - Séparation : `1000.0`

3. **Ensemble 3: Mauvais Clustering**
   - Les clusters sont-ils bien formés, moyennement bien formés, ou mal formés ? Justifiez votre réponse en utilisant les valeurs des métriques suivantes :
     - Score de Silhouette : `0.30`
     - Indice de Davies-Bouldin : `2.00`
     - Indice de Rand Ajusté (ARI) : `0.30`
     - Normalized Mutual Information (NMI) : `0.50`
     - Cohésion : `600.0`
     - Séparation : `800.0`
