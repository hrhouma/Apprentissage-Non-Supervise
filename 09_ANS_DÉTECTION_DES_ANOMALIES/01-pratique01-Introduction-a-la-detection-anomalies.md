# Lien des travaux : 
- https://drive.google.com/drive/folders/1eYlsTNAAoy53DmvL7Ymb07bOi039Ynn4?usp=sharing
---
# Détection d'Anomalies avec K-Means et DBSCAN

Ce document présente trois méthodes différentes pour détecter des anomalies dans des ensembles de données en utilisant les algorithmes K-Means et DBSCAN. Chaque méthode a ses propres caractéristiques et avantages en fonction du type de données et de la nature des anomalies recherchées.

### Approches

# 1. K-Means avec Largeur de Silhouette

##### Algorithme
K-Means est un algorithme de clustering qui partitionne un ensemble de données en `k` clusters. L'algorithme minimise la variance intra-cluster et attribue chaque point au cluster dont le centre est le plus proche. Voici comment ça fonctionne :
1. Choisir le nombre de clusters `k`.
2. Initialiser aléatoirement les centres de clusters (aussi appelés centroides).
3. Assigner chaque point au cluster dont le centre est le plus proche.
4. Recalculer les centres des clusters comme la moyenne des points assignés à chaque cluster.
5. Répéter les étapes 3 et 4 jusqu'à ce que les centres des clusters ne changent plus (ou changent très peu).

##### Détection d'Anomalies
Les anomalies sont détectées en utilisant la largeur de silhouette, qui mesure la cohésion et la séparation des clusters :
- **Largeur de silhouette** : C'est une mesure de la qualité du clustering pour chaque point. Elle varie de -1 à 1. Une valeur proche de 1 signifie que le point est bien assigné à son cluster, une valeur proche de 0 signifie qu'il est sur ou très près de la frontière entre deux clusters, et une valeur négative signifie qu'il pourrait être mieux assigné à un autre cluster.
- **Anomalie** : Un point est considéré comme une anomalie s'il a une valeur de silhouette négative, car cela indique qu'il est mal assigné à son cluster actuel.

Dans le code, cela se fait par :
```python
silhouette_values = silhouette_samples(X, labels)
anomalies = X[silhouette_values < 0]
```
- `silhouette_samples(X, labels)` : Cette fonction calcule la valeur de silhouette pour chaque point de données. Elle prend en entrée les données `X` et les labels des clusters `labels`.
- `anomalies = X[silhouette_values < 0]` : Ici, nous filtrons les points dont la valeur de silhouette est négative et les considérons comme des anomalies.

-----
# 2. DBSCAN

##### Algorithme
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) est un algorithme de clustering basé sur la densité. Il forme des clusters en regroupant des points proches les uns des autres, en fonction de deux paramètres :
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

Dans le code, cela se fait par :
```python
labels_dbscan = dbscan.fit_predict(X)
anomalies = X[labels_dbscan == -1]
```
- `labels_dbscan = dbscan.fit_predict(X)` : Cette fonction exécute l'algorithme DBSCAN sur les données `X` et retourne les labels des clusters. Les points de bruit sont marqués avec le label `-1`.
- `anomalies = X[labels_dbscan == -1]` : Ici, nous filtrons les points dont le label est `-1` et les considérons comme des anomalies.

----
# 3. K-Means avec Distances aux Centres

##### Algorithme
Comme précédemment, K-Means partitionne un ensemble de données en `k` clusters en minimisant la variance intra-cluster.

##### Détection d'Anomalies
Les anomalies sont définies comme les points dont la distance aux centres de leurs clusters respectifs est supérieure à un certain seuil (moyenne des distances plus deux écarts-types) :
- **Anomalie** : Un point est considéré comme une anomalie si sa distance au centre de son cluster est supérieure au seuil défini.

Dans le code, cela se fait par :
```python
distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)
threshold = np.mean(distances) + 2 * np.std(distances)
anomalies = distances > threshold
```
- `distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)` : Cette ligne calcule la distance euclidienne entre chaque point et le centre de son cluster.
- `threshold = np.mean(distances) + 2 * np.std(distances)` : Ici, nous définissons un seuil pour détecter les anomalies. Ce seuil est la moyenne des distances plus deux fois l'écart-type des distances.
- `anomalies = distances > threshold` : Enfin, nous considérons comme anomalies les points dont la distance au centre de leur cluster dépasse ce seuil.

### Comparaison des Méthodes

| **Critère**                      | **K-Means avec Largeur de Silhouette**                                                                                                                | **DBSCAN**                                                                                                  | **K-Means avec Distances aux Centres**                                                                                     |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **Type de données**              | Fonctionne bien avec des clusters sphériques et de taille similaire.                                                                                    | Adapté pour des données avec des formes irrégulières et des densités variées.                               | Fonctionne bien avec des clusters sphériques et de taille similaire.                                                       |
| **Paramètres**                   | Nombre de clusters `k`.                                                                                                                                | Paramètres `eps` (rayon de voisinage) et `min_samples` (nombre minimum de points pour former un cluster).   | Nombre de clusters `k` et seuil pour les distances (moyenne + 2 écarts-types).                                             |
| **Détection d'anomalies**        | Basée sur la largeur de silhouette : les points avec une silhouette négative sont des anomalies.                                                       | Basée sur la densité : les points marqués comme du bruit (label `-1`) sont des anomalies.                   | Basée sur les distances aux centres de clusters : les points au-delà d'un certain seuil sont des anomalies.                |
| **Complexité**                   | Relativement simple à implémenter et à comprendre.                                                                                                      | Peut être plus complexe à paramétrer et à comprendre, mais très efficace pour des données avec des densités variées. | Relativement simple à implémenter et à comprendre.                                                                          |
| **Qualité du clustering**        | La largeur de silhouette offre une mesure claire de la qualité du clustering.                                                                            | Pas de mesure directe de la qualité, mais très efficace pour détecter des anomalies dans des données bruitées.          | Pas de mesure directe de la qualité, mais l'utilisation des distances peut offrir une bonne indication des anomalies.       |

----
# Annexes

# Annexe 1 : K-Means avec Largeur de Silhouette

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
print(f

'La largeur de silhouette moyenne est : {silhouette_avg:.2f}')
print(f'Nombre d\'anomalies détectées : {len(anomalies)}')
```

# Annexe 2 : DBSCAN

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
    if (label == -1):
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

# Annexe 3 : K-Means avec Distances aux Centres

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Génération de données
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, random_state=42)

# Clustering avec k-means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Calcul des distances aux centres de clusters
distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)

# Détection des anomalies
threshold = np.mean(distances) + 2 * np.std(distances)  # Exemple de seuil
anomalies = distances > threshold

# Visualisation
plt.figure(figsize=(10, 6))
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    cluster_points = X[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(label), label=f'Cluster {label + 1}')

# Ajouter les centres des clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', edgecolor='black', label='Centres des clusters')

# Ajouter les anomalies
plt.scatter(X[anomalies, 0], X[anomalies, 1], s=100, c='black', marker='o', edgecolor='red', label='Anomalies')

plt.title(f'Clusters K-Means avec Détection d\'Anomalies (Silhouette moyenne : 0.59)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
```

Pour plus de détails sur les implémentations, voir les annexes ci-dessus et le NOTEBOOK.

# Annexe - table des matières du NOTEBOOK

1. [**PLAN**](#plan)
2. [THÉORIE 1 - Score d'Evaluation de Silhouette pour le Clustering](#theorie-1---score-devaluation-de-silhouette-pour-le-clustering)
3. [CODE 1 LIÉ À LA THÉORIE 1](#code-1-lie-a-la-theorie-1)
4. [EXERCICE 1 LIÉ À LA THÉORIE 1 + CODE 1](#exercice-1-lie-a-la-theorie-1--code-1)
5. [THÉORIE 2 - Cohésion et Séparation dans le Clustering](#theorie-2---cohesion-et-separation-dans-le-clustering)
6. [CODE 2 - Observer visuellement la cohésion et la séparation dans le score de silhoette](#code-2---observer-visuellement-la-cohesion-et-la-separation-dans-le-score-de-silhoette)
7. [**THÉORIE 3 - DÉTECTION D'ANOMALIE**](#theorie-3---detection-danomalie)
8. [**CODE 3 - DÉTECTION D'ANOMALIE (EXAMPLE 1)**](#code-3---detection-danomalie-example-1)
9. [**INTERPRÉTATION 3 du graphique du CODE 3**](#interpretation-3-du-graphique-du-code-3)
10. [**CODE 4 - DÉTECTION D'ANOMALIE (EXAMPLE 2)**](#code-4---detection-danomalie-example-2)
11. [**INTERPRÉTATION 4 du graphique du CODE 4**](#interpretation-4-du-graphique-du-code-4)
12. [**Comment détecter une anomalie , explication de notre code**](#comment-detecter-une-anomalie--explication-de-notre-code)
13. [**CODE 5 - DÉTECTION D'ANOMALIE (EXAMPLE 3)**](#code-5---detection-danomalie-example-3)
14. [**Exemple 3 : détection d'anomalie basé sur la distance (autre méthode - anomalies = distances > threshold)**](#exemple-3--detection-danomalie-base-sur-la-distance-autre-methode---anomalies--distances--threshold)
15. [THÉORIE 1 - Score d'Evaluation de Silhouette pour le Clustering](#theorie-1---score-devaluation-de-silhouette-pour-le-clustering-1)
16. [Introduction](#introduction)
17. [Calcul du Score de Silhouette](#calcul-du-score-de-silhouette)
18. [Interprétation](#interpretation)
19. [Exemple en Python basique pour savoir programmer](#exemple-en-python-basique-pour-savoir-programmer)
20. [Données d'exemple](#donnees-dexemple)
21. [Clustering](#clustering)
22. [Calcul du score de silhouette](#calcul-du-score-de-silhouette-1)
23. [Conclusion](#conclusion)
24. [EXERCICE 1.1 LIÉ À LA THÉORIE 1 + CODE 1](#exercice-11-lie-a-la-theorie-1--code-1)
25. [EXERCICE 1.2 LIÉ À LA THÉORIE 1 + CODE 1](#exercice-12-lie-a-la-theorie-1--code-1)
26. [Interprétation 1 du graphique du CODE 1](#interpretation-1-du-graphique-du-code-1)
27. [Concepts Clés](#concepts-cles)
28. [Interprétation Visuelle](#interpretation-visuelle)
29. [Validation et Analyse](#validation-et-analyse)
30. [Cohésion](#cohesion)
31. [Séparation](#separation)
32. [Exemples Illustratifs](#exemples-illustratifs)
33. [Application dans le Calcul du Score de Silhouette](#application-dans-le-calcul-du-score-de-silhouette)
34. [Conclusion](#conclusion-1)
35. [**DÉTECTION**](#detection)
36. [Calcul de la largeur de silhouette](#calcul-de-la-largeur-de-silhouette)
37. [Ajouter les anomalies (points avec silhouette négative)](#ajouter-les-anomalies-points-avec-silhouette-negative)
38. [Affichage de la largeur de silhouette moyenne](#affichage-de-la-largeur-de-silhouette-moyenne)
39. [Étapes de Détection des Anomalies](#etapes-de-detection-des-anomalies)
40. [Visualisation des Anomalies](#visualisation-des-anomalies)
41. [Résumé](#resume)
42. [Analyse du Graphique](#analyse-du-graphique)
43. [Conclusion](#conclusion-2)
44. [Détection des Anomalies dans le Code](#detection-des-anomalies-dans-le-code)
45. [Calcul et Visualisation de la Largeur de Silhouette](#calcul-et-visualisation-de-la-largeur-de-silhouette)
46. [Résumé](#resume-1)
47. [Étapes de Détection des Anomalies](#etapes-de-detection-des-anomalies-1)
48. [Interprétation du Graphique](#interpretation-du-graphique)
49. [Exemple de Code Utilisé](#exemple-de-code-utilise)
50. [Génération de données](#generation-de-donnees)
51. [Clustering avec k-means](#clustering-avec-k-means)
52. [Calcul des distances aux centres de clusters](#calcul-des-distances-aux-centres-de-clusters)
53. [Détection des anomalies](#detection-des-anomalies)
54. [Visualisation](#visualisation)
55. [Ajouter les centres des clusters](#ajouter-les-centres-des-clusters)
56. [Ajouter les anomalies](#ajouter-les-anomalies)
57. [Résumé](#resume-2)

