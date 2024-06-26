# Exemple DBSCAN en Python : Valeur Optimale pour Epsilon (EPS)
## Choisir la Valeur Appropriée pour le Paramètre Epsilon (ε) dans l'Algorithme DBSCAN
==> Réponse : https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc#:~:text=In%20layman's%20terms%2C%20we%20find,and%20select%20that%20as%20epsilon

DBSCAN, ou Clustering Spatial Basé sur la Densité pour les Applications avec Bruit, est un algorithme d'apprentissage automatique non supervisé. Les algorithmes d'apprentissage automatique non supervisés sont utilisés pour classifier des données non étiquetées. En d'autres termes, les échantillons utilisés pour entraîner notre modèle ne sont pas fournis avec des catégories prédéfinies. Comparé à d'autres algorithmes de clustering, DBSCAN est particulièrement bien adapté aux problèmes nécessitant :

- Une connaissance minimale du domaine pour déterminer les paramètres d'entrée (par exemple, K dans k-means et Dmin dans le clustering hiérarchique).
- La découverte de clusters de formes arbitraires.
- Une bonne efficacité sur de grandes bases de données.

Si vous êtes intéressé par une lecture approfondie sur DBSCAN, l'article original peut être trouvé [ici](https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf).

#### Algorithme

Comme pour la plupart des algorithmes d'apprentissage automatique, le comportement du modèle est dicté par plusieurs paramètres. Dans cet article, nous allons aborder trois d'entre eux.

- **eps** : Deux points sont considérés comme voisins si la distance entre les deux points est inférieure au seuil epsilon.
- **min_samples** : Le nombre minimum de voisins qu'un point donné doit avoir pour être classé comme point central. Il est important de noter que le point lui-même est inclus dans le nombre minimum d'échantillons.
- **metric** : La métrique à utiliser pour calculer la distance entre les instances d'un tableau de caractéristiques (par exemple, la distance euclidienne).

L'algorithme fonctionne en calculant la distance entre chaque point et tous les autres points. Nous classons ensuite les points en trois catégories.

- **Point central** : Un point avec au moins `min_samples` points dont la distance par rapport au point est inférieure au seuil défini par epsilon.
- **Point limite** : Un point qui n'est pas à proximité d'au moins `min_samples` points mais est suffisamment proche d'un ou plusieurs points centraux. Les points limites sont inclus dans le cluster du point central le plus proche.
- **Point de bruit** : Les points qui ne sont pas assez proches des points centraux pour être considérés comme des points limites. Les points de bruit sont ignorés, c'est-à-dire qu'ils ne font partie d'aucun cluster.

#### Code

Voyons comment nous pourrions implémenter DBSCAN en Python. Pour commencer, importez les bibliothèques suivantes :

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()
```

Au lieu d'importer des données, nous pouvons utiliser scikit-learn pour générer des clusters bien définis :

```python
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
```

Comme mentionné précédemment, nous devons fournir une valeur pour epsilon qui définit la distance maximale entre deux points. L'article suivant décrit une approche pour déterminer automatiquement la valeur optimale pour epsilon.

Nous pouvons calculer la distance de chaque point à son voisin le plus proche en utilisant `NearestNeighbors`. La méthode `kneighbors` renvoie deux tableaux, l'un contenant la distance aux points les plus proches et l'autre contenant l'index de chacun de ces points.

```python
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.show()
```

La valeur optimale pour epsilon se trouve au point de courbure maximale.

Nous entraînons notre modèle en sélectionnant 0.3 pour eps et en définissant `min_samples` à 5 :

```python
m = DBSCAN(eps=0.3, min_samples=5)
m.fit(X)
```

La propriété `labels_` contient la liste des clusters et leurs points respectifs :

```python
clusters = m.labels_
```

Ensuite, nous associons chaque cluster à une couleur :

```python
colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
plt.scatter(X[:, 0], X[:, 1], c=vectorizer(clusters))
plt.show()
```

#### Conclusions

Contrairement à k-means, DBSCAN détermine le nombre de clusters. DBSCAN fonctionne en déterminant si le nombre minimum de points sont suffisamment proches les uns des autres pour être considérés comme faisant partie d'un même cluster. DBSCAN est très sensible à l'échelle puisque epsilon est une valeur fixe pour la distance maximale entre deux points.
