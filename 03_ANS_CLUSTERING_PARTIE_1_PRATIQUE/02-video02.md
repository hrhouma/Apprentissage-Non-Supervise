# Référence:
- https://www.youtube.com/watch?v=HuK48FxITao&t=14s&ab_channel=InformatiqueSansComplexe

# Tutoriel sur l'Apprentissage Non Supervisé

L'apprentissage non supervisé est une méthode utilisée pour entraîner une intelligence artificielle (IA) avec des données non étiquetées ou non classées. Contrairement à l'apprentissage supervisé, l'IA n'a pas de réponses correctes pour calculer son erreur. L'objectif est de découvrir des groupes ou des structures cachées dans les données. Parmi les modèles les plus connus, on retrouve le k-means qui permet de regrouper des données en clusters. D'autres modèles peuvent trouver des relations entre les données, comme l'algorithme Apriori, utilisé pour analyser les paniers de marché et identifier les articles fréquemment achetés ensemble.

Pour illustrer ce concept, nous allons utiliser le dataset Iris, un ensemble de données classique en statistique et en apprentissage automatique. Il contient 150 observations de trois espèces différentes d'iris (fleurs), chacune représentée par 50 échantillons. Pour chaque échantillon, quatre caractéristiques sont mesurées : la longueur et la largeur des sépales, ainsi que la longueur et la largeur des pétales, toutes en centimètres. Ce dataset est fréquemment utilisé pour des tâches de classification où l'objectif est de prédire l'espèce de l'iris en fonction de ces mesures.

# Étapes du Tutoriel

#### Importation des Données
```python
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Importer le dataset Iris
iris = datasets.load_iris()
data = iris.data

# Afficher les données
print(data)
```

# Visualisation des Données
```python
# Afficher les données sous forme de nuage de points
plt.scatter(data[:, 2], data[:, 3])
plt.title('Iris Dataset')
plt.xlabel('Longueur des pétales (cm)')
plt.ylabel('Largeur des pétales (cm)')
plt.show()
```

# Création du Modèle K-means
```python
# Importer KMeans et créer le modèle
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

# Afficher les clusters
labels = kmeans.labels_
plt.scatter(data[:, 2], data[:, 3], c=labels)
plt.title('Clusters Iris')
plt.xlabel('Longueur des pétales (cm)')
plt.ylabel('Largeur des pétales (cm)')
plt.show()
```

# Visualisation des Centres des Clusters
```python
# Récupérer et afficher les centres des clusters
centers = kmeans.cluster_centers_
plt.scatter(data[:, 2], data[:, 3], c=labels)
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='x', s=200)
plt.title('Clusters Iris avec Centres')
plt.xlabel('Longueur des pétales (cm)')
plt.ylabel('Largeur des pétales (cm)')
plt.show()
```

# Utilisation du Modèle pour Prédire des Préférences Client
```python
# Création de préférences clients
client_preferences = np.array([[6.2, 2.8, 4.8, 1.8],
                               [5.8, 2.7, 5.1, 1.9],
                               [7.1, 3.0, 5.9, 2.1]])

# Prédire les segments clients
client_segments = kmeans.predict(client_preferences)
print("Segments prédits pour les clients:", client_segments)

# Visualiser les préférences clients dans les clusters
plt.scatter(data[:, 2], data[:, 3], c=labels)
plt.scatter(client_preferences[:, 2], client_preferences[:, 3], c='green', marker='o', s=200)
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='x', s=200)
plt.title('Clusters Iris avec Préférences Clients')
plt.xlabel('Longueur des pétales (cm)')
plt.ylabel('Largeur des pétales (cm)')
plt.show()
```

Avec ce tutoriel, vous avez appris les bases de l'apprentissage non supervisé en utilisant l'exemple du dataset Iris et le modèle k-means. Vous pouvez maintenant appliquer ces concepts pour découvrir des structures cachées dans d'autres jeux de données. N'hésitez pas à partager vos questions et commentaires!

---

# Tutoriel : Clustering KMeans avec le jeu de données Iris

# Étape 1 : Importation des bibliothèques nécessaires

```python
# Importation des bibliothèques nécessaires
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
```

# Étape 2 : Chargement du jeu de données Iris

```python
# Chargement du dataset Iris
iris = datasets.load_iris()
X = iris.data
```

# Étape 3 : Affichage des caractéristiques du dataset

```python
# Affichage des caractéristiques du dataset
print(X)
```

# Étape 4 : Visualisation des données sous forme de nuage de points

```python
# Visualisation des données sous forme de nuage de points
plt.scatter(X[:, 2], X[:, 3])
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()
```

# Étape 5 : Application de l'algorithme KMeans

```python
# Application de l'algorithme KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_

# Affichage des clusters avec des couleurs différentes
plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()
```

# Étape 6 : Affichage des centres des clusters

```python
# Affichage des centres des clusters
centers = kmeans.cluster_centers_
plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='X', s=200)
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()
```

# Étape 7 : Ajout des préférences clients

```python
# Ajout des préférences clients
client_preferences = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.8, 4.1, 1.8],
    [7.9, 4.1, 6.2, 2.3]
])

# Affichage des préférences clients
plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='X', s=200)
plt.scatter(client_preferences[:, 2], client_preferences[:, 3], c='green', marker='o', s=200)
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()
```

# Étape 8 : Prédiction des segments de préférence des clients

```python
# Prédiction des segments de préférence des clients
client_segments = kmeans.predict(client_preferences)
print(client_segments.tolist())
```

# Code complet

```python
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Charger le jeu de données Iris
iris = datasets.load_iris()
X = iris.data

# Affichage des caractéristiques du dataset
print(X)

# Visualisation des données sous forme de nuage de points
plt.scatter(X[:, 2], X[:, 3])
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()

# Appliquer l'algorithme KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_

# Affichage des clusters avec des couleurs différentes
plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()

# Affichage des centres des clusters
centers = kmeans.cluster_centers_
plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='X', s=200)
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()

# Ajout des préférences clients
client_preferences = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.8, 4.1, 1.8],
    [7.9, 4.1, 6.2, 2.3]
])
plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='X', s=200)
plt.scatter(client_preferences[:, 2], client_preferences[:, 3], c='green', marker='o', s=200)
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()

# Prédiction des segments de préférence des clients


client_segments = kmeans.predict(client_preferences)
print(client_segments.tolist())
```

En suivant ce tutoriel, vous devriez être capable d'appliquer l'algorithme KMeans à vos propres ensembles de données, de visualiser les résultats et de prédire les segments de clusters pour de nouvelles données.
