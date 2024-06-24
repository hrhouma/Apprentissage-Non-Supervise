==> https://www.youtube.com/watch?v=HuK48FxITao&t=14s&ab_channel=InformatiqueSansComplexe

# 1 - (PRATIQUE partie 1)

# Tutoriel sur l'Apprentissage Non Supervisé

L'apprentissage non supervisé est une méthode utilisée pour entraîner une intelligence artificielle (IA) avec des données non étiquetées ou non classées. Contrairement à l'apprentissage supervisé, l'IA n'a pas de réponses correctes pour calculer son erreur. L'objectif est de découvrir des groupes ou des structures cachées dans les données. Parmi les modèles les plus connus, on retrouve le k-means qui permet de regrouper des données en clusters. D'autres modèles peuvent trouver des relations entre les données, comme l'algorithme Apriori, utilisé pour analyser les paniers de marché et identifier les articles fréquemment achetés ensemble.

Pour illustrer ce concept, nous allons utiliser le dataset Iris, un ensemble de données classique en statistique et en apprentissage automatique. Il contient 150 observations de trois espèces différentes d'iris (fleurs), chacune représentée par 50 échantillons. Pour chaque échantillon, quatre caractéristiques sont mesurées : la longueur et la largeur des sépales, ainsi que la longueur et la largeur des pétales, toutes en centimètres. Ce dataset est fréquemment utilisé pour des tâches de classification où l'objectif est de prédire l'espèce de l'iris en fonction de ces mesures.

### Étapes du Tutoriel

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

#### Visualisation des Données
```python
# Afficher les données sous forme de nuage de points
plt.scatter(data[:, 2], data[:, 3])
plt.title('Iris Dataset')
plt.xlabel('Longueur des pétales (cm)')
plt.ylabel('Largeur des pétales (cm)')
plt.show()
```

#### Création du Modèle K-means
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

#### Visualisation des Centres des Clusters
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

#### Utilisation du Modèle pour Prédire des Préférences Client
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

# 2 - (PRATIQUE partie 2)


## Tutoriel : Clustering KMeans avec le jeu de données Iris


**Partie 1:** Apprentissage non supervisés, nous l'avons vu, pour fonctionner, une IA doit déjà s'entraîner parmi les différentes méthodes. L'apprentissage non supervisé est une possibilité. Pour mieux le comprendre. Nous allons voir ensemble un cas concret. C'est parti. Dans le cas d'un apprentissage non supervisé, on utilise des données qui ne sont pas étiquetées ou classées. L'IA n'a pas de réponse correcte pour calculer son erreur. L'objectif est de découvrir des groupes ou des structures cachées dans des données. Le plus connu des modèles, c'est sans doute le KMeans qui permet de regrouper des données en clusters. D'autres modèles vont, par exemple, trouver des relations entre les données, comme l'algorithme à priori. C'est un algorithme qui permet d'analyser facilement les paniers de marché et ainsi trouver des articles fréquemment achetés ensemble. Pour réaliser notre apprentissage, nous allons utiliser le dataset Iris. C'est un ensemble de données classiques en statistique et en apprentissage automatique. Il contient 150 observations de trois espèces différentes d'iris, la fleur, chacune représentée par 50 échantillons. Pour chaque échantillon, on dispose de quatre caractéristiques: la longueur et la largeur des sépales et la longueur et la largeur des pétales, toutes mesurées en centimètres. Ce dataset est fréquemment utilisé pour des tâches de classification où l'objectif est de déduire l'espèce de l'iris en fonction de ces mesures. Notre dataset Iris fait partie du sklearn.

**Partie 2:** Et on va l'importer. From sklearn import datasets. Ok, on va dire que c'est iris qui est égal à datasets.load_iris(). On va charger nos données, et nos datas, on va l'appeler X, par exemple, c'est iris.data. On peut les afficher pour voir à quoi elles ressemblent. On voit que c'est 150 lignes et sur chaque ligne, on a quatre nombres qui sont la longueur, la largeur des sépales, la longueur du pétale et la largeur du pétale, tout est caractéristique de la fleur.

```python
# Importation des bibliothèques nécessaires
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Chargement du dataset Iris
iris = datasets.load_iris()
X = iris.data

# Affichage des caractéristiques du dataset
print(X)
```

**Partie 3:** Qui a été étudiée. On a étudié 150 fleurs, 150 lignes. On peut par exemple afficher toutes ces données sous la forme d'un nuage de points. On va par exemple afficher la longueur et la largeur des pétales. On va utiliser matplotlib. On va utiliser la fonction scatter, qui nous permet d'afficher un nuage de points. On va afficher la troisième et la quatrième colonne, qui vont représenter la longueur et la largeur des pétales. On va rajouter un titre, on va l'appeler par exemple Dataset IRIS, un label pour les axes, c'est la longueur du pétale et la largeur du pétale. On va afficher avec plt.show().

```python
# Visualisation des données sous forme de nuage de points
plt.scatter(X[:, 2], X[:, 3])
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()
```

**Partie 4:** On teste et on obtient notre nuage de points.

**Partie 5:** Qui représente chaque point représente une donnée de notre dataset.

**Partie 6:** On va maintenant pouvoir ajouter notre modèle KMeans, ce qui va nous permettre de classer ces iris pour retrouver, en fonction des caractéristiques, les différentes variétés d'iris.

**Partie 7:** Pour ça, c'est toujours dans sklearn. From sklearn.cluster import KMeans, on va importer le modèle KMeans et puis ensuite, on va créer notre modèle. KMeans, on va dire que c'est un KMeans et on lui indique ici le nombre de clusters que l'on veut. Ici, on va choisir trois clusters. On sait que sur les 150 fleurs, il y a trois espèces d'iris différentes. Ça va nous permettre de les regrouper en fonction des caractéristiques. Ensuite, on va entraîner notre modèle avec un fit, en lui donnant nos données, et il va, en fonction de ces données, pouvoir les regrouper. On va les afficher, et pour pouvoir voir ça, on va utiliser le paramètre qui va nous permettre de donner la couleur en fonction des labels qu'il va trouver. Ici, les KMeans.labels_ sont les labels qu'il va donner à chaque fleur des 150 échantillons qui sont dans le dataset. En gros, il va regrouper en trois clusters: le cluster zéro, le cluster un et le cluster deux, c’est-à-dire qu’ici ça va donner, pour la couleur du point, soit la couleur zéro, soit un, soit deux. Et on va voir de façon très visuelle les trois clusters, les trois groupes qui sont créés. Oui, fit, c’est pour entraîner. Là, on voit bien les trois clusters qui ont été créés par notre algorithme KMeans: le premier, le deuxième en violet et le troisième en jaune. On peut aussi afficher les centres de ces clusters, c’est le centre du groupe. Comment on fait ça? C’est très simple. On va récupérer nos centres, donc c’est notre KMeans.cluster_centers_, et on va les afficher en rouge. On va dire que notre marqueur qui va s’afficher, ça va être une croix et la taille, on va mettre 200 pour qu’on les voit bien. On sauvegarde, on relance et on obtient nos trois groupes. Et on voit ici les centres de nos groupes, les croix en rouge.

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

# Affichage des centres des clusters
centers = kmeans.cluster_centers_
plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='X', s=200)
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()
```

**Partie 8:** Une fois notre modèle entraîné, nous allons pouvoir l’utiliser. Pour cela, nous allons proposer aux clients d’un magasin une variété d’iris qui peut leur plaire en fonction des caractéristiques. On utilisera les caractéristiques pour trouver le groupe correspondant. Maintenant, on va utiliser notre modèle et pour cela, on va simuler des clients.

**Partie 9:** On va arrêter de dessiner ça pour l’instant et on va créer notre client. Qu’est-ce que c’est un client? Ça va être une ligne qui possède des caractéristiques, c’est-à-dire des longueurs et des largeurs de pétales et de sépales préférées. Quand vous avez une fleur, vous aimez des fleurs longues et fines, ou des fleurs avec des pétales larges, pas longues. Il existe de nombreuses formes d’iris. Pour ça, on va créer un tableau. Rien de mieux que notre ami Numpy. On va l’appeler client_preferences. C’est juste un tableau de tableaux où, pour chaque valeur, on va donner la longueur du sépale, la largeur du sépale, et on met des valeurs. On va mettre 5.1, 3.5, 1.4, 0.2. On va créer trois clients. On va en faire un deuxième avec 6.2, 2.8, 4.1, 1.8 et un troisième avec 7.9, 4.1, 6.2, 2.3. On a nos trois clients, chacun avec les caractéristiques des fleurs qu’ils préfèrent. On va afficher ces données pour voir ce que ça donne. On va afficher encore la longueur et la largeur des pétales. C’est plus simple, c’est plus parlant. On va utiliser les colonnes trois et quatre. On va faire des ronds verts, par exemple. Marker='o', pour afficher un rond, s=200 encore comme taille. On affiche les trois points.

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

**Partie 10:** On va utiliser ces données pour les passer dans notre modèle, et notre modèle va nous donner la classe et la variété d’iris que le client préfère en fonction des caractéristiques.

**Partie 11:** Pour faire ça, rien de plus simple. Une fois qu’on a les préférences, on va utiliser notre fonction predict pour trouver les segments. On va appeler ça client_segments. Ça va être égal à notre KMeans, notre modèle entraîné. predict(), et on va lui passer client_preferences, une ligne pour chaque client et lui, en retour, il va nous donner les segments, en gros, la classe zéro, un ou deux correspondante à la variété d’iris. Le plus simple pour l’instant, c’est par exemple de transformer tout en liste. On affiche ça directement et là, il nous donne les segments. On obtient ici [1, 0, 2]. On sait que le client un préfère les iris de la variété 1, le client deux de la variété 0 et le client trois de la variété 2. On peut rajouter, par exemple, les données des nuages correspondants aux iris pour voir où on se trouve. C’est plus parlant. Et ici, si on a les données, on voit bien que le premier est dans le premier groupe, le deuxième au milieu du deuxième groupe et le troisième dans le troisième groupe. L’algorithme est capable de nous prédire, de nous donner la préférence de chaque utilisateur. Ici, c’est un exemple simple. Il n’y a pas beaucoup de données, mais imaginez la même chose avec, par exemple, une liste de tous les articles que vous achetez. On est capable ainsi de vous regrouper, et c’est ce que font, par exemple, les grandes enseignes en fonction de vos habitudes de consommation.

**Partie 12:** Vous savez maintenant comment fonctionne l’apprentissage sans supervision. N’hésitez pas à partager et à poser vos questions dans les commentaires. À très vite pour une nouvelle vidéo sur une question informatique en moins de trois minutes.

```python
# Prédiction des segments de préférence des clients
client_segments = kmeans.predict(client_preferences)
print(client_segments.tolist())
```

### Code complet

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

---

Ce tutoriel fournit une introduction pratique et détaillée à l'utilisation de l'apprentissage non supervisé avec l'algorithme KMeans sur le jeu de données Iris, ainsi qu'une méthode pour prédire les préférences des clients.

# 3 - (PRATIQUE partie 3)
## Tutoriel : Clustering KMeans avec le jeu de données Iris

Dans ce tutoriel, nous allons apprendre à utiliser l'algorithme de clustering KMeans pour segmenter le jeu de données Iris. Nous allons également visualiser les résultats à l'aide de Matplotlib et prédire les segments de préférence des clients.

### Étape 1 : Importation des bibliothèques nécessaires

```python
# Nous importons les bibliothèques nécessaires pour notre analyse :
# datasets de sklearn pour charger le jeu de données Iris,
# KMeans de sklearn.cluster pour appliquer l'algorithme KMeans,
# pyplot de matplotlib pour la visualisation,
# et numpy pour la manipulation de tableaux numériques.

from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
```

### Étape 2 : Chargement du jeu de données Iris

```python
# Nous chargeons le jeu de données Iris.
iris = datasets.load_iris()

# Les données d'entraînement se trouvent dans l'attribut 'data' de l'objet Iris.
X = iris.data
```

### Étape 3 : Affichage des caractéristiques du dataset

```python
# Nous affichons les données pour voir à quoi elles ressemblent.
print(X)
```

### Étape 4 : Visualisation des données sous forme de nuage de points

```python
# Nous utilisons matplotlib pour créer un nuage de points
# affichant la longueur et la largeur des pétales des fleurs.

plt.scatter(X[:, 2], X[:, 3])
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()
```

### Étape 5 : Application de l'algorithme KMeans

```python
# Nous créons un modèle KMeans avec 3 clusters (car nous savons que le jeu de données Iris a trois types de fleurs).
kmeans = KMeans(n_clusters=3)

# Nous ajustons le modèle KMeans sur les données.
kmeans.fit(X)

# Nous récupérons les étiquettes des clusters après l'ajustement du modèle.
labels = kmeans.labels_
```

### Étape 6 : Affichage des clusters avec des couleurs différentes

```python
# Nous traçons les points de données, colorés par les étiquettes de clusters prédits par KMeans.
plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()
```

### Étape 7 : Affichage des centres des clusters

```python
# Nous récupérons les centres des clusters.
centers = kmeans.cluster_centers_

# Nous affichons les points de données et les centres des clusters.
plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='X', s=200)
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()
```

### Étape 8 : Ajout des préférences clients

```python
# Nous définissons les préférences des clients sous forme de tableau numpy.
client_preferences = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.8, 4.1, 1.8],
    [7.9, 4.1, 6.2, 2.3]
])

# Nous traçons les préférences des clients sur le graphique, en utilisant des marqueurs 'o' verts.
plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='X', s=200)
plt.scatter(client_preferences[:, 2], client_preferences[:, 3], c='green', marker='o', s=200)
plt.title('Dataset IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()
```

### Étape 9 : Prédiction des segments de préférence des clients

```python
# Nous utilisons le modèle KMeans pour prédire les segments de clusters pour les préférences des clients.
client_segments = kmeans.predict(client_preferences)

# Nous imprimons les segments prédites pour chaque préférence de client.
print(client_segments.tolist())
```

### Code complet

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


# 4 - (pratique - partie 4) Utilisation de l'algorithme de clustering KMeans avec le jeu de données Iris, accompagné de commentaires et du code complet :

---

## Tutoriel : Clustering KMeans avec le jeu de données Iris

Dans ce tutoriel, nous allons apprendre à utiliser l'algorithme de clustering KMeans pour segmenter le jeu de données Iris. Nous allons également visualiser les résultats à l'aide de Matplotlib et prédire les segments de préférence des clients.

### Étape 1 : Importation des bibliothèques nécessaires

```python
# Nous importons les bibliothèques nécessaires pour notre analyse :
# datasets de sklearn pour charger le jeu de données Iris,
# KMeans de sklearn.cluster pour appliquer l'algorithme KMeans,
# pyplot de matplotlib pour la visualisation,
# et numpy pour la manipulation de tableaux numériques.

from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
```

### Étape 2 : Chargement du jeu de données Iris

```python
# Nous chargeons le jeu de données Iris.
iris = datasets.load_iris()

# Les données d'entraînement se trouvent dans l'attribut 'data' de l'objet Iris.
X = iris.data
```

### Étape 3 : Application de l'algorithme KMeans

```python
# Nous créons un modèle KMeans avec 3 clusters (car nous savons que le jeu de données Iris a trois types de fleurs).
kmeans = KMeans(n_clusters=3)

# Nous ajustons le modèle KMeans sur les données.
kmeans.fit(X)

# Nous récupérons les centres des clusters après l'ajustement du modèle.
centers = kmeans.cluster_centers_
```

### Étape 4 : Visualisation des résultats

```python
# Nous traçons les points de données, colorés par les étiquettes de clusters prédits par KMeans.
plt.scatter(X[:, 2], X[:, 3], c=kmeans.labels_)

# Nous ajoutons les centres des clusters au graphique, en utilisant des marqueurs 'X' rouges.
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='X', s=200)

# Nous ajoutons un titre et des labels aux axes.
plt.title('KMeans IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')

# Nous affichons le graphique.
plt.show()
```

### Étape 5 : Ajout des préférences clients

```python
# Nous définissons les préférences des clients sous forme de tableau numpy.
client_preferences = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.8, 4.1, 1.8],
    [7.9, 4.1, 6.2, 2.3]
])

# Nous traçons les préférences des clients sur le graphique, en utilisant des marqueurs 'o' verts.
plt.scatter(client_preferences[:, 2], client_preferences[:, 3], c='green', marker='o', s=200)

# Nous affichons à nouveau le graphique avec les préférences des clients ajoutées.
plt.show()
```

### Étape 6 : Prédiction des segments de préférence des clients

```python
# Nous utilisons le modèle KMeans pour prédire les segments de clusters pour les préférences des clients.
client_segments = kmeans.predict(client_preferences)

# Nous imprimons les segments prédites pour chaque préférence de client.
print(client_segments.tolist())
```

### Code complet

```python
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Charger le jeu de données Iris
iris = datasets.load_iris()
X = iris.data

# Appliquer l'algorithme KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
centers = kmeans.cluster_centers_

# Visualisation des résultats
plt.scatter(X[:, 2], X[:, 3], c=kmeans.labels_)
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='X', s=200)
plt.title('KMeans IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()

# Ajout des préférences clients
client_preferences = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.8, 4.1, 1.8],
    [7.9, 4.1, 6.2, 2.3]
])
plt.scatter(X[:, 2], X[:, 3], c=kmeans.labels_)
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='X', s=200)
plt.scatter(client_preferences[:, 2], client_preferences[:, 3], c='green', marker='o', s=200)
plt.title('KMeans IRIS')
plt.xlabel('Longueur pétale')
plt.ylabel('Largeur pétale')
plt.show()

# Prédiction des segments de préférence des clients
client_segments = kmeans.predict(client_preferences)
print(client_segments.tolist())
```

---

En suivant ce tutoriel, vous devriez être capable d'appliquer l'algorithme KMeans à vos propres ensembles de données, de visualiser les résultats et de prédire les segments de clusters pour de nouvelles données.

