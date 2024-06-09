# Example - Création d'un nouveau carnet pour le clustering

- Pour débuter notre travail sur le clustering, entrons dans notre environnement de travail.
- En cliquant sur "Nouveau Carnet", je choisis Python 3 comme environnement d'exécution.
- Une fois le carnet ouvert, je le nomme "Clustering". 

## Chargement des données

- Avant toute chose, il est essentiel de charger nos données. 
- Dans le dossier `Data`, je trouve mon fichier `Entertainment_Clean.csv`. 
- Pour le lire, j'utilise la bibliothèque `pandas` :

```python
import pandas as pd
data = pd.read_csv("../Data/Entertainment_Clean.csv")
```

Ce fichier contient des données sur le temps passé par les étudiants à lire, regarder la télévision et jouer à des jeux vidéo chaque semaine.

## Vérification des données

Avant de procéder au clustering, vérifions que nos données sont prêtes pour la modélisation :

1. **Granularité des données** : Chaque ligne représente un étudiant unique.
2. **Valeurs non-nulles** : S'assurer qu'il n'y a pas de valeurs manquantes.
3. **Type numérique** : Les données doivent être numériques pour l'analyse.

Après ces vérifications, il est souvent utile de visualiser les données pour comprendre leur distribution et potentiellement identifier des clusters visuellement.

## Modélisation : Clustering K-means

Passons à la modélisation. Nous utiliserons l'algorithme K-means de la bibliothèque scikit-learn pour identifier des groupes d'étudiants selon leurs habitudes de divertissement :

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42).fit(data)
```

Ce code configure un modèle K-means pour trouver deux clusters dans notre ensemble de données. `random_state=42` garantit que les résultats sont reproductibles.

## Visualisation des Clusters

Après avoir ajusté le modèle, il est crucial de visualiser les résultats pour interpréter les clusters. Utilisons `matplotlib` pour créer une visualisation :

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(data['Books'], data['Video Games'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Heures passées à lire')
plt.ylabel('Heures passées à jouer')
plt.title('Visualisation des Clusters d\'Étudiants')
plt.show()
```

Cette visualisation nous aide à voir comment les étudiants sont regroupés selon le temps qu'ils consacrent à lire et à jouer.


## Visualisation des Clusters K-means

Maintenant que nous avons appliqué l'algorithme K-means pour identifier des clusters dans nos données, il est temps de visualiser ces clusters pour mieux comprendre comment les données sont organisées. Cette étape, bien que non essentielle, est extrêmement utile pour interpréter les résultats de manière intuitive, surtout lorsqu'on présente les résultats à un public non technique.

#### Préparation de la visualisation

Pour commencer, nous allons importer les bibliothèques nécessaires et préparer nos données pour la visualisation :

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Intégration des étiquettes de clusters aux données
data['Cluster'] = kmeans.labels_
```

#### Création du nuage de points 3D

Nous allons utiliser un graphique en trois dimensions pour représenter chaque dimension de nos données (livres, émissions de télévision, jeux vidéo) sur un axe différent. Chaque point dans le graphique représentera un étudiant, coloré selon le cluster auquel il appartient :

```python
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(data['Books'], data['TV Shows'], data['Video Games'], 
                     c=data['Cluster'], cmap='viridis', edgecolor='k', s=50, alpha=0.5)

# Ajout des étiquettes
ax.set_xlabel('Heures passées à lire')
ax.set_ylabel('Heures passées à regarder des émissions de télévision')
ax.set_zlabel('Heures passées à jouer à des jeux vidéo')

# Légende
plt.legend(*scatter.legend_elements(), title="Clusters")

# Affichage
plt.title('Visualisation des Clusters d\'Étudiants')
plt.show()
```

### Interprétation des résultats

Ce graphique nous aide non seulement à visualiser la séparation des clusters, mais aussi à observer les tendances et les comportements groupés parmi les étudiants. Par exemple, un cluster peut inclure des étudiants qui passent beaucoup de temps à jouer mais peu à lire, tandis qu'un autre regroupe ceux qui préfèrent regarder la télévision.

#### Décomposer le code

1. **Importation des bibliothèques :** `matplotlib` pour la création de graphiques, `seaborn` pour le style et `Axes3D` pour le support des graphiques 3D.
2. **Préparation des données :** Ajout des étiquettes de clusters aux données initiales pour faciliter la coloration des points dans le graphique.
3. **Configuration du graphique 3D :** Définition du type de graphique comme 3D et ajustement des paramètres visuels tels que la couleur et la transparence.
4. **Ajout de détails :** Étiquetage des axes pour une meilleure compréhension des dimensions représentées et ajout d'une légende pour identifier les clusters.

En explorant visuellement nos clusters, nous pouvons mieux comprendre comment l'algorithme K-means a organisé les données et comment nous pourrions utiliser cette information pour des applications pratiques, comme des campagnes de marketing ciblées ou des recommandations personnalisées. Cette visualisation rend les résultats tangibles et plus faciles à communiquer, surtout lorsqu'il s'agit de présenter des analyses complexes à des parties prenantes qui pourraient ne pas être familières avec le data mining.


## Conclusion

En suivant ces étapes, vous avez non seulement préparé vos données mais aussi appliqué et visualisé les résultats d'un modèle de clustering. Ce processus nous permet de tirer des insights significatifs sur les comportements des étudiants, essentiels pour des décisions ciblées, comme ajuster les stratégies de marketing de la bibliothèque.
