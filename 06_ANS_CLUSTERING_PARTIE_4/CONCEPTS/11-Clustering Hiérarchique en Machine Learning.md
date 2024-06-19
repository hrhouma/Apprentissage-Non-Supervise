# Clustering Hiérarchique en Machine Learning

## Référence : 
- https://www.javatpoint.com/hierarchical-clustering-in-machine-learning
- https://www.geeksforgeeks.org/hierarchical-clustering/
- https://www.youtube.com/watch?v=XJ3194AmH40&ab_channel=VictorLavrenko
  
Le clustering hiérarchique est un autre algorithme de machine learning non supervisé, utilisé pour regrouper des ensembles de données non étiquetées en clusters. Il est également connu sous le nom d'analyse de cluster hiérarchique ou HCA (Hierarchical Cluster Analysis).

Dans cet algorithme, nous développons une hiérarchie de clusters sous la forme d'un arbre, et cette structure en forme d'arbre est appelée dendrogramme.

Parfois, les résultats du clustering K-means et du clustering hiérarchique peuvent sembler similaires, mais ils diffèrent en fonction de leur fonctionnement. Il n'est pas nécessaire de prédéterminer le nombre de clusters comme nous le faisons dans l'algorithme K-means.

La technique de clustering hiérarchique a deux approches :

- **Agglomérative** : L'agglomérative est une approche ascendante, dans laquelle l'algorithme commence par prendre tous les points de données comme des clusters individuels et les fusionne jusqu'à ce qu'il ne reste plus qu'un seul cluster.
- **Divisive** : L'algorithme divisif est l'inverse de l'algorithme agglomératif car il s'agit d'une approche descendante.

## Pourquoi le clustering hiérarchique ?
Comme nous avons déjà d'autres algorithmes de clustering tels que le K-means, pourquoi avons-nous besoin du clustering hiérarchique ? Comme nous l'avons vu dans le clustering K-means, il y a certains défis avec cet algorithme, tels que le nombre prédéterminé de clusters et la tendance à créer des clusters de la même taille. Pour résoudre ces deux défis, nous pouvons opter pour l'algorithme de clustering hiérarchique car, avec cet algorithme, nous n'avons pas besoin de connaître le nombre prédéterminé de clusters.

Dans ce sujet, nous allons discuter de l'algorithme de clustering hiérarchique agglomératif.

## Clustering Hiérarchique Agglomératif
L'algorithme de clustering hiérarchique agglomératif est un exemple populaire de HCA. Pour regrouper les ensembles de données en clusters, il suit une approche ascendante. Cela signifie que cet algorithme considère chaque ensemble de données comme un cluster individuel au début, puis commence à combiner les paires de clusters les plus proches. Il fait cela jusqu'à ce que tous les clusters soient fusionnés en un seul cluster contenant tous les ensembles de données.

Cette hiérarchie de clusters est représentée sous la forme d'un dendrogramme.

## Comment fonctionne le Clustering Hiérarchique Agglomératif ?
Le fonctionnement de l'algorithme AHC peut être expliqué en utilisant les étapes suivantes :

### Étape 1 : Créer chaque point de données comme un cluster individuel.
Disons qu'il y a N points de données, donc le nombre de clusters sera également N.

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/cbcbde95-5f49-44c5-bcf1-1160a0912619)

### Étape 2 : Prendre les deux points de données ou clusters les plus proches et les fusionner pour former un seul cluster. Il y aura donc maintenant N-1 clusters.


![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/2f91a06e-781b-40db-a7a8-c9c672521ac7)

### Étape 3 : À nouveau, prendre les deux clusters les plus proches et les fusionner pour former un seul cluster. Il y aura N-2 clusters.
![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/d7cfa0d4-aafe-4c70-902d-3fcd092448b8)


### Étape 4 : Répéter l'étape 3 jusqu'à ce qu'il ne reste qu'un seul cluster. Ainsi, nous obtiendrons les clusters suivants. Considérez les images ci-dessous :


![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/9f0484e1-beb1-4e3c-b726-cbffe4504ffa)

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/94a035ab-92d3-4320-b476-75e20ac328c4)

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/3f9528c9-c6e1-4036-81f4-c9c8c27fda34)


### Étape 5 : Une fois que tous les clusters sont combinés en un seul grand cluster, développer le dendrogramme pour diviser les clusters selon le problème.

### Mesure de la distance entre deux clusters

Comme nous l'avons vu, la distance la plus proche entre les deux clusters est cruciale pour le clustering hiérarchique. Il existe différentes façons de calculer la distance entre deux clusters, et ces méthodes déterminent la règle de clustering. Ces mesures sont appelées méthodes de liaison (Linkage methods). Voici quelques-unes des méthodes de liaison les plus populaires :

- **Liaison simple (Single Linkage)** : Il s'agit de la distance la plus courte entre les points les plus proches des clusters. Considérez l'image ci-dessous :
![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/8af29332-bd5c-4738-bb1a-e0f9ebcde538)


- **Liaison complète (Complete Linkage)** : Il s'agit de la distance la plus éloignée entre deux points de deux clusters différents. C'est l'une des méthodes de liaison les plus populaires car elle forme des clusters plus serrés que la liaison simple.

- **Liaison moyenne (Average Linkage)** : C'est la méthode de liaison dans laquelle la distance entre chaque paire d'ensembles de données est additionnée, puis divisée par le nombre total d'ensembles de données pour calculer la distance moyenne entre deux clusters. C'est également l'une des méthodes de liaison les plus populaires.

- **Liaison par centroïde (Centroid Linkage)** : C'est la méthode de liaison dans laquelle la distance entre les centroïdes des clusters est calculée. Considérez l'image ci-dessous :

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/20b3fe40-7134-4181-bbb9-c43ee31cebef)

### À partir des approches données ci-dessus, nous pouvons appliquer l'une d'entre elles en fonction du type de problème ou des exigences métier.

## Fonctionnement du Dendrogramme dans le Clustering Hiérarchique

Le dendrogramme est une structure arborescente principalement utilisée pour mémoriser chaque étape que l'algorithme de clustering hiérarchique effectue. Dans le graphique du dendrogramme, l'axe Y montre les distances euclidiennes entre les points de données, et l'axe X montre tous les points de données de l'ensemble de données donné.

Le fonctionnement du dendrogramme peut être expliqué à l'aide du diagramme ci-dessous :

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/2a199da0-36cf-4c06-b9e3-d06ce88293be)

### Dans le diagramme ci-dessus, la partie gauche montre comment les clusters sont créés dans le clustering agglomératif, et la partie droite montre le dendrogramme correspondant.

Comme nous l'avons discuté précédemment, tout d'abord, les points de données P2 et P3 se combinent pour former un cluster, et un dendrogramme correspondant est créé, connectant P2 et P3 avec une forme rectangulaire. La hauteur est déterminée en fonction de la distance euclidienne entre les points de données.

À l'étape suivante, P5 et P6 forment un cluster, et le dendrogramme correspondant est créé. Il est plus élevé que le précédent, car la distance euclidienne entre P5 et P6 est légèrement supérieure à celle entre P2 et P3.

Ensuite, deux nouveaux dendrogrammes sont créés, combinant P1, P2 et P3 dans un dendrogramme, et P4, P5 et P6 dans un autre dendrogramme.

Enfin, le dendrogramme final est créé, combinant tous les points de données ensemble.

### Fonctionnement du Dendrogramme dans le Clustering Hiérarchique

Le dendrogramme est une structure arborescente principalement utilisée pour mémoriser chaque étape que l'algorithme de clustering hiérarchique effectue. Dans le graphique du dendrogramme, l'axe Y montre les distances euclidiennes entre les points de données, et l'axe X montre tous les points de données de l'ensemble de données donné.

Le fonctionnement du dendrogramme peut être expliqué à l'aide du diagramme ci-dessous :

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/4a5f0d12-4947-4c9c-8161-69474d037f8e)


### Explication du Diagramme

- **Partie gauche** : Montre comment les clusters sont créés dans le clustering agglomératif. Chaque point de données (P1, P2, P3, etc.) est initialement considéré comme un cluster individuel. Les points les plus proches sont progressivement fusionnés pour former des clusters plus grands.
- **Partie droite** : Montre le dendrogramme correspondant. Le dendrogramme est une représentation visuelle de la manière dont les clusters sont fusionnés à chaque étape.

### Étapes de Fusion des Clusters

1. **Fusion initiale** : 
    - Les points de données **P2** et **P3** sont les plus proches, ils se combinent pour former un cluster. Dans le dendrogramme, cela se traduit par une connexion entre **P2** et **P3** sous forme d'un rectangle. La hauteur du rectangle est déterminée par la distance euclidienne entre ces points.
  
2. **Deuxième fusion** : 
    - Les points de données **P5** et **P6** forment un cluster. Le dendrogramme montre une connexion entre **P5** et **P6**. La hauteur de cette connexion est plus grande que celle de **P2** et **P3** car la distance entre **P5** et **P6** est légèrement plus grande.

3. **Fusions suivantes** : 
    - Deux nouveaux dendrogrammes sont créés : l'un combine **P1**, **P2** et **P3** ; l'autre combine **P4**, **P5** et **P6**.

4. **Fusion finale** : 
    - Tous les points de données sont combinés ensemble dans un seul grand cluster, représenté par le dendrogramme final.

### Découpage du Dendrogramme

Nous pouvons couper la structure arborescente du dendrogramme à n'importe quel niveau selon nos besoins. Par exemple :
- En coupant au niveau **2 clusters** (ligne bleue), nous obtenons deux grands clusters, comme indiqué dans le diagramme.

### Vulgarisation

Le dendrogramme fonctionne comme une carte qui montre comment les groupes (clusters) se forment et se combinent. Imaginez que chaque point de données est une personne dans une salle. Le clustering agglomératif commence par rassembler les personnes qui sont le plus proches les unes des autres en petits groupes. Ensuite, il regroupe ces petits groupes en groupes plus grands, et ainsi de suite, jusqu'à ce qu'il ne reste plus qu'un seul grand groupe qui inclut tout le monde. Le dendrogramme est comme un arbre généalogique qui montre ces regroupements successifs.

En coupant cet "arbre" à différents niveaux, nous pouvons voir les différents groupes possibles. Par exemple, si nous coupons l'arbre au niveau de la ligne bleue indiquée, nous pouvons voir qu'il y a deux grands groupes principaux dans notre ensemble de données.


### Implémentation en Python du Clustering Hiérarchique Agglomératif

Nous allons maintenant voir l'implémentation pratique de l'algorithme de clustering hiérarchique agglomératif en utilisant Python. Pour ce faire, nous utiliserons le même problème de jeu de données que celui utilisé dans le sujet précédent sur le clustering K-means, afin de pouvoir comparer facilement les deux concepts.

Le jeu de données contient des informations sur les clients ayant visité un centre commercial pour faire des achats. Le propriétaire du centre commercial souhaite identifier des motifs ou des comportements particuliers de ses clients à partir des informations du jeu de données.

### Étapes de l'implémentation du clustering hiérarchique agglomératif en Python :

Les étapes d'implémentation seront les mêmes que pour le clustering K-means, à quelques différences près, telles que la méthode pour trouver le nombre de clusters. Voici les étapes :

1. Prétraitement des données
2. Trouver le nombre optimal de clusters en utilisant le dendrogramme
3. Entraînement du modèle de clustering hiérarchique
4. Visualisation des clusters

### Étapes de Prétraitement des Données :

Dans cette étape, nous allons importer les bibliothèques et les jeux de données pour notre modèle.

**Importation des bibliothèques**
```python
# Importation des bibliothèques  
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
```

Les lignes de code ci-dessus sont utilisées pour importer les bibliothèques afin de réaliser des tâches spécifiques, telles que numpy pour les opérations mathématiques, matplotlib pour dessiner les graphiques ou les diagrammes de dispersion, et pandas pour importer le jeu de données.

**Importation du jeu de données**
```python
# Importation du jeu de données  
dataset = pd.read_csv('Mall_Customers_data.csv')  
```

Comme mentionné ci-dessus, nous avons importé le même jeu de données Mall_Customers_data.csv, que nous avons utilisé dans le clustering K-means. Considérez la sortie ci-dessous :

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/09f44ef5-8105-4b69-9310-d86a4d9034f9)


### Extraction de la matrice de caractéristiques

Nous allons extraire uniquement la matrice de caractéristiques, car nous n'avons pas d'informations supplémentaires sur la variable dépendante. Le code est donné ci-dessous :

```python
x = dataset.iloc[:, [3, 4]].values  
```

Ici, nous avons extrait uniquement les colonnes 3 et 4, car nous utiliserons un diagramme 2D pour voir les clusters. Nous considérons donc le revenu annuel et le score de dépenses comme matrice de caractéristiques.

### Étape 2 : Trouver le nombre optimal de clusters en utilisant le Dendrogramme

Nous allons maintenant trouver le nombre optimal de clusters en utilisant le dendrogramme pour notre modèle. Pour cela, nous allons utiliser la bibliothèque scipy car elle fournit une fonction qui renvoie directement le dendrogramme pour notre code. Considérez les lignes de code ci-dessous :

```python
# Trouver le nombre optimal de clusters en utilisant le dendrogramme  
import scipy.cluster.hierarchy as shc  
dendro = shc.dendrogram(shc.linkage(x, method="ward"))  
plt.title("Dendrogramme")  
plt.ylabel("Distances Euclidiennes")  
plt.xlabel("Clients")  
plt.show()  
```

Dans les lignes de code ci-dessus, nous avons importé le module hierarchy de la bibliothèque scipy. Ce module nous fournit une méthode `shc.dendrogram()`, qui prend `linkage()` comme paramètre. La fonction linkage est utilisée pour définir la distance entre deux clusters, donc ici nous avons passé la matrice de caractéristiques `x` et la méthode "ward", la méthode de liaison populaire dans le clustering hiérarchique.

Les lignes de code restantes sont utilisées pour décrire les étiquettes du graphique du dendrogramme.

Avec ces étapes, nous sommes prêts à implémenter le clustering hiérarchique agglomératif et visualiser les clusters résultants.


### Sortie :

En exécutant les lignes de code ci-dessus, nous obtiendrons la sortie suivante. En utilisant ce dendrogramme, nous allons maintenant déterminer le nombre optimal de clusters pour notre modèle. Pour cela, nous allons trouver la distance verticale maximale qui ne coupe aucune barre horizontale. Considérez le diagramme ci-dessous :

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/6880a7fc-1e17-4d1f-847d-1d0b8f9fa994)


### Détermination du Nombre Optimal de Clusters

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/ae0490dd-4377-494a-a2d7-aa63040b7220)

Dans le diagramme ci-dessus, nous avons montré les distances verticales qui ne coupent pas leurs barres horizontales. Comme nous pouvons le visualiser, la quatrième distance semble être la plus grande, donc selon cela, le nombre de clusters sera de 5 (les lignes verticales dans cette plage). Nous pouvons également prendre le deuxième nombre car il est approximativement égal à la quatrième distance, mais nous considérerons les 5 clusters car c'est le même nombre que nous avons calculé dans l'algorithme K-means.

Ainsi, le nombre optimal de clusters sera de 5, et nous entraînerons le modèle dans l'étape suivante en utilisant ce nombre.

### Étape 3 : Entraînement du Modèle de Clustering Hiérarchique

Comme nous connaissons le nombre optimal de clusters requis, nous pouvons maintenant entraîner notre modèle. Le code est donné ci-dessous :

```python
# Entraînement du modèle hiérarchique sur le jeu de données
from sklearn.cluster import AgglomerativeClustering  
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
y_pred = hc.fit_predict(x)  
```

Dans le code ci-dessus, nous avons importé la classe `AgglomerativeClustering` du module `cluster` de la bibliothèque scikit-learn.

Ensuite, nous avons créé l'objet de cette classe nommé `hc`. La classe `AgglomerativeClustering` prend les paramètres suivants :

- `n_clusters=5` : Il définit le nombre de clusters, et nous avons pris ici 5 car c'est le nombre optimal de clusters.
- `affinity='euclidean'` : C'est une métrique utilisée pour calculer la liaison.
- `linkage='ward'` : Il définit le critère de liaison, ici nous avons utilisé la liaison "ward". Cette méthode est la méthode de liaison populaire que nous avons déjà utilisée pour créer le dendrogramme. Elle réduit la variance dans chaque cluster.

Dans la dernière ligne, nous avons créé la variable dépendante `y_pred` pour ajuster ou entraîner le modèle. Non seulement il entraîne le modèle, mais il retourne également les clusters auxquels chaque point de données appartient.

Après avoir exécuté les lignes de code ci-dessus, si nous passons par l'option explorateur de variables dans notre IDE Spyder, nous pouvons vérifier la variable `y_pred`. Nous pouvons comparer le jeu de données original avec la variable `y_pred`. Considérez l'image ci-dessous :

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/66ea33ba-a19b-476c-a5e3-1f951228ecef)

### Visualisation des Clusters

Comme nous pouvons le voir dans l'image ci-dessus, `y_pred` montre les valeurs des clusters, ce qui signifie que le client avec l'ID 1 appartient au 5ème cluster (car l'indexation commence à partir de 0, donc 4 signifie 5ème cluster), le client avec l'ID 2 appartient au 4ème cluster, et ainsi de suite.

### Étape 4 : Visualisation des Clusters

Comme nous avons entraîné notre modèle avec succès, nous pouvons maintenant visualiser les clusters correspondant au jeu de données.

Nous utiliserons les mêmes lignes de code que nous avons utilisées dans le clustering K-means, sauf une modification. Ici, nous ne tracerons pas le centroïde comme nous l'avons fait dans K-means, car ici nous avons utilisé le dendrogramme pour déterminer le nombre optimal de clusters. Le code est donné ci-dessous :

```python
# Visualisation des clusters
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')  
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')  
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')  
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')  
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')  

plt.title('Clusters de clients')
plt.xlabel('Revenu annuel (k$)')
plt.ylabel('Score de dépenses (1-100)')
plt.legend()
plt.show()
```

Ce code permet de tracer un graphique de dispersion des points de données en fonction des clusters auxquels ils appartiennent. Chaque couleur représente un cluster différent. Les étiquettes et le titre du graphique sont également ajoutés pour plus de clarté.

En exécutant ce code, vous obtiendrez une visualisation claire des différents clusters de clients, en fonction de leur revenu annuel et de leur score de dépenses.

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/eb5fa46a-afc6-4005-9dc4-734a1fb3b79e)

