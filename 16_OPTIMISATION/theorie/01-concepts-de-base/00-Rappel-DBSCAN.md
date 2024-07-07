# 1 - Rappel de l'algorithme DBSCAN

## 1.1 - Théorie : 

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) est une méthode de regroupement utilisée pour découvrir des clusters de différentes formes et tailles dans des données bruitées. Je vous présente une explication simplifiée avec un exemple de la vraie vie pour comprendre les étapes.

## 1.2 - Explication Vulgarisée

Imaginez que vous êtes dans un parc rempli de groupes de personnes, et vous voulez identifier chaque groupe sans savoir combien il y en a au départ. Vous pourriez vous approcher de chaque personne et demander :

1. **Combien de personnes sont à proximité ?** (disons dans un rayon de 2 mètres)
2. **Est-ce que ces personnes à proximité sont également entourées de beaucoup de personnes ?**

En utilisant cette méthode, vous pourriez dire :
- Si une personne a beaucoup de gens autour d'elle, elle fait partie d'un groupe.
- Si une personne n'a pas beaucoup de gens autour d'elle, elle est peut-être seule (du bruit).

## 1.3 - Exemple de la Vraie Vie

Imaginez une application pour détecter les zones d'activité dans une ville en analysant les données GPS des téléphones mobiles. Chaque point GPS représente une personne à un moment donné.

1. **Points denses** : Une place publique où beaucoup de gens sont rassemblés.
2. **Bruit** : Une personne qui se promène seule dans un quartier résidentiel.

DBSCAN peut identifier automatiquement ces zones d'activité (clusters) sans avoir besoin de savoir combien de zones existent à l'avance.

## 1.4 - Étapes de DBSCAN

1. **Définir les paramètres** :
   - **ε (epsilon)** : La distance maximale pour considérer deux points comme voisins.
   - **MinPts** : Le nombre minimum de points pour former un cluster.

2. **Sélectionner un point non visité** :
   - Marquez-le comme visité.

3. **Trouver les voisins** :
   - Trouvez tous les points dans un rayon de ε autour de ce point.

4. **État des voisins** :
   - Si le nombre de voisins est inférieur à MinPts, marquez ce point comme bruit.
   - Si le nombre de voisins est supérieur ou égal à MinPts, créez un nouveau cluster.

5. **Étendre le cluster** :
   - Ajoutez tous les voisins au cluster.
   - Répétez le processus pour chacun des voisins, en cherchant leurs propres voisins et en les ajoutant au cluster si les conditions sont remplies.

6. **Répéter** :
   - Continuez avec les points non visités jusqu'à ce que tous les points soient marqués soit comme faisant partie d'un cluster, soit comme bruit.

## 1.5 - Conclusion

DBSCAN est puissant car il peut identifier des clusters de formes variées et est robuste face au bruit. Dans notre exemple de la ville, il permettrait de repérer automatiquement les lieux de rassemblement importants sans savoir combien il y en a ni où ils sont situés.


----

# Annexe 1 - Une démonstration visuelle du clustering DBSCAN

## Introduction

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) est un algorithme de clustering qui regroupe les points de données en fonction de leur densité. Contrairement à d'autres algorithmes de clustering traditionnels comme KMeans, DBSCAN est capable de créer des clusters de formes variées, ce qui en fait un outil puissant pour l'analyse de données complexes.

![DBSCAN](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/4b3c4f80-0945-4803-9841-814f53426c98)


### Les limitations de KMeans

Avant de plonger dans DBSCAN, examinons quelques-unes des limitations majeures de l'algorithme KMeans :

1. **Il ne prend pas en compte la variance et la forme des clusters** :
   - KMeans suppose que les clusters sont sphériques et de tailles similaires, ce qui peut ne pas être le cas dans les ensembles de données réels.
   
2. **Chaque point de données est assigné à un cluster, y compris le bruit** :
   - KMeans attribue chaque point à un cluster, ce qui signifie que même les points de bruit (points anormaux ou outliers) sont assignés à un cluster.

3. **Il faut spécifier le nombre de clusters** :
   - KMeans nécessite que l'utilisateur définisse à l'avance le nombre de clusters, ce qui peut être difficile à déterminer sans une connaissance préalable des données.

### Comment DBSCAN adresse ces limitations

DBSCAN surmonte les limitations de KMeans grâce aux caractéristiques suivantes :

- **Capacité à détecter des clusters de formes variées** : DBSCAN peut détecter des clusters de formes non sphériques et de tailles variées.
- **Gestion du bruit** : DBSCAN peut identifier et marquer les points de bruit, les excluant des clusters.
- **Pas besoin de spécifier le nombre de clusters** : DBSCAN détermine automatiquement le nombre de clusters en fonction de la densité des points de données.

### Comprendre DBSCAN

Pour comprendre comment fonctionne DBSCAN, il est important de connaître les concepts de base suivants :

1. **Point central** : Un point de données avec au moins `min_samples` points dans son voisinage dans un rayon `eps`.
2. **Point de bordure** : Un point de données qui se trouve dans le voisinage d'un point central mais qui n'a pas suffisamment de points dans son propre voisinage pour être un point central.
3. **Point de bruit** : Un point de données qui n'est ni un point central ni un point de bordure.

### Exemple pratique avec Python

Voici un exemple simple de l'utilisation de DBSCAN avec la bibliothèque Scikit-learn en Python :

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Générer un jeu de données synthétique en forme de lune
X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)

# Appliquer DBSCAN
db = DBSCAN(eps=0.2, min_samples=5)
labels = db.fit_predict(X)

# Tracer les résultats
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering')
plt.show()
```

### Conclusion

DBSCAN est un algorithme puissant pour le clustering de données complexes. Il surmonte les limitations de KMeans en permettant de détecter des clusters de formes variées, en gérant le bruit et en déterminant automatiquement le nombre de clusters.

Pour une compréhension plus approfondie de DBSCAN et de DBSCAN++ (une alternative plus rapide et évolutive), consultez ce [numéro de newsletter](https://lnkd.in/gEfZ23Kh).

# Référence : 
- https://blog.dailydoseofds.com/p/meet-dbscan-the-faster-and-scalable


-----
# Annexe 2 - Démonstration de l'algorithme de clustering DBSCAN

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) trouve des échantillons dans des régions de haute densité et étend les clusters à partir de ceux-ci. Cet algorithme est bien adapté pour les données contenant des clusters de densité similaire.

#### Génération des Données
Nous utilisons `make_blobs` pour créer 3 clusters synthétiques.

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Définir les centres des clusters
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

# Standardiser les données
X = StandardScaler().fit_transform(X)

# Visualiser les données générées
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1])
plt.title("Données générées")
plt.xlabel("Caractéristique 1")
plt.ylabel("Caractéristique 2")
plt.show()
```

#### Calcul de DBSCAN

On peut accéder aux labels assignés par DBSCAN en utilisant l'attribut `labels_`. Les échantillons bruités reçoivent le label `-1`.

```python
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN

# Appliquer DBSCAN sur les données
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

# Nombre de clusters en ignorant le bruit
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(f"Nombre estimé de clusters: {n_clusters_}")
print(f"Nombre estimé de points de bruit: {n_noise_}")
```

#### Évaluation des Clusters

Puisque `make_blobs` donne accès aux vrais labels des clusters synthétiques, il est possible d'utiliser des métriques d'évaluation qui tirent parti de cette information pour quantifier la qualité des clusters résultants.

```python
print(f"Homogénéité: {metrics.homogeneity_score(labels_true, labels):.3f}")
print(f"Complétude: {metrics.completeness_score(labels_true, labels):.3f}")
print(f"V-mesure: {metrics.v_measure_score(labels_true, labels):.3f}")
print(f"Indice de Rand ajusté: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
print(f"Information mutuelle ajustée: {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}")
print(f"Coefficient de silhouette: {metrics.silhouette_score(X, labels):.3f}")
```

#### Résultats du Plot
Les échantillons centraux (gros points) et les échantillons non centraux (petits points) sont colorés selon le cluster assigné. Les échantillons marqués comme bruit sont représentés en noir.

```python
unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Noir utilisé pour le bruit
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"Nombre estimé de clusters: {n_clusters_}")
plt.show()
```

### Résumé des Étapes
1. **Définir les paramètres** (`eps` et `min_samples`).
2. **Standardiser les données** pour uniformiser l'échelle.
3. **Appliquer DBSCAN** pour trouver les clusters.
4. **Évaluer les résultats** avec des métriques pertinentes.
5. **Visualiser les clusters** et le bruit pour une analyse visuelle.

# Démo 2 - DBSCAN 
- voir les projets ici  ==> 


## Références
- https://fr.wikipedia.org/wiki/DBSCAN
- https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
- https://blog.dailydoseofds.com/p/meet-dbscan-the-faster-and-scalable

