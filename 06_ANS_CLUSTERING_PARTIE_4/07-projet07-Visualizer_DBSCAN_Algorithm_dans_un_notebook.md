
- https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html (DÉMO NOTEBOOK)
- https://fr.wikipedia.org/wiki/DBSCAN (THÉORIE)

# Annexe - DBSCAN Simplifié
## Théorie : https://fr.wikipedia.org/wiki/DBSCAN (THÉORIE)
**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) est une méthode de regroupement utilisée pour découvrir des clusters de différentes formes et tailles dans des données bruitées. Voici une explication simplifiée avec un exemple de la vraie vie et les étapes.

#### Explication Vulgarisée

Imaginez que vous êtes dans un parc rempli de groupes de personnes, et vous voulez identifier chaque groupe sans savoir combien il y en a au départ. Vous pourriez vous approcher de chaque personne et demander :

1. **Combien de personnes sont à proximité ?** (disons dans un rayon de 2 mètres)
2. **Est-ce que ces personnes à proximité sont également entourées de beaucoup de personnes ?**

En utilisant cette méthode, vous pourriez dire :
- Si une personne a beaucoup de gens autour d'elle, elle fait partie d'un groupe.
- Si une personne n'a pas beaucoup de gens autour d'elle, elle est peut-être seule (du bruit).

#### Exemple de la Vraie Vie

Imaginez une application pour détecter les zones d'activité dans une ville en analysant les données GPS des téléphones mobiles. Chaque point GPS représente une personne à un moment donné.

1. **Points denses** : Une place publique où beaucoup de gens sont rassemblés.
2. **Bruit** : Une personne qui se promène seule dans un quartier résidentiel.

DBSCAN peut identifier automatiquement ces zones d'activité (clusters) sans avoir besoin de savoir combien de zones existent à l'avance.

#### Étapes de DBSCAN

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

#### Conclusion

DBSCAN est puissant car il peut identifier des clusters de formes variées et est robuste face au bruit. Dans notre exemple de la ville, il permettrait de repérer automatiquement les lieux de rassemblement importants sans savoir combien il y en a ni où ils sont situés.

# Démo 1 DBSCAN
## Pratique : https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html (DÉMO NOTEBOOK)
### Démonstration de l'algorithme de clustering DBSCAN

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

# Démo 2 - DBSCAN - voie le projet 08
