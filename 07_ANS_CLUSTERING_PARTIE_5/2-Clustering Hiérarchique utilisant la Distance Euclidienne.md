**Clustering Hiérarchique utilisant la Distance Euclidienne**

Le clustering hiérarchique est une méthode de regroupement qui cherche à construire une hiérarchie de clusters. Voici une présentation de cette méthode en utilisant la distance euclidienne :

### 1. Introduction au Clustering Hiérarchique

Le clustering hiérarchique est une technique de classification non supervisée qui vise à regrouper des données en formant une hiérarchie de clusters. Cette méthode est souvent visualisée sous forme de dendrogramme.

### 2. Distance Euclidienne

La distance euclidienne est une mesure couramment utilisée pour calculer la similarité entre deux points dans un espace multidimensionnel. Elle est définie comme la longueur du segment de droite reliant deux points \(A\) et \(B\) dans un espace n-dimensionnel, calculée comme suit :

\[ d(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2} \]

### 3. Algorithme de Clustering Hiérarchique

L'algorithme de clustering hiérarchique peut être de deux types :
- **Agglomératif (Ascendant)** : Commence par considérer chaque point comme un cluster unique et fusionne progressivement les clusters les plus proches jusqu'à ce qu'un seul cluster soit formé.
- **Divisif (Descendant)** : Commence par un seul cluster incluant tous les points et divise successivement les clusters jusqu'à ce que chaque point soit dans son propre cluster.

### 4. Processus Agglomératif utilisant la Distance Euclidienne

1. **Initialisation** : Chaque point de données est initialement considéré comme un cluster individuel.
2. **Calcul des distances** : Calculez la matrice des distances euclidiennes entre tous les points.
3. **Fusion des clusters** : Trouvez les deux clusters les plus proches et fusionnez-les pour former un nouveau cluster.
4. **Mise à jour des distances** : Mettez à jour la matrice des distances pour refléter la fusion.
5. **Répétition** : Répétez les étapes 3 et 4 jusqu'à ce qu'il ne reste plus qu'un seul cluster.

### 5. Visualisation avec un Dendrogramme

Un dendrogramme est un arbre de diagrammes qui montre la disposition des clusters produits par le clustering hiérarchique. L'axe vertical représente la distance ou la similarité entre les clusters. Voici comment interpréter un dendrogramme :
- Les points situés à la base du dendrogramme représentent les points de données individuels.
- Les branches du dendrogramme montrent où et quand les clusters sont fusionnés.
- Plus la hauteur de la fusion est grande, moins les clusters sont similaires.

### 6. Exemple Pratique en Python

Voici un exemple simple de clustering hiérarchique en utilisant la distance euclidienne avec la bibliothèque scikit-learn en Python :

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Générer des données de test
X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.60, random_state=0)

# Appliquer le clustering hiérarchique
Z = linkage(X, 'ward', metric='euclidean')

# Tracer le dendrogramme
plt.figure(figsize=(10, 7))
plt.title("Dendrogramme de Clustering Hiérarchique")
plt.xlabel("Indice de l'échantillon")
plt.ylabel("Distance")
dendrogram(Z)
plt.show()
```

Dans cet exemple, `make_blobs` génère des données de test, `linkage` applique le clustering hiérarchique en utilisant la méthode de Ward et la distance euclidienne, et `dendrogram` visualise le dendrogramme.

### Conclusion

Le clustering hiérarchique avec la distance euclidienne est une méthode puissante pour découvrir la structure et les relations entre les données. La visualisation à l'aide d'un dendrogramme permet de mieux comprendre les clusters formés et les relations entre eux.
