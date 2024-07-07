

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
