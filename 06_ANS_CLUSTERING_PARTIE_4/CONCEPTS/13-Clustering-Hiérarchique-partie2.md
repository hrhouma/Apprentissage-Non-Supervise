### Clustering Hiérarchique

#### Agglomératif (Ascendant)
1. **Départ :** Chaque point est un cluster individuel.
2. **Fusion :** Les clusters les plus proches sont fusionnés en un seul, répété jusqu'à ce qu'il ne reste qu'un cluster global.
3. **Dendrogramme :** Représentation visuelle de la hiérarchie de fusion.

#### Divisif (Descendant)
1. **Départ :** Tous les points sont dans un seul cluster.
2. **Division :** Le cluster est divisé en deux sous-clusters, répété jusqu'à ce que chaque point soit isolé ou un nombre de clusters prédéfini soit atteint.
3. **Dendrogramme :** Représentation visuelle de la hiérarchie de division.

### Comparaison
- **Agglomératif :** Part des points individuels et fusionne. Plus commun et simple.
- **Divisif :** Part du cluster global et divise. Plus complexe et coûteux.

### Illustration Simplifiée

Imaginons que vous ayez cinq points de données : A, B, C, D, et E.

**Agglomératif :**
1. A, B, C, D, E (chacun est un cluster individuel)
2. (A, B), C, D, E (A et B sont les plus proches et sont fusionnés)
3. ((A, B), C), D, E (Le cluster (A, B) est fusionné avec C, le plus proche cluster)
4. (((A, B), C), D), E (Le cluster ((A, B), C) est fusionné avec D)
5. ((((A, B), C), D), E) (Tous les clusters sont fusionnés en un seul)

**Divisif :**
1. (A, B, C, D, E) (tout le monde dans un seul cluster)
2. ((A, B, C), (D, E)) (le cluster initial est divisé en deux)
3. (((A, B), C), (D, E)) (le cluster (A, B, C) est divisé en deux)
4. (((A, B), (C)), (D, E)) (le cluster (A, B) est divisé en deux)
5. ((((A), (B)), (C)), (D, E)) (et ainsi de suite)
