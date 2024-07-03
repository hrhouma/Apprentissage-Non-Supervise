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
