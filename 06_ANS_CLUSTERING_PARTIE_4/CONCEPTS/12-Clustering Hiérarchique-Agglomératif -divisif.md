# Introduction: 

- Le clustering hiérarchique est une méthode de regroupement des données qui crée une hiérarchie de clusters. Il existe deux approches principales pour le clustering hiérarchique : l'agglomératif et le divisif. Voici une explication simplifiée des deux méthodes :

### Clustering Hiérarchique Agglomératif (Ascendant)

1. **Début :** Chaque point de données commence dans son propre cluster. Ainsi, si vous avez \( n \) points de données, vous commencez avec \( n \) clusters.
2. **Fusion :** À chaque étape, les deux clusters les plus proches sont fusionnés en un seul. La proximité entre les clusters peut être mesurée de différentes façons (par exemple, la distance minimale, maximale ou moyenne entre les points des clusters).
3. **Répétition :** Ce processus de fusion se poursuit jusqu'à ce qu'il ne reste plus qu'un seul cluster, qui contient tous les points de données.
4. **Résultat :** Le résultat est une hiérarchie de clusters imbriqués, souvent représentée par un dendrogramme. Plus vous montez dans le dendrogramme, plus les clusters sont larges et moins similaires.

### Clustering Hiérarchique Divisif (Descendant)

1. **Début :** Tous les points de données commencent dans un seul cluster.
2. **Division :** À chaque étape, le cluster est divisé en deux sous-clusters. Cette division est basée sur une mesure de dissimilarité, cherchant à maximiser les différences entre les deux sous-clusters résultants.
3. **Répétition :** Ce processus de division continue jusqu'à ce que chaque point de données soit dans son propre cluster ou jusqu'à ce que l'on atteigne un nombre de clusters prédéfini.
4. **Résultat :** Le résultat est également une hiérarchie de clusters, mais construite de manière opposée à celle de l'approche agglomérative.

### Comparaison des Deux Approches

- **Agglomératif :** Commence par des clusters individuels et les fusionne progressivement. C'est plus courant car il est généralement plus simple à implémenter et à comprendre.
- **Divisif :** Commence par un seul cluster et le divise progressivement. Moins courant car il peut être plus complexe et coûteux en termes de calcul, surtout pour de grands ensembles de données.

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

Chacune de ces méthodes offre une perspective différente sur la structure des données et peut être utile en fonction du type d'analyse souhaitée.
