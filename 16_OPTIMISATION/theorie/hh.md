# Métriques Utilisées dans l'Apprentissage Non Supervisé - Partie #1

## Score de Silhouette
Le score de silhouette est une mesure utilisée pour évaluer la qualité des clusters formés par un algorithme de clustering. Chaque point de données reçoit un score de silhouette basé sur deux critères : la cohésion et la séparation. La cohésion mesure à quel point un point est proche des autres points dans le même cluster, tandis que la séparation mesure la distance entre ce point et les points dans les clusters voisins.

Le score de silhouette *s(i)* pour un point *i* est défini comme :

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

où :
- *a(i)* est la distance moyenne entre *i* et tous les autres points du même cluster.
- *b(i)* est la distance moyenne entre *i* et tous les points du cluster le plus proche.

Un score de silhouette varie de -1 à 1, où un score proche de 1 indique que les points sont bien groupés dans leurs clusters respectifs et bien séparés des autres clusters. Un score proche de 0 indique que les points sont à la frontière des clusters, et un score négatif signifie que les points sont probablement dans le mauvais cluster.

## Indice de Davies-Bouldin
L'indice de Davies-Bouldin (DBI) évalue la qualité du clustering en comparant la moyenne des dispersions intra-cluster à la séparation inter-cluster. Il est défini comme :

$$
DBI = \frac{1}{k} \sum_{i=1}^{k} \max_{j \ne i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)
$$

où :
- *k* est le nombre de clusters.
- *sigma_i* est la dispersion intra-cluster pour le cluster \( i \).
- *d(c_i, c_j)* est la distance entre les centres des clusters \( i \) et \( j \).

Un indice de Davies-Bouldin faible indique que les clusters sont compacts et bien séparés les uns des autres, suggérant un bon clustering.

## Cohésion et Séparation
La cohésion (\( a(i) \)) et la séparation (\( b(i) \)) sont deux critères clés pour évaluer la qualité des clusters :
- La cohésion, ou intra-cluster distance, mesure à quel point les points de données dans un même cluster sont proches les uns des autres.
- La séparation, ou inter-cluster distance, mesure la distance entre les différents clusters.

## Indice de Rand Ajusté (ARI)
L'indice de Rand ajusté (ARI) est une mesure de la similarité entre deux partitions d'un ensemble de données. Il est défini comme :

$$
ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2}}{0.5 \left[ \sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2} \right] - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2}}
$$

où \( n_{ij} \) est le nombre de points dans les clusters \( i \) et \( j \), \( a_i \) et \( b_j \) sont les sommes des lignes et des colonnes respectivement.

## Normalized Mutual Information (NMI)
La Normalized Mutual Information (NMI) est une mesure utilisée pour comparer deux partitions d'un ensemble de données. Elle est définie comme :

$$
NMI(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}
$$

où :
- \( I(U; V) \) est l'information mutuelle entre les partitions \( U \) et \( V \).
- \( H(U) \) et \( H(V) \) sont les entropies des partitions \( U \) et \( V \).

## Courbe d'Inertie pour K-means
La courbe d'inertie est un outil graphique utilisé pour déterminer le nombre optimal de clusters dans l'algorithme K-means. L'inertie (\( WCSS \)) est définie comme la somme des distances au carré entre chaque point de données et le centre de son cluster :

$$
WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} \| x - \mu_i \|^2
$$

En traçant l'inertie en fonction du nombre de clusters, on peut observer comment l'inertie diminue à mesure que le nombre de clusters augmente. Le "coude" de la courbe indique le point optimal où ajouter plus de clusters n'améliore plus significativement l'inertie.

