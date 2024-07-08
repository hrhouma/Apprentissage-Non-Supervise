Voici le texte formaté avec les équations LaTeX pour être utilisé dans un fichier README.md sur GitHub :

# Référence :
- https://medium.com/@Suraj_Yadav/understanding-intra-cluster-distance-inter-cluster-distance-and-dun-index-a-comprehensive-guide-a8de726f5769

## Guide Complet sur la Distance Intra-Cluster, la Distance Inter-Cluster et l'Indice de Dunn

### Distance Intra-Cluster

#### Exemple de Calcul

Considérons un cluster avec trois points de données : A, B et C. Supposons que la distance euclidienne soit utilisée comme métrique de distance.

- Les coordonnées des points de données sont :
  - A : (2, 4)
  - B : (3, 5)
  - C : (5, 7)

Pour calculer la distance intra-cluster, nous calculons la distance moyenne entre toutes les paires de points de données au sein du cluster. Nous avons trois paires : (A, B), (A, C), et (B, C).

# Distance entre A et B 
$$
\sqrt{(2-3)^2 + (4-5)^2} = \sqrt{1 + 1} = \sqrt{2} \approx 1.414
$$

# Distance entre A et C
$$
\sqrt{(2-5)^2 + (4-7)^2} = \sqrt{9 + 9} = \sqrt{18} \approx 4.243
$$

# Distance entre B et C 
$$
\sqrt{(3-5)^2 + (5-7)^2} = \sqrt{4 + 4} = \sqrt{8} \approx 2.828
$$

# Distance moyenne 
$$
\text{Distance moyenne} = \frac{1.414 + 4.243 + 2.828}{3} \approx 2.828
$$

### Distance Inter-Cluster

#### Exemple de Calcul

Considérons trois clusters avec leurs centroïdes respectifs :

- Cluster 1 centroïde : (2, 4)
- Cluster 2 centroïde : (6, 8)
- Cluster 3 centroïde : (10, 12)

Nous allons calculer la distance entre chaque paire de centroïdes en utilisant la distance euclidienne.

# Distance entre le Cluster 1 et le Cluster 2 
$$
\sqrt{(6-2)^2 + (8-4)^2} = \sqrt{16 + 16} = \sqrt{32} \approx 5.657
$$

# Distance entre le Cluster 1 et le Cluster 3
$$
\sqrt{(10-2)^2 + (12-4)^2} = \sqrt{64 + 64} = \sqrt{128} \approx 11.314
$$

# Distance entre le Cluster 2 et le Cluster 3
$$
\sqrt{(10-6)^2 + (12-8)^2} = \sqrt{16 + 16} = \sqrt{32} \approx 5.657
$$



### Indice de Dunn

L'indice de Dunn est une métrique utilisée pour évaluer la qualité des résultats de clustering. Il est calculé en utilisant la formule suivante

Voici l'équation de l'indice de Dunn formatée correctement :

$$
\text{Indice de Dunn} = \frac{\text{min\_intercluster\_distance}}{\text{max\_intracluster\_distance}}
$$

Cette formule est utilisée pour évaluer la qualité des résultats de clustering en mesurant la compacité et la séparation des clusters.

Citations:
[1] https://pplx-res.cloudinary.com/image/upload/v1720439257/user_uploads/ovehdxysd/image.jpg
[2] https://fr.wikipedia.org/wiki/Indice_de_Dunn
[3] https://en.wikipedia.org/wiki/Dunn_index
[4] https://permetrics.readthedocs.io/en/joss-paper/pages/clustering/DI.html
[5] https://www.geeksforgeeks.org/dunn-index-and-db-index-cluster-validity-indices-set-1/
[6] https://ruivieira.dev/dunn-index.html
[7] https://www.datanovia.com/en/lessons/cluster-validation-statistics-must-know-methods/
[8] https://www.linkedin.com/pulse/quantifying-excellence-demystifying-clustering-metrics-bathula


#### min\_intercluster\_distance

La distance minimale entre toute paire de points de données de clusters différents.

#### max\_intracluster\_distance

La distance maximale entre toute paire de points de données au sein du même cluster.


#### Exemple de Calcul

Considérons une solution de clustering avec trois clusters : Cluster 1, Cluster 2, et Cluster 3.

- Distance inter-cluster minimale : Supposons que nous avons calculé les distances suivantes :
  - Distance entre Cluster 1 et Cluster 2 : 4.5
  - Distance entre Cluster 1 et Cluster 3 : 3.2
  - Distance entre Cluster 2 et Cluster 3 : 5.1

  Dans ce cas, la distance inter-cluster minimale est 3.2.

- Distance intra-cluster maximale : Supposons que nous trouvons les distances suivantes au sein de chaque cluster :
  - Cluster 1 : 2.1
  - Cluster 2 : 1.8
  - Cluster 3 : 2.5

  La distance intra-cluster maximale est donc 2.5.

# Indice de Dunn 
$$
\text{Indice de Dunn} = \frac{3.2}{2.5} \approx 1.28
$$

Citations:
[1] https://www.geeksforgeeks.org/ml-intercluster-and-intracluster-distance/
[2] http://abdelhamid-djeffal.net/web_documents/coursclustering1819.pdf
[3] https://search.r-project.org/CRAN/refmans/clv/html/cluster_scatter.html
[4] https://en.wikipedia.org/wiki/Cluster_analysis
