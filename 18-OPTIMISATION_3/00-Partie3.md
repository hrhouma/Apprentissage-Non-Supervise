# Analyse comparative des méthodes de clustering pour des activités et la détection de chutes

### Objectif :
L'objectif est de comparer l'efficacité de K-means, DBSCAN et le clustering hiérarchique pour la détection non supervisée des activités journalères comme s'assoir, walking, sitting, chutes en utilisant un ou deux datasets parmi ceux proposés.

# Datasets :

1. **Le2i Fall Dataset** : [Lien vers le dataset](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia)
2. **Fall Detection Dataset** : [Lien vers le dataset](https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset)
3. **CAUCAFall Dataset** : [Lien vers le dataset](https://data.mendeley.com/datasets/7w7fccy7ky/4)
4. **Walker Fall Detection Dataset**: [Lien vers le dataset](https://www.kaggle.com/datasets/antonygarciag/walker-fall-detection)
5. **KFall Dataset** : [Lien vers le dataset](https://sites.google.com/view/kfalldataset)
6. **Inertial Measurement Unit Fall Detection Dataset (IMU Dataset)** : [Lien vers le dataset](https://www.frdr-dfdr.ca/repo/dataset/6998d4cd-bd13-4776-ae60-6d80221e0365)
7. **Dataset de l'ARCO Research Group**:  [Lien vers le dataset](https://arcoresearch.com/2021/04/16/dataset-for-fall-detection/)

# Important :
- *Pour K-means :*
  - Utilisez la méthode du coude pour déterminer le nombre optimal de clusters.
  - Visualisez les clusters.
  - Appliquez l'Analyse en Composantes Principales (PCA) pour visualiser les clusters en 2D ou 3D pour valider les résultats.
- Pour la *tâche 3*:
  - Il n'est pas obligatoire de se limiter au score de silhouette ; si vous identifiez un indicateur plus pertinent ou complémentaire, comme le Dunn ou l'indice de Davies-Bouldin, vous êtes encouragés à l'utiliser en expliquant votre sélection.

# Tâches :

# 1. **Prétraitement des données :**
   - Choisissez un dataset parmi ceux proposés.
   - Extrayez des caractéristiques pertinentes et normalisez les données si nécessaire.

# 2. **Application des algorithmes de clustering :**
   - Appliquez chacun des 3 algorithmes (K-means, DBSCAN et le clustering hiérarchique) sur le dataset choisi.

# 3. **Évaluation de la qualité du clustering :**
   - Calculez et comparez les scores de silhouette pour chaque algorithme appliqué.
   - Si vous trouvez d'autres scores plus intéressants ou complémentaires (par exemple, Dunn et/ou l'indice de Davies-Bouldin), vous pouvez les appliquer en justifiant votre choix.

# 4. **Optimisation :**
   - Il est nécessaire, de réappliquer certains algorithmes avec différents paramètres, comme le DBSCAN avec des configurations variées, et de recalculer les scores de silhouette. Choisissez celui qui reflète le meilleur score de silhouette.

# 5. **Analyse comparative :**

   - Comparez les performances des trois algorithmes sur le dataset sélectionné en utilisant les métriques appropriées.
   - Analysez comment les performances varient en fonction des paramètres choisis et de l'algorithme utilisé.
   - À votre avis, quel algorithme est le meilleur et pourquoi ? (Résistance au bruit ? Adaptabilité à la forme des données ?)
   - Discutez des avantages et des inconvénients de chaque méthode pour la détection de chutes.

# 6. **Interprétation des résultats :**
   - Pour chaque méthode et paramètres, interprétez les clusters obtenus. Peuvent-ils être associés à des chutes ou à des activités normales ?
   - Discutez de la pertinence de l'approche non supervisée pour la détection de chutes.

# 7. **Rapport et présentation :**
   - Rédigez un rapport détaillant la méthodologie, les résultats et les conclusions.
   - Préparez une présentation visuelle des résultats les plus significatifs.

### Bonus :
   - Explorez d'autres méthodes de réduction de dimensionnalité (t-SNE, UMAP) pour la visualisation des clusters.
   - Proposez et implémentez une méthode pour déterminer automatiquement le nombre optimal de clusters pour K-means et le clustering hiérarchique.

# Citations :

1. [Machine Learning Clustering DBSCAN](https://datascientest.com/machine-learning-clustering-dbscan)
  
3. [Kaggle - Machine Learning Non Supervisé](https://www.kaggle.com/code/zoupet/machine-learning-non-supervis-correction)
   
5. [Principaux Algorithmes d'Apprentissage Non Supervisé](https://fr.linedata.com/principaux-algorithmes-dapprentissage-non-supervise)
   
7. [Apprentissage Non Supervisé de Flux de Données Massives](https://www.researchgate.net/publication/333772967_Apprentissage_non_supervise_de_flux_de_donnees_massives_application_aux_Big_Data_d%27assurance)
   
9. [Discovery Unsupervised Learning](https://fr.mathworks.com/discovery/unsupervised-learning.html)

