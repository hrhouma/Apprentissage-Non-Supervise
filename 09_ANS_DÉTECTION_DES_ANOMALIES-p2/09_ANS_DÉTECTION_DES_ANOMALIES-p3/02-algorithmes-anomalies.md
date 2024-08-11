----
# 1 - Table récapitulative des principaux algorithmes de détection d'anomalies
----

| **Algorithme**              | **Description**                                                                                                                                      | **Avantages**                                                                                   | **Inconvénients**                                                                                |
|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| **Isolation Forest**        | Algorithme basé sur des arbres de décision qui isole les observations en construisant des arbres de manière aléatoire.                               | - Efficace pour les grands datasets<br>- Capable de détecter des anomalies multidimensionnelles | - Sensible au choix des hyperparamètres<br>- Moins performant sur les anomalies denses          |
| **Local Outlier Factor (LOF)** | Mesure la densité locale de chaque point de données pour identifier les points qui sont significativement moins denses que leurs voisins.          | - Bon pour détecter des anomalies locales<br>- Adapté aux datasets avec des clusters           | - Sensible au choix du nombre de voisins<br>- Calcul intensif pour de grands datasets           |
| **DBSCAN**                  | Algorithme de clustering qui identifie les points denses et considère les points éloignés comme des anomalies.                                        | - Ne nécessite pas de spécifier le nombre de clusters<br>- Capable de détecter des formes arbitraires de clusters | - Performance dépendante des paramètres epsilon et min_samples<br>- Pas efficace pour des données de haute dimension |
| **One-Class SVM**           | Algorithme basé sur les machines à vecteurs de support, utilisé pour apprendre une frontière de décision pour les données normales.                  | - Bonne performance sur des données de haute dimension<br>- Capable de détecter des anomalies non linéaires | - Sensible au choix des hyperparamètres<br>- Moins efficace pour les grands datasets           |
| **K-means**                 | Algorithme de clustering qui peut être utilisé pour identifier des points qui ne se rapprochent pas de n'importe quel cluster comme des anomalies.   | - Simple à implémenter<br>- Efficace pour les grandes données avec des clusters bien séparés   | - Nécessite de spécifier le nombre de clusters<br>- Pas efficace pour les clusters de formes non sphériques |
| **Elliptic Envelope**       | Modèle statistique qui suppose que les données normales suivent une distribution gaussienne multivariée et identifie les anomalies comme des points éloignés. | - Simple à implémenter<br>- Bonne performance pour des données gaussiennes                     | - Hypothèse de distribution gaussienne peut ne pas être valable<br>- Pas efficace pour des distributions non gaussiennes |
| **Autoencoders**            | Réseaux de neurones utilisés pour la réduction de dimensionnalité, où les anomalies sont identifiées par une erreur de reconstruction élevée.        | - Capable de capturer des relations complexes<br>- Adapté aux données de haute dimension       | - Nécessite beaucoup de données et de puissance de calcul<br>- Complexe à entraîner             |

### Explications supplémentaires

- **Isolation Forest** : Fonctionne en isolant les observations en utilisant des arbres de décision. Les anomalies sont isolées plus rapidement, ce qui les distingue des observations normales.

- **Local Outlier Factor (LOF)** : Compare la densité locale de chaque point avec celle de ses voisins. Les anomalies sont les points ayant une densité significativement plus faible que leurs voisins.

- **DBSCAN** : Algorithme de clustering qui identifie des régions denses et considère les points éloignés de ces régions comme des anomalies. Les anomalies sont des points non atteignables de n'importe quel cluster dense.

- **One-Class SVM** : Algorithme d'apprentissage supervisé qui apprend une frontière de décision pour séparer les données normales des anomalies. Il est efficace pour les données non linéaires.

- **K-means** : Utilisé pour le clustering, les points éloignés de tous les clusters peuvent être considérés comme des anomalies. Il est simple mais nécessite de connaître le nombre de clusters à l'avance.

- **Elliptic Envelope** : Assume que les données normales suivent une distribution gaussienne multivariée et identifie les points éloignés de cette distribution comme des anomalies.

- **Autoencoders** : Réseaux de neurones utilisés pour apprendre une représentation compressée des données. Les anomalies sont identifiées par une erreur de reconstruction élevée, indiquant que le modèle n'a pas bien appris les caractéristiques de ces points.

Ces algorithmes sont souvent utilisés en fonction du type et des caractéristiques des données, ainsi que des exigences spécifiques de l'application.

----
# 02 - Quand utiliser ? Quand ne pas utiliser ? 
----

- Pour choisir le bon algorithme de détection d'anomalies, il est important de comprendre les caractéristiques de vos données et les spécificités de chaque algorithme. 
- Je vous propose un guide pour savoir quand appliquer chaque algorithme et quand éviter de les utiliser.

### Isolation Forest

**Quand l'appliquer :**
- Pour des datasets de grande taille.
- Lorsque les anomalies sont rares et clairement distinctes des données normales.
- Pour des anomalies multidimensionnelles.

**Quand ne pas l'appliquer :**
- Pour des datasets très petits.
- Si les anomalies ne sont pas clairement distinctes des données normales.

### Local Outlier Factor (LOF)

**Quand l'appliquer :**
- Lorsque les anomalies sont définies par leur densité locale.
- Pour des datasets avec des clusters locaux de différentes densités.
- Pour des anomalies locales (anomalies relatives à leur voisinage).

**Quand ne pas l'appliquer :**
- Pour des grands datasets, car LOF peut être computationnellement coûteux.
- Si vous avez des données de haute dimension.

### DBSCAN

**Quand l'appliquer :**
- Pour détecter des anomalies dans des clusters de formes arbitraires.
- Lorsque vous ne souhaitez pas spécifier le nombre de clusters à l'avance.
- Pour des données à densité variable.

**Quand ne pas l'appliquer :**
- Si vous avez des données de haute dimension.
- Si vous ne pouvez pas déterminer les bons paramètres pour epsilon et min_samples.

### One-Class SVM

**Quand l'appliquer :**
- Pour des données de haute dimension.
- Lorsque vous avez des anomalies non linéaires.
- Lorsque vous avez besoin d'une approche basée sur des vecteurs de support.

**Quand ne pas l'appliquer :**
- Pour des datasets très grands (peut être lent à entraîner).
- Si vous avez des données avec peu de dimensions et que des méthodes plus simples suffisent.

### K-means

**Quand l'appliquer :**
- Pour des données avec des clusters bien séparés.
- Lorsque vous connaissez le nombre de clusters à l'avance.
- Pour des datasets de taille moyenne.

**Quand ne pas l'appliquer :**
- Si les anomalies sont dans des clusters de formes non sphériques.
- Pour des données de haute dimension avec des clusters de densité variable.

### Elliptic Envelope

**Quand l'appliquer :**
- Lorsque les données suivent une distribution gaussienne multivariée.
- Pour des anomalies univariées ou multivariées simples.

**Quand ne pas l'appliquer :**
- Si les données ne suivent pas une distribution gaussienne.
- Pour des données de forme complexe ou distribution non gaussienne.

### Autoencoders

**Quand l'appliquer :**
- Pour des données de haute dimension ou complexes.
- Lorsque vous avez une grande quantité de données pour entraîner le modèle.
- Pour des anomalies non linéaires.

**Quand ne pas l'appliquer :**
- Pour des datasets de petite taille (nécessite beaucoup de données).
- Si vous manquez de puissance de calcul ou de temps pour entraîner un modèle complexe.

### Tableau récapitulatif

| **Algorithme**              | **Quand l'appliquer**                                                                                                                                      | **Quand ne pas l'appliquer**                                                                                          |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **Isolation Forest**        | - Grands datasets<br>- Anomalies rares et distinctes<br>- Anomalies multidimensionnelles                                                                     | - Datasets petits<br>- Anomalies non distinctes                                                                      |
| **Local Outlier Factor (LOF)** | - Anomalies locales<br>- Datasets avec clusters locaux de différentes densités                                                                             | - Grands datasets<br>- Données de haute dimension                                                                     |
| **DBSCAN**                  | - Clusters de formes arbitraires<br>- Densité variable<br>- Pas besoin de spécifier le nombre de clusters                                                    | - Données de haute dimension<br>- Paramètres epsilon et min_samples difficiles à déterminer                           |
| **One-Class SVM**           | - Données de haute dimension<br>- Anomalies non linéaires<br>- Approche basée sur des vecteurs de support                                                    | - Grands datasets<br>- Données avec peu de dimensions                                                                 |
| **K-means**                 | - Clusters bien séparés<br>- Connaissance du nombre de clusters<br>- Datasets de taille moyenne                                                              | - Clusters de formes non sphériques<br>- Données de haute dimension avec densité variable                             |
| **Elliptic Envelope**       | - Distribution gaussienne multivariée<br>- Anomalies univariées ou multivariées simples                                                                     | - Distribution non gaussienne<br>- Forme complexe de données                                                          |
| **Autoencoders**            | - Données de haute dimension<br>- Grande quantité de données pour l'entraînement<br>- Anomalies non linéaires                                                | - Datasets petits<br>- Manque de puissance de calcul ou de temps pour entraîner le modèle                             |

- Ce tableau devrait vous aider à choisir le bon algorithme de détection d'anomalies en fonction des caractéristiques spécifiques de vos données et des besoins de votre analyse.
