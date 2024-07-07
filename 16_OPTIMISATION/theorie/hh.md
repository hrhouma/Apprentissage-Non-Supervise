# Table des Matières

## [1. Métriques Utilisées dans l'Apprentissage Non Supervisé - Partie #1 (sans Math)](#1-métriques-utilisées-dans-lapprentissage-non-supervisé---partie-1-sans-math)
## [2. Métriques Utilisées dans l'Apprentissage Non Supervisé - Partie #2 (avec Math)](#2-métriques-utilisées-dans-lapprentissage-non-supervisé---partie-2-avec-math)
## [3. Métriques Utilisées dans l'Apprentissage Non Supervisé - Partie #3 (Quand utiliser et quand ne pas utiliser)](#3-métriques-utilisées-dans-lapprentissage-non-supervisé---partie-3-quand-utiliser-et-quand-ne-pas-utiliser)
## [4. Récapitulation sur les métriques Utilisées dans l'Apprentissage Non Supervisé - Partie #4 (Tableau Comparatif)](#4-récapitulation-sur-les-métriques-utilisées-dans-lapprentissage-non-supervisé---partie-4-tableau-comparatif)
## [5. Récapitulation sur les métriques Utilisées dans l'Apprentissage Non Supervisé - Partie #5 (avec Math)](#5-récapitulation-sur-les-métriques-utilisées-dans-lapprentissage-non-supervisé---partie-5-avec-math)

----
# 1- Métriques Utilisées dans l'Apprentissage Non Supervisé - Partie #1 (sans Math)

## [Revenir en haut](#Table-des-Matières)

## 1.1 - Score de Silhouette
Le score de silhouette est une mesure utilisée pour évaluer la qualité des clusters formés par un algorithme de clustering. Chaque point de données reçoit un score de silhouette basé sur deux critères : la cohésion et la séparation. La cohésion mesure à quel point un point est proche des autres points dans le même cluster, tandis que la séparation mesure la distance entre ce point et les points dans les clusters voisins. Un score de silhouette varie de -1 à 1, où un score proche de 1 indique que les points sont bien groupés dans leurs clusters respectifs et bien séparés des autres clusters. Un score proche de 0 indique que les points sont à la frontière des clusters, et un score négatif signifie que les points sont probablement dans le mauvais cluster.

## 1.2 -  Indice de Davies-Bouldin
L'indice de Davies-Bouldin évalue la qualité du clustering en comparant la moyenne des dispersions intra-cluster à la séparation inter-cluster. La dispersion intra-cluster est une mesure de la distance moyenne entre les points de données et le centre de leur cluster respectif, tandis que la séparation inter-cluster mesure la distance entre les centres de clusters. Un indice de Davies-Bouldin faible indique que les clusters sont compacts et bien séparés les uns des autres, suggérant un bon clustering. Cette mesure aide à identifier si les clusters sont bien définis, ce qui est essentiel pour des applications où la distinction claire entre groupes est cruciale.

## 1.3 -  Cohésion et Séparation
La cohésion et la séparation sont deux critères clés pour évaluer la qualité des clusters. La cohésion, ou intra-cluster distance, mesure à quel point les points de données dans un même cluster sont proches les uns des autres. Une bonne cohésion signifie que les membres d'un cluster sont très similaires entre eux, ce qui est souhaitable. La séparation, ou inter-cluster distance, mesure la distance entre les différents clusters. Une bonne séparation signifie que les clusters sont bien distincts les uns des autres. Ensemble, une bonne cohésion et une bonne séparation indiquent que les clusters sont bien formés, ce qui est crucial pour des analyses significatives et interprétables.

## 1.4 - Indice de Rand Ajusté (ARI)
L'indice de Rand ajusté (ARI) est une mesure de la similarité entre deux partitions d'un ensemble de données, généralement une partition réelle et une partition obtenue par un algorithme de clustering. ARI tient compte de toutes les paires de points de données et compare combien de paires sont assignées de manière cohérente dans les deux partitions. Une valeur élevée d'ARI signifie que les clusters obtenus sont très similaires aux clusters attendus ou réels. Contrairement à d'autres indices, l'ARI est ajusté pour la probabilité de correspondances par hasard, offrant ainsi une évaluation plus robuste de la qualité du clustering.

## 1.5 -  Normalized Mutual Information (NMI)
La Normalized Mutual Information (NMI) est une mesure utilisée pour comparer deux partitions d'un ensemble de données en termes d'information partagée. Elle évalue combien d'information sur l'une des partitions est contenue dans l'autre, et vice versa. Une NMI élevée indique que les deux partitions partagent beaucoup d'information, ce qui signifie qu'elles sont similaires. NMI est particulièrement utile pour évaluer des algorithmes de clustering où les partitions peuvent avoir des tailles différentes et où l'information partagée doit être normalisée pour fournir une comparaison équitable.

## 1.6 -  Courbe d'Inertie pour K-means
La courbe d'inertie est un outil graphique utilisé pour déterminer le nombre optimal de clusters dans l'algorithme K-means. L'inertie mesure la somme des distances au carré entre chaque point de données et le centre de son cluster. En traçant l'inertie en fonction du nombre de clusters, on peut observer comment l'inertie diminue à mesure que le nombre de clusters augmente. Le "coude" de la courbe indique le point optimal où ajouter plus de clusters n'améliore plus significativement l'inertie, suggérant ainsi le nombre de clusters à utiliser. Cette approche aide à équilibrer la précision du clustering et la complexité du modèle.


----
# 2 - Métriques Utilisées dans l'Apprentissage Non Supervisé - Partie#2 (avec Math)
## [Revenir en haut](#Table-des-Matières)
# 2.1 -  Score de Silhouette
Le score de silhouette est une mesure utilisée pour évaluer la qualité des clusters formés par un algorithme de clustering. Chaque point de données reçoit un score de silhouette basé sur deux critères : la cohésion et la séparation. La cohésion mesure à quel point un point est proche des autres points dans le même cluster, tandis que la séparation mesure la distance entre ce point et les points dans les clusters voisins.

Le score de silhouette *s(i)* pour un point *i* est défini comme :

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

où :
- *a(i)* est la distance moyenne entre *i* et tous les autres points du même cluster.
- *b(i)* est la distance moyenne entre *i* et tous les points du cluster le plus proche.

Un score de silhouette varie de -1 à 1, où un score proche de 1 indique que les points sont bien groupés dans leurs clusters respectifs et bien séparés des autres clusters. Un score proche de 0 indique que les points sont à la frontière des clusters, et un score négatif signifie que les points sont probablement dans le mauvais cluster.

# 2.2 - Indice de Davies-Bouldin
L'indice de Davies-Bouldin (DBI) évalue la qualité du clustering en comparant la moyenne des dispersions intra-cluster à la séparation inter-cluster. Il est défini comme :

$$
DBI = \frac{1}{k} \sum_{i=1}^{k} \max_{j \ne i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)
$$

où :
- *k* est le nombre de clusters.
- *sigma_i* est la dispersion intra-cluster pour le cluster *i*.
- *d(c_i, c_j)* est la distance entre les centres des clusters *i* et *j*.

Un indice de Davies-Bouldin faible indique que les clusters sont compacts et bien séparés les uns des autres, suggérant un bon clustering.

# 2.3 - Cohésion et Séparation
La cohésion *a(i)* et la séparation *b(i)* sont deux critères clés pour évaluer la qualité des clusters :
- La cohésion, ou intra-cluster distance, mesure à quel point les points de données dans un même cluster sont proches les uns des autres.
- La séparation, ou inter-cluster distance, mesure la distance entre les différents clusters.

# 2.4 - Indice de Rand Ajusté (ARI)
L'indice de Rand ajusté (ARI) est une mesure de la similarité entre deux partitions d'un ensemble de données. Il est défini comme :

$$
ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2}}{0.5 \left[ \sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2} \right] - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2}}
$$

où *n_{ij}* est le nombre de points dans les clusters *i* et *j*, *a(i)*  et *b(i)* sont les sommes des lignes et des colonnes respectivement.

# 2.5 - Normalized Mutual Information (NMI)
La Normalized Mutual Information (NMI) est une mesure utilisée pour comparer deux partitions d'un ensemble de données. Elle est définie comme :

$$
NMI(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}
$$

où :
- *I(U; V)* est l'information mutuelle entre les partitions *U* et *V*.
- *H(U)* et *H(V)* sont les entropies des partitions *U* et *V*.

# 2.6 - Courbe d'Inertie pour K-means
La courbe d'inertie est un outil graphique utilisé pour déterminer le nombre optimal de clusters dans l'algorithme K-means. L'inertie (\( WCSS \)) est définie comme la somme des distances au carré entre chaque point de données et le centre de son cluster :

$$
WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} \| x - \mu_i \|^2
$$

En traçant l'inertie en fonction du nombre de clusters, on peut observer comment l'inertie diminue à mesure que le nombre de clusters augmente. Le "coude" de la courbe indique le point optimal où ajouter plus de clusters n'améliore plus significativement l'inertie.

----
# 3 - Métriques Utilisées dans l'Apprentissage Non Supervisé - Partie #3 (Quand utiliser et quand ne pas utiliser)

## Score de Silhouette

**Utilisation :**
Le score de silhouette est utilisé pour évaluer la qualité du clustering après avoir appliqué un algorithme de clustering, comme K-means ou DBSCAN. Il est particulièrement utile lorsque vous avez plusieurs solutions de clustering et que vous souhaitez choisir la meilleure.

**Quand l'utiliser :**
- Pour comparer différents résultats de clustering et choisir le plus approprié.
- Lorsqu'on veut comprendre à quel point les clusters sont distincts les uns des autres.
- Pour identifier des points de données mal classés ou situés à la frontière des clusters.

**Quand ne pas l'utiliser :**
- Pour des données avec des clusters très imbriqués ou de formes irrégulières, où le score de silhouette peut ne pas être fiable.
- Lorsque le nombre de clusters est extrêmement grand, ce qui peut rendre l'interprétation du score de silhouette difficile.

**Avantages :**
- Fournit une mesure intuitive et visuelle de la qualité du clustering.
- Facile à interpréter : un score élevé est bon, un score faible est mauvais.
- Permet d'identifier des points de données mal classés.

**Inconvénients :**
- Peut être biaisé pour les clusters de formes non sphériques ou de tailles très différentes.
- Sensible aux valeurs aberrantes.
- Le calcul peut être coûteux en temps pour de très grands ensembles de données.

## Indice de Davies-Bouldin

**Utilisation :**
L'indice de Davies-Bouldin est utilisé pour évaluer la qualité des clusters après un clustering. Il compare la dispersion intra-cluster avec la séparation inter-cluster.

**Quand l'utiliser :**
- Pour comparer différents résultats de clustering de manière quantitative.
- Lorsqu'on souhaite une mesure simple et rapide à calculer.

**Quand ne pas l'utiliser :**
- Pour des ensembles de données où les clusters sont de formes très irrégulières ou de densités variées.
- Lorsque l'évaluation qualitative est plus importante que quantitative.

**Avantages :**
- Simple à comprendre et à calculer.
- Prend en compte la compacité et la séparation des clusters.
- Peut être utilisé pour évaluer différents algorithmes de clustering.

**Inconvénients :**
- Sensible aux valeurs aberrantes.
- Peut ne pas bien fonctionner avec des clusters de formes irrégulières.
- La mesure repose sur les centres de clusters, ce qui peut être biaisé pour des clusters de tailles inégales.

## Cohésion et Séparation
## [Revenir en haut](#Table-des-Matières)
**Utilisation :**
Ces mesures sont utilisées pour évaluer la compacité des clusters (cohésion) et la distinction entre eux (séparation).

**Quand l'utiliser :**
- Pour obtenir une vision détaillée de la structure des clusters.
- Lorsqu'on veut optimiser la formation de clusters en termes de compacité et de séparation.

**Quand ne pas l'utiliser :**
- Lorsque les clusters sont très imbriqués ou présentent des formes complexes.
- Pour des ensembles de données très volumineux où le calcul de ces mesures peut être coûteux.

**Avantages :**
- Fournit une analyse détaillée de la structure des clusters.
- Utile pour affiner et améliorer les algorithmes de clustering.

**Inconvénients :**
- Les calculs peuvent être coûteux pour de grands ensembles de données.
- Peut être difficile à interpréter sans visualisation.
- Sensible aux valeurs aberrantes et aux variations de densité.

## Indice de Rand Ajusté (ARI)

**Utilisation :**
L'ARI est utilisé pour comparer la similarité entre deux partitions de données, souvent une partition obtenue et une partition de référence.

**Quand l'utiliser :**
- Pour évaluer la performance d'un algorithme de clustering par rapport à une vérité terrain.
- Lorsqu'on compare plusieurs résultats de clustering.

**Quand ne pas l'utiliser :**
- Pour des ensembles de données où aucune vérité terrain n'est disponible.
- Lorsque les partitions sont très déséquilibrées.

**Avantages :**
- Ajusté pour les correspondances aléatoires, offrant une évaluation plus robuste.
- Facile à interpréter : un ARI élevé indique une bonne correspondance.

**Inconvénients :**
- Peut être biaisé pour des clusters très déséquilibrés.
- Nécessite une partition de référence pour la comparaison.

## Normalized Mutual Information (NMI)

**Utilisation :**
La NMI est utilisée pour comparer deux partitions de données en termes d'information partagée.

**Quand l'utiliser :**
- Pour évaluer la qualité du clustering par rapport à une partition de référence.
- Lorsqu'on compare plusieurs solutions de clustering.

**Quand ne pas l'utiliser :**
- Pour des ensembles de données où la normalisation peut induire en erreur.
- Lorsque les partitions à comparer sont très déséquilibrées.

**Avantages :**
- Mesure basée sur l'information, donc robuste pour différentes tailles de clusters.
- Normalisée, permettant une comparaison équitable entre différentes partitions.

**Inconvénients :**
- Peut être difficile à interpréter sans une compréhension de la théorie de l'information.
- Peut être biaisée par des distributions très déséquilibrées.

# Courbe d'Inertie pour K-means

**Utilisation :**
La courbe d'inertie est utilisée pour déterminer le nombre optimal de clusters en K-means.

**Quand l'utiliser :**
- Lorsqu'on doit déterminer le nombre optimal de clusters pour K-means.
- Pour visualiser la diminution de l'inertie avec l'augmentation du nombre de clusters.

**Quand ne pas l'utiliser :**
- Pour des algorithmes de clustering autres que K-means.
- Lorsque les clusters attendus ne sont pas sphériques.

**Avantages :**
- Visuel et intuitif, aide à identifier le "coude" de la courbe.
- Simple à calculer et à interpréter.

**Inconvénients :**
- Peut ne pas bien fonctionner pour des clusters non sphériques.
- La détermination du "coude" peut être subjective.

----
# 4 - Récapitulation sur les métriques Utilisées dans l'Apprentissage Non Supervisé - Partie#4 (Tableau Comparatif)
## [Revenir en haut](#Table-des-Matières)

| Critère                    | Score de Silhouette | Indice de Davies-Bouldin | Cohésion et Séparation   | ARI                  | NMI                  | Courbe d'Inertie    |
|----------------------------|---------------------|--------------------------|--------------------------|----------------------|----------------------|---------------------|
| **Utilisation Principale** | Évaluer la qualité  | Évaluer la qualité       | Évaluer structure        | Comparer partitions  | Comparer partitions  | Déterminer clusters |
| **Avantages**              | Intuitif, visuel    | Simple à calculer        | Analyse détaillée        | Ajusté pour hasard   | Robuste              | Visuel et intuitif  |
| **Inconvénients**          | Biaisé pour formes irrégulières | Sensible aux valeurs aberrantes | Coûteux en calculs     | Biaisé pour partitions déséquilibrées | Difficile à interpréter | Subjectif          |
| **Quand l'utiliser**       | Comparer résultats  | Comparer résultats       | Optimiser formation      | Évaluer performance  | Évaluer qualité      | Déterminer K optimal|
| **Quand ne pas l'utiliser**| Clusters imbriqués  | Clusters irréguliers     | Données volumineuses     | Pas de référence     | Partitions déséquilibrées | Clusters non sphériques |



# 5 - Récapitulation sur les métriques Utilisées dans l'Apprentissage Non Supervisé - Partie#5 (avec Math)
## [Revenir en haut](#Table-des-Matières)
L'apprentissage non supervisé est une méthode d'analyse des données qui permet de découvrir des motifs cachés dans des données non étiquetées. Les métriques sont essentielles pour évaluer la qualité des clusters trouvés par des algorithmes comme K-means, DBSCAN, et l'analyse en composantes principales (PCA). Cette partie rappel et récapitule les principales métriques utilisées pour évaluer les algorithmes d'apprentissage non supervisé, avec des formules mathématiques et des exemples vulgarisés.

## Équations

### Équation #1
$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

### Équation #2
$$
DB = \frac{1}{n} \sum_{i=1}^{n} \max_{j \neq i} \left( \frac{s_i + s_j}{d(c_i, c_j)} \right)
$$

### Équation #3
$$
\text{Cohésion} = \sum_{i=1}^{n} \sum_{x \in C_i} \| x - \mu_i \|^2
$$

### Équation #4
$$
ARI = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}
$$

### Équation #5
$$
NMI = \frac{2 \cdot I(U; V)}{H(U) + H(V)}
$$

### Équation #6
$$
\text{Inertie} = \sum_{i=1}^{k} \sum_{x \in C_i} \| x - \mu_i \|^2
$$



### 1. Rappel sur les Métriques de Clustering

- **Définition et importance des métriques** : Les métriques permettent de quantifier la qualité des clusters formés par les algorithmes d'apprentissage non supervisé.
- **Types de métriques** : 
  - **Métriques internes** : Évaluent la qualité des clusters en utilisant uniquement les données et la partition trouvée.
  - **Métriques externes** : Comparent les clusters obtenus avec une vérité terrain (labels connus).
  - **Métriques relatives** : Comparent différentes partitions ou résultats obtenus avec des paramètres différents.

### 2. Métriques Internes

#### 2.1 Indice de Silhouette

- **Définition** : Mesure de la compacité et de la séparation des clusters.
- **Formule** : Voir Équation #1
- **Interprétation** : Valeur entre -1 et 1. Plus proche de 1 signifie de bons clusters, proche de -1 signifie de mauvais clusters.
- **Exemple Vulgarisé** : Imaginez que vous organisez une fête et que vous essayez de diviser vos invités en groupes pour qu'ils aient des conversations intéressantes. L'indice de silhouette vous aide à voir si chaque invité est dans le bon groupe ou non.

#### 2.2 Indice de Davies-Bouldin

- **Définition** : Ratio de la distance intra-cluster à la distance inter-cluster.
- **Formule** : Voir Équation #2
- **Interprétation** : Plus la valeur est basse, meilleure est la séparation des clusters.
- **Exemple Vulgarisé** : L'indice de Davies-Bouldin mesure à quel point chaque groupe d'invités est serré et à quel point les groupes sont éloignés les uns des autres.

#### 2.3 Cohésion et Séparation

- **Cohésion** : Mesure de la compacité des clusters (somme des distances intra-cluster).
- **Formule** : Voir Équation #3
- **Séparation** : Mesure de la distance entre les clusters (somme des distances inter-cluster).
- **Formule** : Voir Équation #3
- **Exemple Vulgarisé** : Pensez à la cohésion comme à la proximité des invités dans chaque groupe. Si les invités sont proches les uns des autres, la cohésion est élevée. La séparation est la distance entre les différents groupes.

### 3. Métriques Externes

#### 3.1 Indice de Rand Ajusté

- **Définition** : Mesure de la similarité entre deux partitions, ajustée pour les chances.
- **Formule** : Voir Équation #4
- **Interprétation** : Valeurs entre -1 et 1. 1 signifie une correspondance parfaite, 0 signifie une correspondance aléatoire.
- **Exemple Vulgarisé** : Supposons que vous avez divisé les invités en groupes en fonction de leurs intérêts, et vous comparez votre division avec la liste de préférences des invités.

#### 3.2 NMI (Normalized Mutual Information)

- **Définition** : Mesure de l'information partagée entre les clusters trouvés et les vrais clusters.
- **Formule** : Voir Équation #5
- **Interprétation** : Valeurs entre 0 et 1. 1 signifie une correspondance parfaite.
- **Exemple Vulgarisé** : La NMI mesure combien d'information est partagée entre votre division des invités et leurs préférences réelles.

### 4. Métriques Relatives

#### 4.1 Courbe d'Inertie pour K-means

- **Définition** : Somme des distances au carré des points au centre du cluster.
- **Formule** : Voir Équation #6
- **Utilisation** : Déterminer le nombre optimal de clusters (méthode de l'épaule).
- **Exemple Vulgarisé** : La courbe d'inertie montre comment l'ajout de plus de groupes réduit la distance moyenne entre les invités et le centre de leur groupe.

#### 4.2 Validation Croisée en Clustering

- **Définition** : Évaluer la stabilité des clusters en utilisant des sous-échantillons des données.
- **Application** : Comparaison de partitions pour différents algorithmes de clustering.
- **Exemple Vulgarisé** : La validation croisée vous aide à voir si les groupes sont cohérents d'une méthode à l'autre.


----
### 5. Exemple Pratique : Évaluation des Algorithmes de Clustering

#### Cas d'Étude : Données Iris

- **Application de K-means, DBSCAN, et PCA**
- **Calcul des différentes métriques** : Indice de Silhouette, Indice de Davies-Bouldin, NMI
- **Interprétation des résultats**

#### Cas d'Étude : Données Clients

- **Segmentation des clients avec K-means**
- **Évaluation des clusters avec des métriques internes et externes**
- **Ajustement des paramètres et réévaluation**

### Conclusion

- **Récapitulatif des principales métriques**
- **Importance de choisir les bonnes métriques en fonction des objectifs de l'analyse**
- **Conseils pour l'interprétation et l'application des résultats dans des projets réels**


