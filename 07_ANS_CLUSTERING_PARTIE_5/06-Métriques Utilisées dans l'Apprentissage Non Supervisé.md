# Métriques Utilisées dans l'Apprentissage Non Supervisé - Partie # 3

L'apprentissage non supervisé est une méthode d'analyse des données qui permet de découvrir des motifs cachés dans des données non étiquetées. Les métriques sont essentielles pour évaluer la qualité des clusters trouvés par des algorithmes comme K-means, DBSCAN, et l'analyse en composantes principales (PCA). Ce cours explore les principales métriques utilisées pour évaluer les algorithmes d'apprentissage non supervisé, avec des formules mathématiques et des exemples vulgarisés.

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

## Objectifs du Cours

- Comprendre l'importance des métriques dans l'apprentissage non supervisé.
- Découvrir les différentes métriques utilisées pour évaluer les clusters.
- Apprendre à interpréter les résultats de ces métriques.
- Appliquer ces métriques dans des exemples pratiques avec des algorithmes.

## Contenu du Cours

### 1. Introduction aux Métriques de Clustering

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

