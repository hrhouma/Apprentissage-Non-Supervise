
# FORMULES 


---
$$
  Z = \frac{(X - \mu)}{\sigma}
$$
  où \(Z\) est le Z-score, \(X\) est la valeur du point de données, \(\mu\) est la moyenne de l'ensemble de données, et \(\sigma\) est l'écart-type.

---
$$
  IQR = Q3 - Q1
$$


---
$$
  d = \sqrt{\sum_{i=1}^{n}(X_i - Y_i)^2}
$$
  où \(d\) est la distance Euclidienne entre les points \(X\) et \(Y\).

---
$$
  d = \sum_{i=1}^{n} |X_i - Y_i|
$$
  où \(d\) est la distance de Manhattan entre les points \(X\) et \(Y\).
---
$$
  d_M = \sqrt{(X - \mu)^T \Sigma^{-1} (X - \mu)}
$$
  où \(\mu\) est le vecteur de la moyenne et \(\Sigma\) est la matrice de covariance.

---
# Section 1: Introduction à la détection d'anomalies

- La détection d'anomalies est un domaine de plus en plus adopté dans l'industrie.
- Au cœur de ce concept, une anomalie est quelque chose qui ne correspond pas au reste des données ou des transactions.
- Ce principe est largement applicable dans différents secteurs, tels que l'industrie manufacturière sous forme de maintenance prédictive, ainsi que dans la détection de fraudes.
- Cette présentation commence par une compréhension des anomalies et des différents types d'anomalies.
- La présentation est structurée pour introduire les trois types de détection d'anomalies et les domaines d'application associés.
- Nous aborderons ensuite le codage pertinent pour les types de détection en apprentissage non supervisé.
- Les données et les codes utilisés dans cette présentation sont disponibles dans le dossier ressources, vous permettant ainsi de les télécharger et de pratiquer en même temps.

 # Section 2: Comprendre les anomalies

- Avant de plonger dans les algorithmes, il est essentiel de comprendre ce que sont les anomalies.
- Une anomalie peut être un ou plusieurs points de données qui ne s'intègrent pas au reste des données.
- Identifier des anomalies dans un petit ensemble de données est généralement facile, mais cela devient difficile avec des ensembles de données plus volumineux.
- Il existe trois types principaux d'anomalies :
  - **Anomalies basées sur le temps** : Les points de données sont dépendants du temps. Par exemple, le prix de l'essence à différents jours d'un mois.
  - **Anomalies non basées sur le temps** : Les points de données ne sont pas dépendants du temps. Par exemple, le prix d'un appartement dans une ville donnée en fonction de plusieurs facteurs.
  - **Anomalies d'image** : Détecter des anomalies dans un ensemble d'images, comme pour les données classiques.
- Les anomalies peuvent être détectées à l'aide d'algorithmes supervisés ou non supervisés :
  - **Supervisé** : Lorsque des références passées d'anomalies sont disponibles.
  - **Non supervisé** : Lorsque ces références ne sont pas disponibles, ce qui est souvent le cas en entreprise.
- Les algorithmes non supervisés sont souvent préférés en raison de l'absence de données robustes. Ils peuvent être basés sur des clusters ou non.
- Exemples de jeux de données :
  - **Jeux de données basés sur le temps** : Par exemple, les transactions d'un client bancaire à différents moments.
  - **Jeux de données non basés sur le temps** : Par exemple, le prix d'un appartement en fonction de différents facteurs sans référence au temps.
- La détection d'anomalies basée sur le temps est particulièrement utile pour identifier des événements rares mais critiques, comme la maintenance prédictive dans l'industrie.
- **Concept d'outliers (valeurs aberrantes)** :
  - Les outliers sont des points de données extrêmes.
  - Chaque outlier est une anomalie, mais toutes les anomalies ne sont pas des outliers.
  - Techniques de détection des outliers :
    - **Box plot** : Utilisation de l'intervalle interquartile.
    - **Diagramme de contrôle** : Utilisation de la moyenne et de l'écart-type pour définir des limites de contrôle.
- En résumé, bien que chaque outlier soit une anomalie, l'inverse n'est pas toujours vrai.


# Section 3: Exercice Pratique

**Instructions de l'exercice**
- Temps alloué : 30 minutes
- Nombre de solutions étudiantes : 61

Dans cet exercice, les apprenants sont invités à fournir des exemples d'application des anomalies et de la détection d'anomalies dans la vie réelle. Cet exercice les aidera dans leur parcours d'apprentissage, car ils pourront se référer à ces exemples tout au long de la présentation et évaluer l'importance de la détection d'anomalies dans des situations réelles.

**Questions pour cet exercice :**
1. Décrivez ce qu'est une anomalie avec vos propres mots.
2. Donnez des exemples d'anomalies basées sur le temps. Les exemples peuvent provenir de votre domaine professionnel ou d'un domaine que vous connaissez bien.
3. Donnez des exemples d'anomalies non basées sur le temps (non supervisées). Les exemples peuvent provenir de votre domaine professionnel ou d'un domaine que vous connaissez bien.
4. Donnez des exemples d'anomalies non basées sur le temps (supervisées). Les exemples peuvent provenir de votre domaine professionnel ou d'un domaine que vous connaissez bien.
5. Donnez des exemples d'anomalies basées sur les images. Les exemples peuvent provenir de votre domaine professionnel ou d'un domaine que vous connaissez bien.

---

**Exemple de l'instructeur :**

1. **Décrivez ce qu'est une anomalie avec vos propres mots.**
   - Une anomalie est un point de données qui ne correspond pas ou ne s'intègre pas bien avec les autres points de données.

2. **Donnez des exemples d'anomalies basées sur le temps.**
   - <Il s'agit d'une question basée sur l'application des concepts. Il n'y a donc pas de réponse standard. Les apprenants doivent répondre en fonction de leur compréhension.>

3. **Donnez des exemples d'anomalies non basées sur le temps (non supervisées).**
   - <Il s'agit d'une question basée sur l'application des concepts. Il n'y a donc pas de réponse standard. Les apprenants doivent répondre en fonction de leur compréhension.>

4. **Donnez des exemples d'anomalies non basées sur le temps (supervisées).**
   - <Il s'agit d'une question basée sur l'application des concepts. Il n'y a donc pas de réponse standard. Les apprenants doivent répondre en fonction de leur compréhension.>

5. **Donnez des exemples d'anomalies basées sur les images.**
   - <Il s'agit d'une question basée sur l'application des concepts. Il n'y a donc pas de réponse standard. Les apprenants doivent répondre en fonction de leur compréhension.>


Voici la version révisée de la Section 4 avec toutes les formules et explications nécessaires :

---

# Section 4 : Détection des Outliers

Dans cette section, nous allons explorer différentes méthodes pour détecter les outliers, c'est-à-dire des points de données extrêmes qui se situent nettement en dehors de la majorité d'un ensemble de données ou d'un cluster. La détection des outliers est cruciale pour une analyse statistique précise et la performance des modèles.

## Méthode du Z-Score

La méthode du Z-score est une technique statistique utilisée pour identifier les outliers en mesurant le nombre d'écarts-types d'un point de données par rapport à la moyenne de l'ensemble de données.

# Voir FORMULE 1

- **Interprétation :** Les points de données avec un Z-score supérieur à 3 ou inférieur à -3 sont considérés comme des outliers car ils se situent bien en dehors de la plage normale des données. Parfois, une valeur seuil de 2 au lieu de 3 est également utilisée.

- **Exemple :** Prenons un ensemble de données comprenant des identifiants de factures et leurs montants respectifs. En calculant le Z-score pour chaque montant, les valeurs avec un Z-score supérieur à 3 ou inférieur à -3 sont marquées comme outliers.

- **Programmation :** En Python, cela peut être accompli en utilisant les bibliothèques `numpy` et `pandas`. Vous calculez d'abord la moyenne et l'écart-type, puis vous appliquez la formule du Z-score pour chaque point de données.

## Méthode de l'Intervalle Interquartile (IQR)

La méthode de l'Intervalle Interquartile (IQR) identifie les outliers en mesurant l'étendue des 50 % de données centrales. L'IQR est calculé comme la différence entre le troisième quartile (\(Q3\)) et le premier quartile (\(Q1\)).

# Voir FORMULE 2

  où \(Q3\) est le troisième quartile (75e percentile) et \(Q1\) est le premier quartile (25e percentile).

- **Interprétation :** Les valeurs situées en dessous de \(Q1 - 1.5 \times IQR\) ou au-dessus de \(Q3 + 1.5 \times IQR\) sont considérées comme des outliers.

- **Exemple :** Dans notre ensemble de données de factures, les montants situés en dehors de ces bornes (par exemple, en dessous de 77.5 ou au-dessus de 105.5) sont identifiés comme des outliers.

- **Programmation :** Vous pouvez calculer les quantiles avec la méthode `quantile()` de `pandas`, puis identifier les outliers en appliquant la formule de l'IQR.

## Méthode Basée sur la Distance

La méthode basée sur la distance identifie les outliers en mesurant la distance entre les points de données et la moyenne de l'ensemble de données. Il existe plusieurs types de distances couramment utilisées :

- **Distance Euclidienne :**  

# Voir FORMULE 3

- **Distance de Manhattan :**  

# Voir FORMULE 4

- **Distance de Mahalanobis :**  

# Voir FORMULE 5

- **Exemple :** Dans l'ensemble de données de factures, vous pouvez utiliser ces différentes distances pour identifier les points de données qui sont significativement éloignés de la moyenne, et donc les marquer comme outliers.

- **Programmation :** Utilisez `scipy.spatial.distance` pour calculer ces distances en Python.

## Méthode DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) est un algorithme de clustering qui regroupe les points de données proches tout en identifiant les points dans les régions à faible densité comme des outliers.

- **Paramètres :** L'algorithme nécessite deux paramètres :
  - \(\epsilon\) : la distance maximale pour qu'un point soit considéré comme voisin d'un autre.
  - \(minPts\) : le nombre minimal de points requis pour former une région dense.

- **Types de points :**
  - **Point de cœur :** Un point avec au moins \(minPts\) voisins dans son voisinage défini par \(\epsilon\).
  - **Point de bordure :** Un point dans le voisinage d'un point de cœur mais avec moins de \(minPts\) voisins.
  - **Point de bruit :** Un point qui n'est ni un point de cœur ni un point de bordure, et qui est donc considéré comme un outlier.

- **Exemple :** En appliquant DBSCAN à un ensemble de données de factures, les montants isolés peuvent être identifiés comme des outliers en fonction de leur densité locale.

- **Programmation :** Utilisez `DBSCAN` de `scikit-learn` pour appliquer cet algorithme à vos données.

## Exercice Pratique

### Instructions de l'exercice
1. **Z-Score Method :** Déterminez le nombre d'outliers en utilisant cette méthode.
2. **IQR Method :** Utilisez la méthode de l'IQR pour identifier les outliers.
3. **Distance Method :** Appliquez les distances Euclidienne, Manhattan et Mahalanobis pour détecter les outliers.
4. **DBSCAN :** Ajustez les paramètres \(\epsilon\) et \(minPts\) pour identifier les outliers et observez les résultats.

