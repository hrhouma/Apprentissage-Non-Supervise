# Table des Matières
1. [Formule 1: Z-Score](#formule-1-z-score)
2. [Formule 2: Intervalle Interquartile (IQR)](#formule-2-intervalle-interquartile-iqr)
3. [Formule 3: Distance Euclidienne](#formule-3-distance-euclidienne)
4. [Formule 4: Distance de Manhattan](#formule-4-distance-de-manhattan)
5. [Formule 5: Distance de Mahalanobis](#formule-5-distance-de-mahalanobis)
6. [Section 1: Introduction à la détection d'anomalies](#section-1-introduction-à-la-détection-danomalies)
7. [Section 2: Comprendre les anomalies](#section-2-comprendre-les-anomalies)
8. [Section 3: Exercice Pratique](#section-3-exercice-pratique)
9. [Section 4: Détection des Outliers](#section-4-détection-des-outliers)
10. [Section 5: Algorithmes Non Basés sur le Clustering](#section-5-algorithmes-non-basés-sur-le-clustering)
11. [Section 6: Explication des Résultats de la Détection d'Anomalies dans un Contexte de Fraude](#section-6-explication-des-résultats-de-la-détection-danomalies-dans-un-contexte-de-fraude)
12. [Section 7: Comparaison des Algorithmes de Détection d'Anomalies avec PyOD](#section-7-comparaison-des-algorithmes-de-détection-danomalies-avec-pyod)

---

# FORMULES 

---

## Formule 1: Z-Score
<a name="formule-1-z-score"></a>

$$
Z = \frac{(X - \mu)}{\sigma}
$$

où \(Z\) est le Z-score, \(X\) est la valeur du point de données, \(\mu\) est la moyenne de l'ensemble de données, et \(\sigma\) est l'écart-type.

**Explication :** Le Z-score permet de standardiser les données pour les rendre comparables, même si elles proviennent de différentes distributions. Un Z-score de 2 indique, par exemple, que le point de données est à deux écarts-types de la moyenne.

[Retour en haut](#table-des-matières)

---

## Formule 2: Intervalle Interquartile (IQR)
<a name="formule-2-intervalle-interquartile-iqr"></a>

$$
IQR = Q3 - Q1
$$

où \(Q3\) est le troisième quartile (75e percentile) et \(Q1\) est le premier quartile (25e percentile).

**Explication :** L'IQR mesure la dispersion des valeurs médianes d'un ensemble de données. Il aide à identifier les outliers en définissant les bornes pour les valeurs normales.

[Retour en haut](#table-des-matières)

---

## Formule 3: Distance Euclidienne
<a name="formule-3-distance-euclidienne"></a>

$$
d = \sqrt{\sum_{i=1}^{n}(X_i - Y_i)^2}
$$

où \(d\) est la distance Euclidienne entre les points \(X\) et \(Y\).

**Explication :** La distance Euclidienne mesure la distance directe entre deux points dans un espace multidimensionnel, couramment utilisée dans les algorithmes de clustering.

[Retour en haut](#table-des-matières)

---

## Formule 4: Distance de Manhattan
<a name="formule-4-distance-de-manhattan"></a>

$$
d = \sum_{i=1}^{n} |X_i - Y_i|
$$

où \(d\) est la distance de Manhattan entre les points \(X\) et \(Y\).

**Explication :** La distance de Manhattan mesure la somme des différences absolues entre les points, utile dans les situations où le déplacement suit un chemin rectiligne, comme dans un réseau routier.

[Retour en haut](#table-des-matières)

---

## Formule 5: Distance de Mahalanobis
<a name="formule-5-distance-de-mahalanobis"></a>

$$
d_M = \sqrt{(X - \mu)^T \Sigma^{-1} (X - \mu)}
$$

où \(\mu\) est le vecteur de la moyenne et \(\Sigma\) est la matrice de covariance.

**Explication :** La distance de Mahalanobis prend en compte la corrélation entre les variables pour évaluer la distance, utile pour identifier les outliers dans des données multidimensionnelles.

[Retour en haut](#table-des-matières)

---

# Section 1: Introduction à la détection d'anomalies
<a name="section-1-introduction-à-la-détection-danomalies"></a>

- La détection d'anomalies est essentielle dans divers secteurs, tels que la finance, la santé, et l'industrie pour prévenir des erreurs ou des événements critiques.
- Une anomalie est une déviation significative par rapport à un comportement attendu, ce qui peut signaler une potentielle fraude, panne ou autre événement inhabituel.
- Types d'anomalies :
  - **Ponctuelles :** Un seul point de données dévie des autres.
  - **Contextuelles :** Anomalie dépendante du contexte (ex : saison).
  - **Collectives :** Un groupe de points dévie ensemble.
- Applications :
  - **Finance :** Détection des transactions frauduleuses.
  - **Santé :** Identification des signes précurseurs de maladies graves.
  - **Industrie :** Maintenance prédictive.

[Retour en haut](#table-des-matières)

---

# Section 2: Comprendre les anomalies
<a name="section-2-comprendre-les-anomalies"></a>

- Une anomalie est un point de données qui ne correspond pas aux autres, souvent signe d'un problème ou d'un événement unique.
- Types principaux d'anomalies :
  - **Basées sur le temps :** Les anomalies dépendent du temps, comme des pics de consommation d'énergie.
  - **Non basées sur le temps :** Anomalies indépendantes du temps, comme une maison vendue à un prix anormalement élevé.
  - **Anomalies d'image :** Objets ou formes inhabituels détectés dans des images.
- Méthodes de détection :
  - **Supervisée :** Utilise des données étiquetées pour identifier les anomalies.
  - **Non supervisée :** Apprend les caractéristiques normales des données et identifie les déviations.

[Retour en haut](#table-des-matières)

---

# Section 3: Exercice Pratique
<a name="section-3-exercice-pratique"></a>

**Objectif :** Identifier des anomalies dans divers contextes.

**Questions :**
1. Définissez ce qu'est une anomalie.
2. Donnez des exemples d'anomalies basées sur le temps.
3. Donnez des exemples d'anomalies non basées sur le temps.
4. Donnez des exemples d'anomalies d'image.

**Exemple :**
- **Définition :** Une anomalie est un événement ou une observation qui dévie significativement des autres points dans un ensemble de données.
- **Exemples :**
  - **Basée sur le temps :** Un pic soudain dans la consommation d'énergie.
  - **Non basée sur le temps :** Une transaction bancaire anormalement élevée.

[Retour en haut](#table-des-matières)

---

# Section 4 : Détection des Outliers
<a name="section-4-détection-des-outliers"></a>

## Méthode du Z-Score

- **Z-Score :** Utilisé pour identifier les points de données situés plusieurs écarts-types au-dessus ou en dessous de la moyenne.
- **Exemple :** Identifier les transactions anormalement élevées dans un ensemble de données financières.

## Méthode de l'Intervalle Interquartile (IQR)

- **IQR :** Mesure la dispersion des 50 % de données centrales pour identifier les outliers.
- **Exemple :** Détecter les valeurs extrêmes de factures dans une entreprise.

## Méthode Basée sur la Distance

- **Distance Euclidienne et de Manhattan :** Mesurent la similarité ou la différence entre les points de données.
- **Distance de Mahalanobis :** Prend en compte la covariance entre les variables.
- **Exemple :** Identifier les points de données éloignés de la moyenne dans des données multidimensionnelles.

## Méthode DBSCAN

- **DBSCAN :** Algorithme de clustering qui identifie les points dans les régions à faible densité comme des outliers.
- **Exemple :** Identifier les anomalies dans un ensemble de données de factures en fonction de leur densité locale.

**Exercice Pratique :**
1. Utilisez la méthode du Z-Score pour identifier les outliers.
2. Appliquez l'IQR pour identifier les outliers.
3. Utilisez les distances Euclidienne, Manhattan, et Mahalanobis

 pour détecter les outliers.

[Retour en haut](#table-des-matières)

---

# Section 5 : Algorithmes Non Basés sur le Clustering
<a name="section-5-algorithmes-non-basés-sur-le-clustering"></a>

## Introduction

- Différences entre algorithmes de clustering et non clustering.
- Les algorithmes non clustering évaluent chaque point de données individuellement, en fonction de propriétés statistiques locales.

## Isolation Forest

- **Isolation Forest :** Partitions les données pour isoler rapidement les anomalies.
- **Exemple :** Identifier les transactions bancaires anormales.

## Histogram-Based Outlier Score (HBOS)

- **HBOS :** Utilise des histogrammes pour détecter les anomalies en fonction de leur densité.
- **Exemple :** Détecter les anomalies dans les données de ventes.

## Approche Hybride

- Combinaison des avantages des algorithmes de clustering et non clustering pour une détection plus robuste.

**Exercice Pratique :**
1. Appliquez Isolation Forest pour détecter les anomalies.
2. Utilisez HBOS pour détecter les anomalies.
3. Combinez les méthodes pour une détection hybride.

[Retour en haut](#table-des-matières)

---

### Section 6 : Explication des Résultats de la Détection d'Anomalies dans un Contexte de Fraude
<a name="section-6-explication-des-résultats-de-la-détection-danomalies-dans-un-contexte-de-fraude"></a>

#### Contexte : Détection de Fraude dans l'Assurance Automobile

- Importance de l'explication des anomalies dans un contexte sensible comme la fraude.
- **SHAP :** Algorithme pour expliquer les décisions des modèles de détection d'anomalies.

#### Construction du Modèle : Isolation Forest

- **Isolation Forest :** Utilisé pour identifier les anomalies dans les données de fraude.
- **Exemple :** Analyse des transactions frauduleuses dans les assurances.

#### Explication des Anomalies avec SHAP

- SHAP permet d'identifier les facteurs critiques qui ont conduit à l'identification des anomalies.
- **Exemple :** Expliquer pourquoi une transaction a été identifiée comme une anomalie en utilisant SHAP.

#### Identification des Facteurs Critiques à l'Échelle Globale

- Utilisation de SHAP pour identifier les principaux facteurs influençant les anomalies dans l'ensemble des données.
- **Application Pratique :** Mettre en place des contrôles préventifs basés sur les résultats de SHAP.

### Conclusion de la partie 6

- L'importance de la transparence dans la détection d'anomalies et l'utilisation d'outils comme SHAP pour justifier les résultats et améliorer la prévention des fraudes.

[Retour en haut](#table-des-matières)

---

### Section 7 : Comparaison des Algorithmes de Détection d'Anomalies avec PyOD
<a name="section-7-comparaison-des-algorithmes-de-détection-danomalies-avec-pyod"></a>

#### Introduction à PyOD et Contexte de l'Exercice

- **PyOD :** Une bibliothèque pour la détection d'anomalies dans les données multivariées, avec plus de 30 algorithmes différents.
- Application de plusieurs algorithmes sur un jeu de données pour évaluer leur performance.

#### Application des Algorithmes de Détection d'Anomalies

- Comparaison des performances de différents algorithmes sur un jeu de données d'assurances santé.
- Analyse des résultats pour identifier le meilleur algorithme pour ce contexte.

#### Précision et Évaluation des Modèles

- Importance d'évaluer la précision des modèles pour les anomalies et les non-anomalies.
- **Exemple :** Évaluer l'impact des fausses alertes dans le contexte des assurances.

#### Implémentation du Code et Résultats

- Prétraitement des données, application des algorithmes, et analyse des résultats.
- **Exemple de Code :** Utilisation de PyOD pour appliquer différents algorithmes de détection d'anomalies.

#### Conclusion et Recommandations

- Recommandations sur l'utilisation des algorithmes de PyOD et l'importance de tester les modèles avant de les déployer dans un contexte commercial.

[Retour en haut](#table-des-matières)


