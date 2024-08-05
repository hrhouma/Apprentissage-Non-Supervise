### Comparaison des Méthodes de Détection d'Anomalies

# ÉQUATION 1 : 
  $$
  d(x_i, c_j) = \sqrt{\sum_{k=1}^{n} (x_{ik} - c_{jk})^2}
  $$


# ÉQUATION 2 : 
$$ \frac{b - a}{\max(a, b)} $$


# ÉQUATION 3 : 
  $$ d(x_i, c_j) = \sqrt{\sum_{k=1}^{n} (x_{ik} - c_{jk})^2} $$

Chaque méthode de détection d'anomalies utilise une approche différente pour identifier les points de données aberrants dans un ensemble de données. Voici une comparaison détaillée des trois méthodes discutées : K-Means avec largeur de silhouette, DBSCAN, et K-Means avec distances aux centres.

#### 1. K-Means avec Largeur de Silhouette

L'algorithme K-Means partitionne les données en `k` clusters en minimisant la variance intra-cluster. Pour détecter les anomalies, on utilise la largeur de silhouette.

- **silhouette_samples(X, labels)** : Cette fonction calcule le score de silhouette pour chaque point de données. Le score de silhouette mesure la qualité du clustering pour chaque point en évaluant la cohésion et la séparation :
  - **Score proche de 1** : Le point est bien assigné à son cluster.
  - **Score proche de 0** : Le point est sur ou très près de la frontière entre deux clusters.
  - **Score négatif** : Le point pourrait être mieux assigné à un autre cluster.
- **anomalies = X[silhouette_values < 0]** : Les points avec un score de silhouette négatif sont considérés comme des anomalies car ils sont mal assignés à leur cluster actuel.

#### 2. DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) est un algorithme de clustering basé sur la densité qui forme des clusters en regroupant des points proches les uns des autres.

- **dbscan.fit_predict(X)** : Cette fonction exécute l'algorithme DBSCAN sur les données `X` et retourne les labels des clusters. Les points de bruit sont marqués avec le label `-1`.
- **anomalies = X[labels_dbscan == -1]** : Les points marqués par le label `-1` sont considérés comme des anomalies, car ils ne font partie d'aucun cluster dense et sont donc traités comme du bruit.

#### 3. K-Means avec Distances aux Centres

Cette méthode utilise également l'algorithme K-Means, mais détecte les anomalies en calculant la distance entre chaque point et le centre de son cluster.

- **distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)** : Cette ligne calcule la distance euclidienne entre chaque point et le centre de son cluster respectif. Mathématiquement, la distance euclidienne \( d \) entre un point \( x_i \) et le centre du cluster \( c_j \) est donnée par :
==> ÉQUATION 1 
  où \( x_{ik} \) est la \( k \)-ième dimension du point \( x_i \), et \( c_{jk} \) est la \( k \)-ième dimension du centre \( c_j \).
- **threshold = np.mean(distances) + 2 * np.std(distances)** : Un seuil est défini comme la moyenne des distances plus deux écarts-types.
- **anomalies = distances > threshold** : Les points dont la distance dépasse ce seuil sont considérés comme des anomalies.

### Tableau Récapitulatif des Méthodes

| **Critère**                      | **K-Means avec Largeur de Silhouette**                                                                                                                | **DBSCAN**                                                                                                  | **K-Means avec Distances aux Centres**                                                                                     |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **Algorithme**                   | Partitionnement des données en `k` clusters en minimisant la variance intra-cluster.                                                                    | Clustering basé sur la densité avec les paramètres `eps` et `min_samples`.                                  | Partitionnement des données en `k` clusters en minimisant la variance intra-cluster.                                         |
| **Détection d'anomalies**        | Utilisation du score de silhouette : les points avec une silhouette négative sont des anomalies.                                                       | Points marqués comme du bruit (label `-1`) sont des anomalies.                                               | Calcul de la distance euclidienne entre chaque point et le centre de son cluster : les points au-delà d'un certain seuil sont des anomalies. |
| **Mesure principale**            | Largeur de silhouette, score entre -1 et 1.                                                                                                            | Densité des points, label `-1` pour les anomalies.                                                          | Distance euclidienne aux centres de clusters, seuil défini comme la moyenne des distances plus deux écarts-types.                        |
| **Formule clé**                  | Silhouette : ==> ÉQUATION 2, où \(a\) est la distance moyenne entre un point et les autres points du même cluster, et \(b\) est la distance moyenne entre un point et les points du cluster le plus proche. | Distance entre les points et leurs voisins dans un rayon `eps`.                                             | Distance euclidienne : ==> ÉQUATION 3.                                        |
| **Avantages**                    | Simple à implémenter et à comprendre, fournit une mesure claire de la qualité du clustering.                                                            | Efficace pour des données avec des formes irrégulières et des densités variées, détecte automatiquement le bruit.  | Simple à implémenter et à comprendre, l'utilisation des distances offre une bonne indication des anomalies.                   |
| **Inconvénients**                | Fonctionne mieux avec des clusters sphériques et de taille similaire.                                                                                   | Peut être plus complexe à paramétrer, nécessite des choix appropriés pour `eps` et `min_samples`.           | Fonctionne mieux avec des clusters sphériques et de taille similaire, nécessite de définir un seuil approprié pour les distances. |

### Conclusion

Chacune de ces méthodes a ses propres forces et faiblesses en fonction du type de données et de la nature des anomalies recherchées. La méthode basée sur la largeur de silhouette est idéale pour des données avec des clusters bien définis, tandis que DBSCAN est plus adapté aux données avec des formes et des densités variées. L'approche utilisant les distances aux centres est une alternative simple et efficace pour les données avec des clusters sphériques. En comprenant les principes et les formules clés derrière chaque méthode, on peut choisir l'approche la plus appropriée pour une tâche spécifique de détection d'anomalies.
