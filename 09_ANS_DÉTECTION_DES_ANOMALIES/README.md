# https://drive.google.com/drive/folders/1kGx1bGp_TmMUm5hdzj0t0ILYvUY6MVln?usp=sharing

# Références :

- https://medium.com/analytics-vidhya/anomaly-detection-using-generative-adversarial-networks-gan-ca433f2ac287
- https://www.linkedin.com/pulse/generative-models-anomaly-detection-enhancing-efficiency-vaes-mudtc/
- https://www.nature.com/articles/s41598-024-52378-9
- https://medium.com/data-reply-it-datatech/detecting-the-unseen-anomaly-detection-with-gans-8b20f3056a11
- https://medium.com/simform-engineering/anomaly-detection-with-unsupervised-machine-learning-3bcf4c431aff

# ==> Équation 1
$$
d(x_i, c_j) = \sqrt{\sum_{k=1}^{n} (x_{ik} - c_{jk})^2}
$$



########################################################################################
# 01 - Explication de la Détection des Anomalies dans chaque Code Proposé
########################################################################################

#### 1. K-Means avec Largeur de Silhouette

**Code** :
```python
silhouette_values = silhouette_samples(X, labels)
anomalies = X[silhouette_values < 0]
```

- **silhouette_samples(X, labels)** : Cette fonction calcule le score de silhouette pour chaque point de données. Le score de silhouette mesure la qualité du clustering pour chaque point, en évaluant la cohésion (comment un point est proche des autres points de son cluster) et la séparation (comment un point est éloigné des points des autres clusters). Le score de silhouette varie de -1 à 1 :
  - **Score proche de 1** : Le point est bien assigné à son cluster.
  - **Score proche de 0** : Le point est sur ou très près de la frontière entre deux clusters.
  - **Score négatif** : Le point pourrait être mieux assigné à un autre cluster.
- **anomalies = X[silhouette_values < 0]** : Les points avec un score de silhouette négatif sont considérés comme des anomalies, car ils sont mal assignés à leur cluster actuel.

#### 2. DBSCAN

**Code** :
```python
labels_dbscan = dbscan.fit_predict(X)
anomalies = X[labels_dbscan == -1]
```

- **dbscan.fit_predict(X)** : Cette fonction exécute l'algorithme DBSCAN sur les données `X` et retourne les labels des clusters. Les paramètres principaux de DBSCAN sont :
  - **eps** : La distance maximale entre deux points pour qu'ils soient considérés comme voisins.
  - **min_samples** : Le nombre minimum de points pour former un cluster dense.
- **labels_dbscan == -1** : Les points marqués par le label `-1` sont considérés comme des anomalies par DBSCAN, car ils ne font partie d'aucun cluster dense (ils sont considérés comme du bruit).

#### 3. K-Means avec Distances aux Centres

**Code** :
```python
distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)
threshold = np.mean(distances) + 2 * np.std(distances)
anomalies = distances > threshold
```

- **distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)** : Cette ligne calcule la distance euclidienne entre chaque point et le centre de son cluster respectif. Mathématiquement, la distance euclidienne \( d \) entre un point \( x_i \) et le centre du cluster \( c_j \) est donnée par l'équation 1:

# ==> Équation 1:
$$
d(x_i, c_j) = \sqrt{\sum_{k=1}^{n} (x_{ik} - c_{jk})^2}
$$ 

ou: 
$$
x_{ik}
$$ 
 
 est la  **k**-ième dimension du point x_i, et c_{jk} est la \( k \)-ième dimension du centre de cluster \( c_j \).





#### 4. DBSCAN avec Données de Melbourne



- **model = DBSCAN(eps=0.63).fit(X1)** : Applique l'algorithme DBSCAN sur les données normalisées `X1` avec un paramètre `eps` de 0.63.
- **labels = model.labels_** : Les points marqués par le label `-1` sont considérés comme des anomalies, car ils ne font partie d'aucun cluster dense et sont donc traités comme du bruit.
- **labels = [('anomaly' if x == -1 else 'normal') for x in labels]** : Les anomalies sont étiquetées comme 'anomaly' si leur label est `-1`.




**Code** :
```python
from google.colab import drive

# Monter Google Drive pour accéder aux fichiers
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os

# Changer le répertoire de travail vers le chemin spécifié dans Google Drive
os.chdir('drive/My Drive/datacolab')

# Charger le dataset
df = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv')
# Afficher le nombre de lignes dans le dataset
print(len(df))
# Afficher les premières lignes du dataset
df.head()

# Sélectionner uniquement les colonnes numériques
df_num = df.select_dtypes(include=["float64", "int64"])

# Remplir les valeurs manquantes avec la médiane de chaque colonne
df_num.fillna(df_num.median(), inplace=True)

# Normaliser les données numériques pour le clustering
X = StandardScaler().fit_transform(df_num)
# Créer un DataFrame avec les données normalisées
X1 = pd.DataFrame(X, columns=df_num.columns)
# Afficher les premières lignes des données normalisées
X1.head()

# Fonction pour tracer les résultats du modèle DBSCAN
def plot_model(labels, alg_name, plot_index):
    fig = plt.figure(figsize=(15, 15))  # Créer une figure pour le tracé
    ax = fig.add_subplot(1, 1, plot_index)  # Ajouter un sous-graphique
    color_code = {'anomaly': 'red', 'normal': 'green'}  # Définir les couleurs pour les anomalies et les points normaux
    colors = [color_code[x] for x in labels]  # Assigner des couleurs basées sur les étiquettes

    # Tracé en nuage de points des données avec des couleurs basées sur leurs étiquettes
    ax.scatter(X1.iloc[:, 0], X1.iloc[:, 1], color=colors, marker='.', label='red = anomaly')
    ax.legend(loc="lower right")  # Ajouter une légende au graphique
    ax.set_title(alg_name)  # Définir le titre du graphique
    plt.show()  # Afficher le graphique

# Appliquer l'algorithme de clustering DBSCAN
model = DBSCAN(eps=0.63).fit(X1)
# Obtenir les étiquettes assignées par DBSCAN
labels = model.labels_

# Étiqueter les anomalies et les points normaux
labels = [('anomaly' if x == -1 else 'normal') for x in labels]

# Tracer les résultats
plot_model(labels, 'DBSCAN', 1)
```

- **labels = model.labels_** : Les points marqués par le label `-1` sont considérés comme des anomalies, car ils ne font partie d'aucun cluster dense et sont donc traités comme du bruit.
- **labels = [('anomaly' if x == -1 else 'normal') for x in labels]** : Les anomalies sont étiquetées comme 'anomaly' si leur label est `-1`.

#### 5. Isolation Forest


- **clf = IsolationForest(contamination=0.1)** : Crée et entraîne un modèle Isolation Forest pour détecter les anomalies dans les données en supposant que 10% des employés ont des comportements problématiques.
- **df['anomaly'] = clf.predict(df[features])** : Prédit les anomalies en utilisant le modèle entraîné. Les anomalies sont marquées par le label `-1`.
- **df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Problématique'})** : Les valeurs numériques des anomalies sont remplacées par des étiquettes lisibles.



**Code** :
```python
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Créer un jeu de données fictif
data = {
    'employee_id': range(1, 101),
    'num_complaints': [0, 1, 0, 2, 0, 20, 0, 0, 3, 0] * 10,
    'num_absences': [1, 0, 0, 0, 1, 20, 2, 1, 0, 0] * 10,
    'work_hours': [40, 38, 40, 35, 40, 42, 40, 39, 40, 40] * 10,
    'performance_score': [3, 4, 3, 2, 3, 1, 3, 4, 3, 3] * 10
}

# Convertir en DataFrame
df = pd.DataFrame(data)

# Afficher les premières lignes du DataFrame
print(df.head())

# Définir les fonctionnalités pour le modèle
features = ['num_complaints', 'num_absences', 'work_hours', 'performance_score']

# Appliquer l'IsolationForest pour détecter les anomalies
clf = IsolationForest(contamination=0.1)  # On suppose que 10% des employés ont des comportements problématiques
clf.fit(df[features])
df['anomaly'] = clf.predict(df[features])

# Remplacer les valeurs d'anomalie par des étiquettes lisibles
df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Problématique'})

# Compter et afficher le nombre d'employés normaux et problématiques
print(df['anomaly'].value_counts())

# Visualiser les résultats
normal = df[df['anomaly'] == 'Normal']
problematic = df[df['anomaly'] == 'Problématique']

plt.scatter(normal['num_complaints'], normal['num_absences'], label='Normal', c='green')
plt.scatter(problematic['num_complaints'], problematic['num_absences'], label='Problématique', c='red')
plt.xlabel('Nombre de Plaintes')
plt.ylabel('Nombre d\'Absences')
plt.title('Détection des Employés Problématiques')
plt.legend()
plt.show()

# Afficher les employés problématiques
print(df[df['anomaly'] == 'Problématique'])
```

- **clf.fit(df[features])** : Entraîne un modèle Isolation Forest pour détecter les anomalies dans les données.
- **df['anomaly'] = clf.predict(df[features])** : Prédit les anomalies en utilisant le modèle entraîné. Les anomalies sont marquées par le label `-1`.
- **df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Problématique'})** : Les valeurs numériques des anomalies sont remplacées par des étiquettes lisibles.

### Comparaison des Méthodes de Détection d'Anomalies

Chaque méthode de détection d'anomalies utilise une approche différente pour identifier les points de données aberrants dans un ensemble de données. Voici une comparaison détaillée des trois méthodes discutées : K-Means avec largeur de silhouette, DBSCAN, et K-Means avec distances aux centres.

### Tableau Récapitulatif des Méthodes

| **Critère**                      | **K-Means avec Largeur de Silhouette**                                                                                                                | **DBSCAN**                                                                                                  | **K-Means avec Distances aux Centres**                                                                                     |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **Algorithme**                   | Partitionnement des données en `k` clusters en minimisant la variance intra-cluster.                                                                    | Clustering basé sur la densité avec les paramètres `eps` et `min_samples`.                                  | Partitionnement des données en `k` clusters en minimisant la variance intra-cluster.                                         |
| **Détection d'anomalies**        | Utilisation du score de silhouette : les points avec une silhouette négative sont des anomalies.                                                       | Points marqués comme du bruit (label `-1`) sont des anomalies.                                               | Calcul de la distance euclidienne entre chaque point et le centre de son cluster : les points au-delà d'un certain seuil sont des anomalies. |
| **Mesure principale**            | Largeur de silhouette, score entre -1 et 1.                                                                                                            | Densité des points, label `-1` pour les anomalies.                                                          | Distance euclidienne aux centres de clusters, seuil défini comme la moyenne des distances plus deux écarts-types.                        |
| **Formule clé**                  | Silhouette : \(\frac{b - a}{\max(a, b)}\), où \(a\) est la distance moyenne entre un point et les autres points du même cluster, et \(b\) est la distance moyenne entre un point et les points du cluster le plus proche. | Distance entre les points et leurs voisins dans un rayon `eps`.                                             | Distance euclidienne : \( d(x_i, c_j) = \sqrt{\sum_{k=1}^{n} (x_{ik} - c_{jk})^2} \).                                        |
| **Avantages**                    | Simple à implémenter et à comprendre, fournit une mesure claire de la qualité du clustering.                                                            | Efficace pour des données avec des formes irrégulières et des densités variées, détecte automatiquement le bruit.  | Simple à implémenter et à comprendre, l'utilisation des distances offre une bonne indication des anomalies.                   |
| **Inconvénients**                | Fonctionne mieux avec des clusters sphériques et de taille similaire.                                                                                   | Peut être plus complexe à paramétrer, nécessite des choix appropriés pour `eps` et `min_samples`.           | Fonctionne mieux avec des clusters sphériques et de taille similaire, nécessite de définir un seuil approprié pour les distances. |

### Conclusion

Chacune de ces méthodes a ses propres forces et faiblesses en fonction du type de données et de la nature des anomalies recherchées. La méthode basée sur la largeur de silhouette est idéale pour des données avec des clusters bien définis, tandis que DBSCAN est plus adapté aux données avec des formes et des densités variées. L'approche utilisant les distances aux centres est une alternative simple et efficace pour les données avec des clusters sphériques. En comprenant les principes et les formules clés derrière chaque méthode, on peut choisir l'approche la plus appropriée pour une tâche spécifique de détection d'anomalies.


---

##########################################################################################
# 02 -  Comparaison des Méthodes de Détection d'Anomalies
##########################################################################################

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
==> **ÉQUATION 1** 
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
























### Conclusion

Chacune de ces méthodes a ses propres forces et faiblesses en fonction du type de données et de la nature des anomalies recherchées. La méthode basée sur la largeur de silhouette est idéale pour des données avec des clusters bien définis, tandis que DBSCAN est plus adapté aux données avec des formes et des densités variées. L'approche utilisant les distances aux centres est une alternative simple et efficace pour les données avec des clusters sphériques. Isolation Forest est efficace pour une grande variété de types de données, y compris celles avec des anomalies subtiles. En comprenant les principes et les formules clés derrière chaque méthode, on peut choisir l'approche la plus appropriée pour une tâche spécifique de détection d'anomalies.
