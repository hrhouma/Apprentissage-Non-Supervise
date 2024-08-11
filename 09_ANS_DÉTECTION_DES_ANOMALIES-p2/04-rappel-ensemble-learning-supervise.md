# (THÉORIE) 1 - Rappel - ensemble learning en  machine learning supervisé

L'ensemble learning (apprentissage par ensemble) est principalement une technique de machine learning supervisé, bien qu'il puisse également être adapté pour des approches non supervisées dans certains cas.

En machine learning supervisé, les méthodes d'ensemble combinent plusieurs modèles pour améliorer les performances globales. Voici quelques exemples populaires d'ensemble learning supervisé :

1. **Bagging (Bootstrap Aggregating)** :
   - Méthode : Utilise plusieurs échantillons aléatoires avec remplacement pour entraîner des modèles distincts.
   - Exemple : Random Forest, qui utilise des arbres de décision.

2. **Boosting** :
   - Méthode : Construit des modèles séquentiels où chaque modèle corrige les erreurs du précédent.
   - Exemple : AdaBoost, Gradient Boosting Machines (GBM), XGBoost.

3. **Stacking** :
   - Méthode : Combine les prédictions de plusieurs modèles de base à l'aide d'un modèle de niveau supérieur (métamodèle).
   - Exemple : Utiliser des régressions linéaires, SVMs, et des réseaux neuronaux comme modèles de base, et un autre modèle pour combiner leurs prédictions.

Cependant, il existe également des méthodes d'ensemble pour l'apprentissage non supervisé. 
- Par exemple, dans le clustering (regroupement) non supervisé, différentes méthodes de clustering peuvent être combinées pour obtenir des clusters plus stables et robustes.
- Un exemple d'ensemble learning non supervisé est l'utilisation de plusieurs algorithmes de clustering (comme K-means, DBSCAN, et l'agglomération hiérarchique) pour obtenir une meilleure compréhension des structures de données complexes en combinant les résultats des différents algorithmes.
- En résumé, l'ensemble learning est principalement utilisé en apprentissage supervisé pour améliorer la précision et la robustesse des modèles, mais il peut également être appliqué dans des contextes non supervisés, bien que cela soit moins courant.

---

# 2 - Exemple - Ensemble learning en supervisé

- Je vous présente un exemple concret d'ensemble learning supervisé, en utilisant deux des méthodes les plus populaires : **Bagging** avec **Random Forest** et **Boosting** avec **AdaBoost**. 
- Cet exemple va illustrer comment combiner plusieurs modèles pour améliorer la performance globale d'un modèle de classification.

### Exemple d'Ensemble Learning Supervisé : Classification des Iris

#### Contexte
Nous allons utiliser le jeu de données classique "Iris" pour effectuer une classification supervisée des différentes espèces de fleurs. Les caractéristiques incluent la longueur et la largeur des sépales et des pétales, et l'objectif est de prédire l'espèce de l'iris.

#### 1. **Préparation des Données**
   - Chargement du jeu de données et division en ensemble d'entraînement et de test.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger le jeu de données Iris
iris = load_iris()
X = iris.data
y = iris.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### (SUPERVISÉ) 2. **Bagging avec Random Forest**

   - **Random Forest** est une méthode d'ensemble basée sur le bagging, où plusieurs arbres de décision sont entraînés sur des sous-échantillons aléatoires des données.

```python
from sklearn.ensemble import RandomForestClassifier

# Créer un modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prédire les labels pour l'ensemble de test
rf_predictions = rf_model.predict(X_test)

# Évaluer la performance du modèle
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f'Random Forest Accuracy: {rf_accuracy:.2f}')
```

#### 3. **Boosting avec AdaBoost**

   - **AdaBoost** (Adaptive Boosting) est une méthode de boosting qui entraîne séquentiellement plusieurs modèles faibles (par exemple, des arbres de décision simples) et ajuste les poids des observations en fonction des erreurs des modèles précédents.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Créer un modèle AdaBoost
adaboost_model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1), 
    n_estimators=50, 
    random_state=42
)
adaboost_model.fit(X_train, y_train)

# Prédire les labels pour l'ensemble de test
adaboost_predictions = adaboost_model.predict(X_test)

# Évaluer la performance du modèle
adaboost_accuracy = accuracy_score(y_test, adaboost_predictions)
print(f'AdaBoost Accuracy: {adaboost_accuracy:.2f}')
```

#### 4. **Stacking (En Option)**
   - Vous pouvez aussi utiliser **stacking** pour combiner les prédictions des modèles Random Forest et AdaBoost avec un autre modèle (par exemple, une régression logistique) pour essayer d'améliorer encore les performances.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Créer un modèle de stacking
stacking_model = StackingClassifier(
    estimators=[('rf', rf_model), ('adaboost', adaboost_model)], 
    final_estimator=LogisticRegression()
)
stacking_model.fit(X_train, y_train)

# Prédire les labels pour l'ensemble de test
stacking_predictions = stacking_model.predict(X_test)

# Évaluer la performance du modèle
stacking_accuracy = accuracy_score(y_test, stacking_predictions)
print(f'Stacking Accuracy: {stacking_accuracy:.2f}')
```

#### 5. **Comparaison des Modèles**
   - Comparons les performances des différents modèles d'ensemble :

```python
print(f'Random Forest Accuracy: {rf_accuracy:.2f}')
print(f'AdaBoost Accuracy: {adaboost_accuracy:.2f}')
print(f'Stacking Accuracy: {stacking_accuracy:.2f}')
```

Cet exemple montre comment l'ensemble learning supervisé peut être utilisé pour améliorer les performances d'un modèle. **Random Forest** utilise le bagging pour créer une forêt d'arbres de décision robustes, tandis que **AdaBoost** utilise le boosting pour améliorer progressivement les prédictions. **Stacking** permet de combiner plusieurs modèles pour potentiellement améliorer encore la précision. Ces techniques d'ensemble learning sont couramment utilisées en pratique pour augmenter la performance des modèles de machine learning supervisé.

----

# 3 (NON SUPERVISÉ) - Exemple - Ensemble learning Non Supervisé : Clustering

#### Contexte
Imaginons que nous avons un ensemble de données contenant des informations sur des clients d'un magasin, et nous voulons segmenter ces clients en groupes distincts pour mieux comprendre leurs comportements. Nous utiliserons trois algorithmes de clustering : K-means, DBSCAN et l'agglomération hiérarchique, et nous combinerons leurs résultats pour obtenir un clustering final plus robuste.

#### Étapes de l'Ensemble Learning Non Supervisé

1. **Préparation des Données**
   - Normalisation des données pour s'assurer que toutes les caractéristiques sont sur la même échelle.

2. **Application des Algorithmes de Clustering**

   - **K-means**
     ```python
     from sklearn.cluster import KMeans
     kmeans = KMeans(n_clusters=5, random_state=42)
     kmeans_labels = kmeans.fit_predict(data)
     ```

   - **DBSCAN**
     ```python
     from sklearn.cluster import DBSCAN
     dbscan = DBSCAN(eps=0.5, min_samples=5)
     dbscan_labels = dbscan.fit_predict(data)
     ```

   - **Agglomération Hiérarchique**
     ```python
     from sklearn.cluster import AgglomerativeClustering
     agglo = AgglomerativeClustering(n_clusters=5)
     agglo_labels = agglo.fit_predict(data)
     ```

3. **Combinaison des Résultats de Clustering**

   - **Ensemble Learning par Vote Majoritaire**
     - On crée une matrice où chaque ligne représente un client et chaque colonne représente un algorithme de clustering.
     - On attribue à chaque client le cluster qui a le plus grand nombre de votes des différents algorithmes.

     ```python
     import numpy as np
     from scipy.stats import mode

     labels_matrix = np.vstack((kmeans_labels, dbscan_labels, agglo_labels)).T
     final_labels, _ = mode(labels_matrix, axis=1)
     final_labels = final_labels.flatten()
     ```

4. **Évaluation et Visualisation**
   - Visualisation des clusters obtenus pour vérifier la cohérence des segments créés.

   ```python
   import matplotlib.pyplot as plt
   from sklearn.decomposition import PCA

   pca = PCA(n_components=2)
   reduced_data = pca.fit_transform(data)

   plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=final_labels, cmap='viridis', marker='o')
   plt.title('Clusters par Ensemble Learning')
   plt.xlabel('PCA Component 1')
   plt.ylabel('PCA Component 2')
   plt.show()
   ```

#### Conclusion
Cet exemple montre comment on peut utiliser l'ensemble learning non supervisé pour combiner les résultats de plusieurs algorithmes de clustering. En combinant K-means, DBSCAN et l'agglomération hiérarchique, nous obtenons des segments de clients plus robustes et cohérents. Cette méthode permet de tirer parti des points forts de chaque algorithme et de compenser leurs faiblesses respectives.
