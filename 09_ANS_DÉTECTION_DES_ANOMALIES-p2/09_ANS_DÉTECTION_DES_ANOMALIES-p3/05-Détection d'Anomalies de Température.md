#  Détection d'Anomalies de Température avec des Algorithmes d'Apprentissage Non Supervisé

---

#### Table des Matières

1. [Introduction](#introduction)
2. [Problématique](#problématique)
3. [Contexte et Motivation](#contexte-et-motivation)
4. [Présentation des Algorithmes Utilisés](#présentation-des-algorithmes-utilisés)
   - 4.1 [Isolation Forest](#isolation-forest)
   - 4.2 [DBSCAN](#dbscan)
   - 4.3 [Autoencodeurs](#autoencodeurs)
   - 4.4 [One-Class SVM](#one-class-svm)
   - 4.5 [Clustering Hiérarchique](#clustering-hiérarchique)
5. [Exemples d'Implémentation](#exemples-dimplémentation)
   - 5.1 [Exemple d'Isolation Forest](#exemple-disolation-forest)
   - 5.2 [Exemple de DBSCAN](#exemple-de-dbscan)
   - 5.3 [Exemple d'Autoencodeur](#exemple-dautoencodeur)
6. [Analyse et Comparaison des Résultats](#analyse-et-comparaison-des-résultats)
7. [Conclusion](#conclusion)
8. [Références et Ressources Complémentaires](#références-et-ressources-complémentaires)

---

#### Introduction

- La détection d'anomalies est un domaine crucial dans la surveillance des systèmes, notamment lorsqu'il s'agit de détecter des événements climatiques extrêmes tels que des variations anormales de température. 
- Cet exercice se concentre sur l'application d'algorithmes d'apprentissage non supervisé pour identifier ces anomalies. 
- L'objectif principal est de développer un système capable de signaler des dépassements de seuil de température, ce qui pourrait indiquer des situations potentiellement dangereuses ou inhabituelles.

---

#### Problématique

**Question:**
- Comment peut-on utiliser des algorithmes d'apprentissage non supervisé pour détecter efficacement des événements de température qui dépassent un seuil prédéfini, en minimisant les faux positifs tout en garantissant une détection rapide et précise des anomalies ?

---

#### Contexte et Motivation

- Les systèmes de surveillance climatique sont essentiels pour anticiper et réagir aux événements météorologiques extrêmes.
- La température, en particulier, est un indicateur clé de changements climatiques soudains. Dépasser un seuil de température pourrait indiquer des événements tels que des vagues de chaleur, des incendies de forêt, ou des défaillances de systèmes de refroidissement dans des installations industrielles.
- Un système automatisé de détection d'anomalies pourrait permettre une réaction plus rapide et plus efficace à ces événements.

- L'apprentissage non supervisé est particulièrement adapté à ce type de tâche car il permet de détecter des anomalies sans nécessiter de données étiquetées, ce qui est souvent difficile à obtenir dans les systèmes de surveillance en temps réel.

---

#### Présentation des Algorithmes Utilisés

- Cette section détaille les algorithmes d'apprentissage non supervisé utilisés dans cet exercice.
- Chaque algorithme est présenté avec une explication théorique, un exemple d'implémentation, et une discussion sur les avantages et inconvénients dans le contexte de la détection d'anomalies de température.

---

##### 4.1 Isolation Forest

###### Théorie
- Isolation Forest est un algorithme de détection d'anomalies qui fonctionne sur le principe d'isolation des points de données.
- Contrairement aux méthodes traditionnelles qui modélisent les points normaux, Isolation Forest isole les points anormaux en les partitionnant de manière répétée jusqu'à ce qu'ils soient isolés des autres données.
- L'idée sous-jacente est que les anomalies sont des points rares et distincts, et nécessitent donc moins de partitions pour être isolées.

###### Implémentation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Génération de données de température
np.random.seed(42)
temperatures = np.random.normal(loc=20, scale=5, size=1000)
# Ajout de valeurs anormales
temperatures = np.append(temperatures, [50, 55, 60, 65, 70])

# Conversion en DataFrame
df = pd.DataFrame(temperatures, columns=['Temperature'])

# Modèle Isolation Forest
model = IsolationForest(contamination=0.01)  # 1% d'anomalies attendues
df['anomaly'] = model.fit_predict(df[['Temperature']])

# Identification des anomalies
anomalies = df[df['anomaly'] == -1]
print(anomalies)
```

###### Avantages
- **Simplicité** : L'algorithme est facile à comprendre et à implémenter.
- **Efficacité** : Capable de détecter des anomalies même avec un grand nombre de dimensions.
- **Scalabilité** : Peut être utilisé sur de grandes quantités de données.

###### Inconvénients
- **Sensibilité aux hyperparamètres** : Le choix du seuil de contamination peut grandement influencer les résultats.
- **Assumptions linéaires** : L'algorithme peut ne pas bien fonctionner pour des données très non-linéaires.

---

##### 4.2 DBSCAN

###### Théorie
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise) est un algorithme de clustering basé sur la densité qui peut également être utilisé pour détecter des anomalies.
- L'idée principale est de regrouper les points qui sont proches les uns des autres en fonction d'un critère de densité.
- Les points qui ne font pas partie d'un cluster dense sont considérés comme des anomalies.

###### Implémentation

```python
from sklearn.cluster import DBSCAN

# Paramètres DBSCAN
epsilon = 3  # Distance maximale entre deux points pour qu'ils soient considérés comme voisins
min_samples = 5  # Nombre minimum de points pour former un cluster dense

# Application de DBSCAN
model = DBSCAN(eps=epsilon, min_samples=min_samples)
df['cluster'] = model.fit_predict(df[['Temperature']])

# Identification des anomalies (points non assignés à un cluster)
anomalies = df[df['cluster'] == -1]
print(anomalies)
```

###### Avantages
- **Non-linéarité** : Capable de détecter des clusters de formes arbitraires, ce qui est utile pour des données non-linéaires.
- **Résistant au bruit** : Peut ignorer les points bruyants qui ne font pas partie de clusters significatifs.

###### Inconvénients
- **Scalabilité limitée** : Peut ne pas bien fonctionner avec des ensembles de données très grands.
- **Sensibilité aux paramètres** : La performance dépend beaucoup des choix de `epsilon` et `min_samples`.

---

##### 4.3 Autoencodeurs

###### Théorie
- Les autoencodeurs sont des types de réseaux de neurones utilisés pour l'apprentissage non supervisé, notamment pour la réduction de dimensionnalité et la détection d'anomalies.
- Un autoencodeur apprend à reconstruire ses entrées, et l'erreur de reconstruction est utilisée pour identifier les anomalies.
- Les anomalies sont des points pour lesquels l'autoencodeur a une haute erreur de reconstruction, indiquant qu'ils diffèrent significativement des données normales.

###### Implémentation

```python
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Normalisation des données
df['Temperature'] = (df['Temperature'] - df['Temperature'].mean()) / df['Temperature'].std()

# Définition de l'autoencodeur
input_dim = df.shape[1]
encoding_dim = 2  # Nombre de dimensions après encodage

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement de l'autoencodeur
autoencoder.fit(df[['Temperature']], df[['Temperature']], epochs=50, batch_size=10, shuffle=True, validation_split=0.2)

# Détection des anomalies
df['reconstruction_error'] = np.mean(np.abs(df[['Temperature']] - autoencoder.predict(df[['Temperature']])), axis=1)
anomalies = df[df['reconstruction_error'] > df['reconstruction_error'].quantile(0.99)]
print(anomalies)
```

###### Avantages
- **Puissance** : Peut capturer des relations complexes dans les données.
- **Flexibilité** : Peut être adapté pour différents types de données et de structures.

###### Inconvénients
- **Complexité** : Nécessite des ressources computationnelles importantes pour l'entraînement.
- **Paramétrage** : Le choix de la structure du réseau et des hyperparamètres peut être complexe.

---

##### 4.4 One-Class SVM

###### Théorie
- Le One-Class SVM est une variante du Support Vector Machine qui est utilisée pour identifier les anomalies en créant une frontière qui sépare toutes les données d'entraînement de l'origine dans l'espace des caractéristiques.
- Les points qui se trouvent à l'extérieur de cette frontière sont considérés comme des anomalies.

###### Implémentation

```python
from sklearn.svm import OneClassSVM

# Application du One-Class SVM

model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)  # Paramètres : kernel Radial Basis Function, gamma contrôle la largeur du kernel, nu détermine la fraction de points considérés comme anomalies
df['anomaly'] = model.fit_predict(df[['Temperature']])

# Identification des anomalies
anomalies = df[df['anomaly'] == -1]
print(anomalies)
```

###### Avantages
- **Flexibilité** : Le choix du kernel permet de capturer des formes de données complexes.
- **Robustesse** : Peut bien fonctionner même avec des données bruitées ou des distributions non-gaussiennes.

###### Inconvénients
- **Sensibilité aux paramètres** : Les résultats peuvent être très dépendants du choix des paramètres `gamma` et `nu`.
- **Scalabilité** : Peut être inefficace sur de grands ensembles de données, en particulier avec des kernels complexes.

---

##### 4.5 Clustering Hiérarchique

###### Théorie
- Le clustering hiérarchique est une méthode de clustering qui regroupe les points de données en une hiérarchie de clusters.
- Il existe deux approches principales : l'algorithme agglomératif (fusion de clusters) et l'algorithme divisif (division de clusters).
- Pour la détection d'anomalies, les points qui ne s'intègrent pas bien dans les clusters formés sont considérés comme des anomalies.

###### Implémentation

```python
from scipy.cluster.hierarchy import linkage, fcluster

# Application du clustering hiérarchique agglomératif
Z = linkage(df[['Temperature']], method='ward')  # 'ward' minimise la variance à chaque étape
df['cluster'] = fcluster(Z, t=2, criterion='maxclust')  # t=2 signifie qu'on forme 2 clusters

# Identification des anomalies (points dans les petits clusters)
anomalies = df[df['cluster'] == 2]  # Supposons que le cluster 2 est celui qui contient les anomalies
print(anomalies)
```

###### Avantages
- **Interprétabilité** : Les résultats peuvent être visualisés sous forme de dendrogrammes, ce qui facilite l'analyse.
- **Flexibilité** : Peut être utilisé pour différents types de données, avec des méthodes de liaison variées (e.g., Ward, simple, complet).

###### Inconvénients
- **Complexité computationnelle** : Le temps de calcul augmente rapidement avec la taille de l'ensemble de données.
- **Sensibilité au bruit** : Les petits changements dans les données peuvent entraîner des changements significatifs dans la hiérarchie des clusters.

---

#### Exemples d'Implémentation

- Dans cette section, nous allons illustrer comment utiliser ces algorithmes pour détecter des anomalies de température.
- Chaque exemple inclura des explications détaillées sur le code, des visualisations potentielles, et une analyse des résultats.

##### 5.1 Exemple d'Isolation Forest

- Isolation Forest est particulièrement bien adapté à la détection d'anomalies dans des ensembles de données avec de nombreuses dimensions ou lorsque les anomalies sont rares.
- Le code présenté plus haut montre comment il peut être utilisé pour identifier des températures anormales.

###### Explication et Analyse du Code
- L'algorithme divise les données en sous-ensembles jusqu'à ce que chaque point de données soit isolé.
- Les anomalies, qui sont des points rares et éloignés des autres, seront isolées plus rapidement que les points normaux.
- L'exemple utilise un taux de contamination de 1%, ce qui signifie que l'on s'attend à ce que 1% des points de données soient des anomalies.

###### Visualisation
- Vous pouvez visualiser les résultats en utilisant un graphique de points, où les anomalies sont mises en évidence :

```python
import matplotlib.pyplot as plt

plt.scatter(df.index, df['Temperature'], c=df['anomaly'], cmap='coolwarm')
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.title('Isolation Forest - Détection des Anomalies')
plt.show()
```

##### 5.2 Exemple de DBSCAN

- DBSCAN est utile pour détecter des anomalies dans des données où les clusters sont bien définis et les anomalies ne sont pas connectées à des clusters denses.
- Le code présenté plus haut montre comment utiliser DBSCAN pour identifier des anomalies.

###### Explication et Analyse du Code
- DBSCAN regroupe les points de données qui sont proches les uns des autres en clusters basés sur la densité.
- Les points qui ne peuvent pas être regroupés dans un cluster sont considérés comme du bruit ou des anomalies.
- Le choix des paramètres `epsilon` et `min_samples` est crucial et peut nécessiter des ajustements en fonction des données.

###### Visualisation
- Un graphique similaire à celui utilisé pour l'Isolation Forest peut être utilisé ici pour visualiser les résultats :

```python
plt.scatter(df.index, df['Temperature'], c=df['cluster'], cmap='coolwarm')
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.title('DBSCAN - Détection des Anomalies')
plt.show()
```

##### 5.3 Exemple d'Autoencodeur

- Les autoencodeurs sont particulièrement efficaces pour des ensembles de données où les relations entre les caractéristiques sont complexes.
- Ils nécessitent cependant *un entraînement significatif*, ce qui peut prendre du temps pour des ensembles de données volumineux.

###### Explication et Analyse du Code
- L'autoencodeur est un réseau de neurones qui apprend à reconstruire ses entrées. Les anomalies sont détectées en mesurant l'erreur de reconstruction.
- Les points avec une erreur de reconstruction élevée sont considérés comme des anomalies, car le modèle n'a pas réussi à bien les représenter.

###### Visualisation
Vous pouvez visualiser l'erreur de reconstruction pour identifier les anomalies :

```python
plt.hist(df['reconstruction_error'], bins=50)
plt.xlabel('Erreur de Reconstruction')
plt.ylabel('Fréquence')
plt.title('Autoencodeur - Distribution de l\'Erreur de Reconstruction')
plt.show()

plt.scatter(df.index, df['reconstruction_error'], c=df['anomaly'], cmap='coolwarm')
plt.xlabel('Index')
plt.ylabel('Erreur de Reconstruction')
plt.title('Autoencodeur - Détection des Anomalies')
plt.show()
```

---

#### Analyse et Comparaison des Résultats

- Dans cette section, nous analysons les résultats obtenus avec chaque algorithme.
- Nous comparons leurs performances, leur précision dans la détection des anomalies, et leur applicabilité à différents types de données. Les points d'intérêt incluent :

- **Taux de faux positifs** : Combien d'anomalies détectées sont en réalité des points normaux ?
- **Taux de faux négatifs** : Combien d'anomalies réelles n'ont pas été détectées ?
- **Robustesse** : L'algorithme est-il sensible aux changements de données ?
- **Scalabilité** : Comment l'algorithme se comporte-t-il avec des ensembles de données de grande taille ?

- Cette analyse détaillée vous aide à choisir l'algorithme le plus adapté à vos besoins spécifiques.

---

#### Conclusion

- La détection d'anomalies de température est essentielle pour de nombreuses applications, notamment la surveillance climatique et industrielle.
- Les algorithmes d'apprentissage non supervisé offrent une approche puissante pour identifier ces anomalies sans nécessiter de données étiquetées.
- Chacun des algorithmes présentés a ses avantages et inconvénients, et le choix dépendra de la nature de vos données et de vos exigences spécifiques.

Pour des implémentations plus avancées, vous pouvez également envisager de combiner plusieurs algorithmes pour améliorer la précision et la robustesse de la détection.

---

#### Références et Ressources Complémentaires

- [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [Autoencoders - Keras Documentation](https://keras.io/examples/autoencoders/)
- [One-Class SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
- [Clustering Hiérarchique](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
