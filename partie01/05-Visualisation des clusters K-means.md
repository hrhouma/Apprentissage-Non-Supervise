# Example - Création d'un nouveau carnet pour le clustering

- Pour débuter notre travail sur le clustering, entrons dans notre environnement de travail.
- En cliquant sur "Nouveau Carnet", je choisis Python 3 comme environnement d'exécution.
- Une fois le carnet ouvert, je le nomme "Clustering". 

## Chargement des données

- Avant toute chose, il est essentiel de charger nos données. 
- Dans le dossier `Data`, je trouve mon fichier `Entertainment_Clean.csv`. 
- Pour le lire, j'utilise la bibliothèque `pandas` :

```python
import pandas as pd
data = pd.read_csv("../Data/Entertainment_Clean.csv")
```

Ce fichier contient des données sur le temps passé par les étudiants à lire, regarder la télévision et jouer à des jeux vidéo chaque semaine.

## Vérification des données

Avant de procéder au clustering, vérifions que nos données sont prêtes pour la modélisation :

1. **Granularité des données** : Chaque ligne représente un étudiant unique.
2. **Valeurs non-nulles** : S'assurer qu'il n'y a pas de valeurs manquantes.
3. **Type numérique** : Les données doivent être numériques pour l'analyse.

Après ces vérifications, il est souvent utile de visualiser les données pour comprendre leur distribution et potentiellement identifier des clusters visuellement.

## Modélisation : Clustering K-means

Passons à la modélisation. Nous utiliserons l'algorithme K-means de la bibliothèque scikit-learn pour identifier des groupes d'étudiants selon leurs habitudes de divertissement :

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42).fit(data)
```

Ce code configure un modèle K-means pour trouver deux clusters dans notre ensemble de données. `random_state=42` garantit que les résultats sont reproductibles.

## Visualisation des Clusters

Après avoir ajusté le modèle, il est crucial de visualiser les résultats pour interpréter les clusters. Utilisons `matplotlib` pour créer une visualisation :

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(data['Books'], data['Video Games'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Heures passées à lire')
plt.ylabel('Heures passées à jouer')
plt.title('Visualisation des Clusters d\'Étudiants')
plt.show()
```

Cette visualisation nous aide à voir comment les étudiants sont regroupés selon le temps qu'ils consacrent à lire et à jouer.

## Conclusion

En suivant ces étapes, vous avez non seulement préparé vos données mais aussi appliqué et visualisé les résultats d'un modèle de clustering. Ce processus nous permet de tirer des insights significatifs sur les comportements des étudiants, essentiels pour des décisions ciblées, comme ajuster les stratégies de marketing de la bibliothèque.
