# Partie 4 - DÉMO : Clustering K-means en Python

## Cours Complet sur l'Application Pratique du K-means avec Python

### Introduction
Nous avons déjà exploré en profondeur la théorie derrière le regroupement K-means, mais passons maintenant à la pratique avec Python. L'application de cet algorithme est relativement simple grâce à la bibliothèque `scikit-learn`, qui propose une implémentation robuste et facile à utiliser du K-means.

### Installation et Importation de K-means
La première étape consiste à importer le module K-means de `scikit-learn`. Assurez-vous d'avoir installé la bibliothèque `scikit-learn` avant de procéder. Vous pouvez l'installer via pip si nécessaire:

```bash
pip install scikit-learn
```

Ensuite, importez K-means de la manière suivante :

```python
from sklearn.cluster import KMeans
```

### Création de l'Instance K-means
Lors de la création d'une instance de K-means, plusieurs paramètres peuvent être configurés selon vos besoins spécifiques. Voici les principaux arguments à considérer :

- **n_clusters** : Le nombre de clusters à former. Cela correspond à 'K' dans K-means. Vous devez décider de cette valeur en tant que data scientist, car elle influence directement la granularité du clustering.

  ```python
  kmeans = KMeans(n_clusters=2)
  ```

- **n_init** : Le nombre de fois que l'algorithme sera exécuté avec des centroides initiaux différents. L'algorithme peut aboutir à des résultats différents en fonction de ces points de départ aléatoires.

  ```python
  kmeans = KMeans(n_clusters=2, n_init=10)
  ```

- **random_state** : Fixe la graine du générateur de nombres aléatoires utilisé pour l'initialisation des centroides. Ceci est utile pour la reproductibilité des résultats.

  ```python
  kmeans = KMeans(n_clusters=2, random_state=42)
  ```

### Application de K-means
Une fois que vous avez configuré votre instance K-means, vous pouvez l'appliquer à vos données. Utilisez la méthode `.fit()` pour ajuster le modèle aux données :

```python
kmeans.fit(data)
```

### Interprétation des Résultats
Après avoir ajusté le modèle, vous pouvez obtenir les étiquettes des clusters pour chaque point de données :

```python
labels = kmeans.labels_
```

Ces étiquettes vous indiquent à quel cluster chaque point appartient. Vous pouvez maintenant analyser ces clusters pour extraire des insights, effectuer des visualisations ou ajuster le modèle si nécessaire.

### Visualisation et Ajustement
Il est souvent utile de visualiser les résultats pour mieux comprendre la distribution des clusters. Utilisez des bibliothèques comme `matplotlib` ou `seaborn` pour créer des graphiques de dispersion montrant les clusters :

```python
import matplotlib.pyplot as plt

plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.show()
```

### Exploration Plus Avancée
Si vous n'êtes pas sûr du nombre de clusters optimal, vous pouvez tester plusieurs valeurs de 'K' et utiliser des méthodes comme la méthode du coude pour choisir la meilleure option. Cela implique de regarder la variance expliquée en fonction du nombre de clusters et de chercher un 'coude' dans le graphique.

### Conclusion
L'application de K-means en Python avec `scikit-learn` est un processus direct une fois que vous maîtrisez les paramètres et les méthodes. Cela vous permet non seulement d'appliquer la théorie à des cas pratiques mais aussi d'obtenir des résultats tangibles et visuellement interprétables. Restez à l'écoute pour une exploration plus approfondie dans un notebook Jupyter où nous pouvons expérimenter et visualiser directement les résultats.
