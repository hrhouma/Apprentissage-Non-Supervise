## 03-Questions-Reponses


### 1. Concept de division des données en données d'entraînement et de test

**Qu'est-ce que l'entraînement et le test ?**  

```python
(all_x_train, all_y_train), (x_test, y_test) = fashion_mnist_data.load_data()
```

La division des données en ensembles d'entraînement et de test est une pratique en apprentissage automatique pour évaluer la performance des modèles. L'entraînement d'un modèle implique l'ajustement de ses paramètres sur un ensemble de données d'entraînement pour qu'il puisse faire des prédictions précises. Le test évalue les performances du modèle sur un ensemble de données distinct pour garantir sa capacité sur de nouvelles données.

### 2. Différence entre les x et les y dans la ligne du code ci-dessus

Le préfixe `x` représente les données (les images dans ce cas), tandis que celles avec le préfixe `y` représentent les étiquettes correspondantes de ces images.

### 3. Types de données : différence entre `float32` et `float64`

[The Real Difference Between float32 and float64 (StackOverflow)](https://stackoverflow.com/questions/43440821/the-real-difference-between-float32-and-float64)  
La différence entre `float32` et `float64` réside principalement dans la précision et la taille en mémoire. `float64` est la double précision de `float32`.

### 4. Variables catégoriques et One-Hot Encoding

**Qu'est-ce que le concept des variables catégoriques et en quoi est-il important en IA ?**  
Les variables catégoriques sont des variables qui représentent différents groupes ou catégories distinctes au sein d'un ensemble de données.

**Qu'est-ce que le One-Hot Encoding ?**  
Le One-Hot Encoding est une technique qui encode les données catégorielles (T-shirt, Pull, etc.) en données numériques. Elle est utilisée pour donner un poids aux données catégorielles afin qu'elles puissent être utilisées dans un modèle de régression linéaire.

### 5. Données de validation vs. Données de test

**Qu'est-ce que les données de validation ? Est-ce la même chose que les données de test ?**  
Les données de validation sont un ensemble de données utilisé pour ajuster et évaluer la performance d'un modèle pendant son entraînement. En revanche, les données de test sont réservées pour évaluer la performance finale du modèle.

### 6. Normalisation en Intelligence Artificielle (IA)

**Qu'est-ce que la normalisation en IA ?**  
La normalisation en IA est le processus consistant à mettre à l'échelle les caractéristiques (variables) des données dans un intervalle spécifique ou une distribution spécifique pour améliorer la performance et la stabilité du modèle.

## Exemples de Code et Questions-Réponses

### 7. Considérez le code suivant

```python
fashion_mnist_class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                             "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Création du modèle séquentiel
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(150, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
```

#### 7.1) À l’origine, quel est le type de `y_train` et `y_test` ?

`uint8` : un type de données entier non signé sur 8 bits.  
Exemple :  
```python
print(y_train.dtype)  # Output: uint8
print(y_train.shape)  # Output: (55000,)
print(y_test.shape)   # Output: (10000,)
```

#### 7.2) Pour l'algorithme, est-il préférable d'avoir un type de données numérique ou chaîne de caractères ? Pourquoi ?

Il est généralement préférable d'utiliser des types de données numériques car les algorithmes d'IA traitent des valeurs numériques (comme les pixels d'images). De plus, les bibliothèques populaires comme TensorFlow et PyTorch sont optimisées pour travailler avec des tenseurs numériques.

#### 7.3) Pour les humains, est-il préférable d'avoir un type de données numérique ou chaîne de caractères ? Pourquoi ?

Les types de données numériques sont essentiels pour les calculs mathématiques et les traitements informatiques. Cependant, pour la communication et la compréhension humaine, les chaînes de caractères sont généralement plus appropriées.

## Partie 1 - Questions Simples

#### 8.1) Qu'est-ce qu'un modèle `Sequential` ?

Le modèle `Sequential` de Keras est une pile linéaire de couches où chaque couche a exactement un tenseur d'entrée et un tenseur de sortie. C'est un moyen simple de créer des modèles d'apprentissage profond couche par couche.

#### 8.2) Qu'est-ce que la couche `Flatten` ?

La couche `Flatten` dans Keras est utilisée pour aplatir les entrées multidimensionnelles en un tenseur 1D.  
Exemple :  
```python
model.add(keras.layers.Flatten(input_shape=[28, 28]))
# Cette couche reçoit des données d'entrée de forme (28, 28) et les aplatit en un tenseur 1D de forme (784).
```

#### 8.3) Qu'est-ce que la couche de sortie ?

La couche de sortie est la dernière couche du réseau qui produit les prédictions finales.  
**Pourquoi avons-nous 10 neurones dans la couche de sortie ?**  
Le nombre de neurones dans la couche de sortie correspond généralement au nombre de classes, ici 10, correspondant aux catégories dans `fashion_mnist_class_names`.

## Partie 2 - Questions Intermédiaires

#### 8.5) Qu'est-ce qu'une couche `Dense` ?

Une couche `Dense` est une couche où chaque neurone est connecté à tous les neurones de la couche précédente.

#### 8.6) Qu'est-ce qu'une fonction d'activation ?

Une fonction d'activation est une opération mathématique appliquée à la sortie d'un neurone artificiel pour introduire de la non-linéarité dans le modèle.

**Pourquoi `relu` et `softmax` pour la sortie ?**  
- **ReLU (Rectified Linear Unit)** : Remplace les valeurs négatives par 0, souvent utilisé pour sa simplicité et performance.
- **Softmax** : Convertit un vecteur en distribution de probabilités, utilisée pour la classification multi-classes.

#### 8.7) Pourquoi avons-nous choisi 300 neurones, 150 neurones et pourquoi avons-nous choisi 2 couches ?

Le choix du nombre de neurones et de couches dépend de la complexité de la tâche et de la dimensionnalité des données d'entrée. Plus la tâche est complexe, plus on augmente la taille des couches denses. Ce sont des hyperparamètres définis avant le début de l'entraînement du modèle.

---

Ce README vous guidera à travers la compréhension et l'implémentation du projet de classification avec le dataset Fashion MNIST. Référez-vous aux références vidéo et code pour des exemples supplémentaires et des démonstrations pratiques.
