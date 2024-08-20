# 🚀 Considérez le code suivant :

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 🚨 Masquer certains messages d'erreur de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 🛠️ Chargement des données Fashion MNIST
fashion_mnist_data = keras.datasets.fashion_mnist
(all_x_train, all_y_train), (x_test, y_test) = fashion_mnist_data.load_data()

# 🔍 Préparation des données
all_x_train = all_x_train.astype('float32')
x_test = x_test.astype('float32')
x_validation, x_train = all_x_train[:5000] / 255.0, all_x_train[5000:] / 255.0
y_validation, y_train = all_y_train[:5000], all_y_train[5000:]

# 🧵 Noms des classes Fashion MNIST
fashion_mnist_class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                             "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 🧩 Création du modèle séquentiel
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(150, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# 🔧 Compilation du modèle
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# 📂 Configurer le chemin d'accès aux logs de TensorBoard
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 🏋️‍♂️ Entraînement du modèle
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_validation, y_validation), callbacks=[tensorboard_cb])
```

---

## 📝 03-Questions-Réponses

### 1. 📊 Concept de division des données en données d'entraînement et de test

**❓ Qu'est-ce que l'entraînement et le test ?**

```python
(all_x_train, all_y_train), (x_test, y_test) = fashion_mnist_data.load_data()
```

La division des données en ensembles d'entraînement et de test est une pratique clé en apprentissage automatique pour évaluer la performance des modèles. **L'entraînement** d'un modèle implique l'ajustement de ses paramètres sur un ensemble de données d'entraînement pour qu'il puisse faire des prédictions précises. **Le test** évalue ensuite les performances du modèle sur un ensemble de données distinct pour garantir sa capacité sur de nouvelles données.

### 2. 🔍 Différence entre les `x` et les `y` dans la ligne du code ci-dessus

Le préfixe `x` représente les données (les images dans ce cas), tandis que celles avec le préfixe `y` représentent les étiquettes correspondantes de ces images.

### 3. 🔢 Types de données : différence entre `float32` et `float64`

La différence entre `float32` et `float64` réside principalement dans la précision et la taille en mémoire. `float64` est la double précision de `float32`. [Lire plus ici](https://stackoverflow.com/questions/43440821/the-real-difference-between-float32-and-float64).


## float32 est moins précis mais plus rapide que float64, et float64 est plus précis que float32 mais consomme plus de mémoire. Si la précision est plus importante que la vitesse, vous pouvez utiliser float64. Et si la vitesse est plus importante que la précision, vous pouvez utiliser float32.
 

### 4. 📦 Variables catégoriques et One-Hot Encoding

**❓ Qu'est-ce que le concept des variables catégoriques et en quoi est-il important en IA ?**  
Les variables catégoriques sont des variables qui représentent différents groupes ou catégories distinctes au sein d'un ensemble de données.

**🔑 Qu'est-ce que le One-Hot Encoding ?**  
Le One-Hot Encoding est une technique qui encode les données catégorielles (T-shirt, Pull, etc.) en données numériques, permettant leur utilisation dans des modèles de régression ou de classification.

### 5. 🧪 Données de validation vs. Données de test

**❓ Qu'est-ce que les données de validation ? Est-ce la même chose que les données de test ?**  
Les données de validation sont utilisées pour ajuster et évaluer la performance d'un modèle pendant son entraînement, alors que les données de test sont réservées pour évaluer la performance finale du modèle.

### 6. 🧠 Normalisation en Intelligence Artificielle (IA)

**🔄 Qu'est-ce que la normalisation en IA ?**  
La normalisation est le processus de mise à l'échelle des caractéristiques des données pour améliorer la performance et la stabilité du modèle.

---


### 6.1. 🔄 Normalisation vs. Standardisation

**❓ Qu'est-ce que la différence entre la normalisation et la standardisation en IA ?**

- **Normalisation :**  
  La normalisation est le processus qui consiste à redimensionner les valeurs d'une variable afin qu'elles se situent dans une plage spécifique, généralement entre 0 et 1. Cela est utile lorsque vous avez des données de différents ordres de grandeur et que vous souhaitez les rendre comparables.  
  Par exemple, pour normaliser une variable \(X\), on utilise souvent la formule suivante :
  
$$
X_{\text{normalisé}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$

- **Standardisation :**  
  La standardisation transforme les données de manière à ce qu'elles aient une moyenne de 0 et un écart-type de 1. Cela est utile lorsque les données suivent une distribution gaussienne (ou une distribution proche de la normale) et que vous souhaitez centrer et mettre à l'échelle les données.  
  Par exemple, pour standardiser une variable \(X\), on utilise la formule suivante 

$$
X_{\text{standardisé}} = \frac{X - \mu}{\sigma}
$$


où *mu* est la moyenne de X et *sigma* est l'écart-type.

**Quand utiliser quoi ?**  
- Utilisez la **normalisation** lorsque vos données ne suivent pas une distribution gaussienne et que vous avez des caractéristiques avec des gammes différentes.
- Utilisez la **standardisation** lorsque vos données sont normalement distribuées ou lorsque vous travaillez avec des modèles linéaires ou des algorithmes basés sur des distances (comme SVM ou KNN).


---

### 📝 Exemples de Code et Questions-Réponses

### 7. 🎯 Considérez le code suivant

```python
fashion_mnist_class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                             "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 🧩 Création du modèle séquentiel
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(150, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
```

**❓ 7.1) À l’origine, quel est le type de `y_train` et `y_test` ?**

`uint8` : un type de données entier non signé sur 8 bits.  
```python
print(y_train.dtype)  # Output: uint8
print(y_train.shape)  # Output: (55000,)
print(y_test.shape)   # Output: (10000,)
```

**❓ 7.2) Pour l'algorithme, est-il préférable d'avoir un type de données numérique ou chaîne de caractères ? Pourquoi ?**

Il est préférable d'utiliser des types de données numériques car les algorithmes d'IA traitent des valeurs numériques. Les bibliothèques comme TensorFlow sont optimisées pour les tenseurs numériques.

**❓ 7.3) Pour les humains, est-il préférable d'avoir un type de données numérique ou chaîne de caractères ? Pourquoi ?**

Les chaînes de caractères sont plus compréhensibles pour les humains, mais les types numériques sont essentiels pour les calculs et le traitement informatique.



---

### 🎯 Partie 1 - Questions Simples

**❓ 8.1) Qu'est-ce qu'un modèle `Sequential` ?**

Le modèle `Sequential` est une pile linéaire de couches où chaque couche a exactement un tenseur d'entrée et un tenseur de sortie. C'est une approche simple pour créer des modèles d'apprentissage profond.

**❓ 8.2) Qu'est-ce que la couche `Flatten` ?**

La couche `Flatten` transforme les entrées multidimensionnelles en un vecteur 1D pour être traité par les couches suivantes du modèle.

**❓ 8.3) Pourquoi avons-nous 10 neurones dans la couche de sortie ?**

Nous avons 10 neurones dans la couche de sortie car il y a 10 classes dans les données `fashion_mnist_class_names`.

---

### 🧠 Partie 2 - Questions Intermédiaires

**❓ 8.5) Qu'est-ce qu'une couche `Dense` ?**

Une couche `Dense` connecte chaque neurone à tous les neurones de la couche précédente.

**❓ 8.6) Pourquoi utilise-t-on les fonctions d'activation `relu` et `softmax` ?**

- **ReLU** : Pour les couches cachées, elle introduit de la non-linéarité et aide à éviter les problèmes de gradient.
- **Softmax** : Pour la couche de sortie, elle génère une distribution de probabilités pour la classification.

**❓ 8.7) Pourquoi avons-nous choisi 300 neurones, 150 neurones et 2 couches ?**

Le choix des neurones et du nombre de couches dépend de la complexité de la tâche et de la dimensionnalité des données. Ces paramètres sont ajustés pour optimiser la performance du modèle.
