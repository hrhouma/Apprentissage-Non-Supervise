### Les Réseaux de Neurones Convolutifs (CNN)

# Table des Matières
- [**Chapitre 1 : Introduction aux Images Numériques**](#chapitre-1--introduction-aux-images-numériques)
  - [1.1 Qu'est-ce qu'une image numérique ?](#11-quest-ce-quune-image-numerique)
  - [1.2 Types d'images numériques](#12-types-d-images-numériques)
  - [1.3 Résolution d'une image](#13-résolution-d-une-image)

- [**Chapitre 2 : Les Bases des Réseaux de Neurones**](#chapitre-2)
  - [2.1 Qu'est-ce qu'un réseau de neurones ?](#2.1)
  - [2.2 Neurone artificiel](#2.2)
  - [2.3 Couches d'un réseau de neurones](#2.3)
- [**Chapitre 3 : Introduction aux Réseaux de Neurones Convolutifs (CNN)**](#chapitre-3)
  - [3.1 Historique des CNN](#3.1)
  - [3.2 Structure d'un CNN](#3.2)
- [**Chapitre 4 : Convolution dans les CNN**](#chapitre-4)
  - [4.1 Qu'est-ce que la convolution ?](#4.1)
  - [4.2 Filtre (Kernel)](#4.2)
  - [4.3 Stride](#4.3)
  - [4.4 Padding](#4.4)
- [**Chapitre 5 : Couches de ReLU dans les CNN**](#chapitre-5)
  - [5.1 Fonction d'activation ReLU](#5.1)
- [**Chapitre 6 : Couches de Pooling dans les CNN**](#chapitre-6)
  - [6.1 Qu'est-ce que le pooling ?](#6.1)
  - [6.2 Avantages du pooling](#6.2)
- [**Chapitre 7 : Couches Entièrement Connectées et Classification**](#chapitre-7)
  - [7.1 Couches entièrement connectées](#7.1)
  - [7.2 Fonctionnement](#7.2)
- [**Chapitre 8 : Entraînement des CNN**](#chapitre-8)
  - [8.1 Processus d'entraînement](#8.1)
  - [8.2 Hyperparamètres](#8.2)
- [**Chapitre 9 : Applications des CNN**](#chapitre-9)
  - [9.1 Domaines d'application](#9.1)
- [**Chapitre 10 : Avantages et Inconvénients des CNN**](#chapitre-10)
  - [10.1 Avantages](#10.1)
  - [10.2 Inconvénients](#10.2)
- [**Conclusion de la première partie**](#conclusion-partie-1)
- [**Chapitre 11 : Introduction à Keras**](#chapitre-11)
  - [Qu'est-ce que Keras ?](#11.1)
  - [Pourquoi utiliser Keras ?](#11.2)
  - [Environnement Keras](#11.3)
- [**Chapitre 12 : Créer Votre Premier Réseau de Neurones avec Keras**](#chapitre-12)
  - [Introduction](#12.1)
  - [Importation de Keras](#12.2)
  - [Installation de Keras](#12.3)
  - [Création d'un Modèle Simple](#12.4)
  - [Changer de Backend](#12.5)
- [**Chapitre 13 : Construire des Modèles avec Keras**](#chapitre-13)
  - [Modèle Sequential](#13.1)
  - [Classe Model avec API Fonctionnelle](#13.2)
- [**Chapitre 14 : Introduction aux Réseaux de Neurones Convolutionnels (CNN) avec Keras**](#chapitre-14)
  - [Vue d'ensemble des Réseaux de Neurones Convolutionnels](#14.1)
  - [Comprendre les Couches dans Keras](#14.2)
  - [Les Couches Communes dans Keras](#14.3)
  - [Les Couches Convolutionnelles](#14.4)
  - [Construire un Réseau de Neurones Convolutionnel avec Keras](#14.5)
  - [Entraîner le CNN](#14.6)
  - [Évaluer le Modèle](#14.7)
- [**Chapitre 15 : Composants d'un CNN**](#chapitre-15)
  - [Convolution](#15.1)
  - [Activation Non-Linéaire (ReLU)](#15.2)
  - [Pooling (Sous-échantillonnage)](#15.3)
- [**Chapitre 16 : Exemple Pratique avec MNIST et Fashion MNIST**](#chapitre-16)
  - [MNIST](#16.1)
  - [Fashion MNIST](#16.2)
- [**Chapitre 17 : Apprentissage par Transfert**](#chapitre-17)
  - [Concept](#17.1)
  - [Exemple avec Inception V3](#17.2)
- [**Structure des Dossiers de Données**](#structure-dossiers)
- [**Synthèse**](#synthèse)


# Chapitre 1 : Introduction aux Images Numériques

### 1.1 Qu'est-ce qu'une image numérique ? <a name="11-quest-ce-quune-image-numerique"></a>

- **Définition** : Une image numérique est une représentation visuelle d'une scène sous forme de grille de pixels.
- **Pixel** : L'élément de base d'une image numérique. Chaque pixel contient des informations sur la couleur ou la luminosité.

```python
# Exemple : Afficher une image en niveaux de gris
import matplotlib.pyplot as plt
import numpy as np

# Créer une image 8x8 avec des valeurs de niveaux de gris
image = np.array([
    [0, 30, 60, 90, 120, 150, 180, 210],
    [30, 60, 90, 120, 150, 180, 210, 240],
    [60, 90, 120, 150, 180, 210, 240, 255],
    [90, 120, 150, 180, 210, 240, 255, 255],
    [120, 150, 180, 210, 240, 255, 255, 255],
    [150, 180, 210, 240, 255, 255, 255, 255],
    [180, 210, 240, 255, 255, 255, 255, 255],
    [210, 240, 255, 255, 255, 255, 255, 255]
])

plt.imshow(image, cmap='gray')
plt.colorbar()
plt.show()
```

### 1.2 Types d'images numériques
- **Images en niveaux de gris** : Chaque pixel est une valeur de luminosité allant de 0 (noir) à 255 (blanc).
- **Images en couleurs** : Chaque pixel est représenté par trois valeurs correspondant aux canaux de couleur rouge, vert et bleu (RVB).

```python
# Exemple : Afficher une image en couleurs
import matplotlib.pyplot as plt
import numpy as np

# Créer une image 8x8 avec des valeurs RVB
image = np.zeros((8, 8, 3), dtype=np.uint8)
image[..., 0] = np.linspace(0, 255, 8)  # Canal rouge
image[..., 1] = np.linspace(0, 255, 8)[:, np.newaxis]  # Canal vert
image[..., 2] = 128  # Canal bleu

plt.imshow(image)
plt.show()
```

### 1.3 Résolution d'une image
- **Définition** : La résolution d'une image est déterminée par le nombre de pixels horizontaux et verticaux.
- **Exemple** : Une image de 128 x 128 pixels contient 16 384 pixels.

```python
# Exemple : Créer une image de 128x128 pixels
import numpy as np

image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
print(f"Nombre de pixels dans l'image : {image.size}")
```

---

#### Chapitre 2 : Les Bases des Réseaux de Neurones

##### 2.1 Qu'est-ce qu'un réseau de neurones ?
- **Définition** : Un réseau de neurones est un modèle mathématique inspiré du cerveau humain, conçu pour reconnaître des motifs et apprendre à partir de données.

##### 2.2 Neurone artificiel
- **Fonctionnement** : Chaque neurone reçoit des entrées, les transforme à l'aide de poids et de biais, applique une fonction d'activation, puis produit une sortie.
- **Fonction d'activation** : Une fonction mathématique qui introduit de la non-linéarité, essentielle pour la modélisation de relations complexes.

```python
# Exemple : Neurone simple avec fonction d'activation sigmoïde
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Entrées du neurone
inputs = np.array([1.0, 2.0, 3.0])
# Poids et biais
weights = np.array([0.2, 0.8, -0.5])
bias = 2.0

# Calcul de la sortie
output = sigmoid(np.dot(inputs, weights) + bias)
print(f"Sortie du neurone : {output}")
```

##### 2.3 Couches d'un réseau de neurones
- **Couche d'entrée** : Reçoit les données brutes.
- **Couches cachées** : Effectuent des transformations et extraient des caractéristiques.
- **Couche de sortie** : Produit le résultat final, par exemple, la classe prédite d'une image.

```python
# Exemple : Réseau de neurones simple avec une couche cachée
from keras.models import Sequential
from keras.layers import Dense

# Créer le modèle
model = Sequential()
model.add(Dense(4, input_dim=3, activation='relu'))  # Couche cachée
model.add(Dense(1, activation='sigmoid'))  # Couche de sortie

# Afficher le résumé du modèle
model.summary()
```

---

#### Chapitre 3 : Introduction aux Réseaux de Neurones Convolutifs (CNN)

##### 3.1 Historique des CNN
- **Origines biologiques** : Inspirés par le fonctionnement du cortex visuel des mammifères (recherches de Hubel et Wiesel).
- **Développement en informatique** : Introduction par Yann LeCun et al. en 1998 pour la classification du dataset MNIST.

##### 3.2 Structure d'un CNN
- **Couche d'entrée** : Reçoit l'image sous forme de matrice de pixels.
- **Couches de convolution** : Appliquent des filtres pour extraire des caractéristiques locales.
- **Couches de ReLU** : Introduisent de la non-linéarité en remplaçant les valeurs négatives par zéro.
- **Couches de pooling** : Réduisent la dimensionnalité de l'image tout en conservant les informations essentielles.
- **Couches entièrement connectées** : Effectuent la classification en se basant sur les caractéristiques extraites.

```python
# Exemple : Construire un simple CNN avec Keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Créer le modèle
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # Couche de convolution
model.add(MaxPooling2D((2, 2)))  # Couche de pooling
model.add(Flatten())  # Couche de mise à plat
model.add(Dense(64, activation='relu'))  # Couche entièrement connectée
model.add(Dense(10, activation='softmax'))  # Couche de sortie

# Afficher le résumé du modèle
model.summary()
```

---

#### Chapitre 4 : Convolution dans les CNN

##### 4.1 Qu'est-ce que la convolution ?
- **Définition** : Opération mathématique qui combine deux fonctions pour produire une troisième fonction.
- **Application dans les CNN** : Utilisée pour appliquer des filtres sur les images afin d'extraire des caractéristiques spécifiques.

```python
# Exemple : Appliquer un filtre de convolution à une image
from scipy.ndimage import convolve

# Définir un filtre (kernel) simple
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Appliquer la convolution
convoluted_image = convolve(image, kernel)
plt.imshow(convoluted_image, cmap='gray')
plt.show()
```

##### 4.2 Filtre (Kernel)
- **Définition** : Une petite matrice utilisée pour balayer l'image et effectuer des opérations de convolution.
- **Exemple** : Un filtre 3x3 appliqué sur une image 5x5.

```python
# Exemple : Définir et appliquer un filtre 3x3
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

convoluted_image = convolve(image, kernel)
plt.imshow(convoluted_image, cmap='gray')
plt.show()
```

##### 4.3 Stride
- **Définition** : Le nombre de pixels par lesquels le filtre se déplace sur l'image.
- **Impact sur la dimension de l'image convoluée** : Un stride plus grand réduit la dimension de l'image de sortie.

```python
# Exemple : Appliquer la convolution avec différents strides
def apply_convolution(image, kernel, stride):
    output_shape = (
        (image.shape[0] - kernel.shape[0]) // stride + 1,
        (image.shape[1] - kernel.shape[1]) // stride + 1
    )
    output = np.zeros(output_shape)
    for i in range(0, output_shape[0], stride):
        for j in range(0, output_shape[1], stride):
            output[i, j] = np.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return output

convoluted_image_stride1 = apply_convolution(image, kernel, 1)
con

voluted_image_stride2 = apply_convolution(image, kernel, 2)

plt.subplot(1, 2, 1)
plt.title('Stride 1')
plt.imshow(convoluted_image_stride1, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Stride 2')
plt.imshow(convoluted_image_stride2, cmap='gray')

plt.show()
```

##### 4.4 Padding
- **Définition** : Ajout de bordures de pixels autour de l'image pour conserver sa dimension après la convolution.
- **Types de padding** : Valid (sans padding) et Same (avec padding pour conserver la taille d'origine).

```python
# Exemple : Appliquer la convolution avec et sans padding
def apply_padding(image, padding_size):
    padded_image = np.pad(image, pad_width=padding_size, mode='constant', constant_values=0)
    return padded_image

padded_image = apply_padding(image, 1)
convoluted_image_padded = convolve(padded_image, kernel)

plt.subplot(1, 2, 1)
plt.title('Sans Padding')
plt.imshow(convoluted_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Avec Padding')
plt.imshow(convoluted_image_padded, cmap='gray')

plt.show()
```

---

#### Chapitre 5 : Couches de ReLU dans les CNN

##### 5.1 Fonction d'activation ReLU
- **Définition** : Fonction d'activation qui remplace toutes les valeurs négatives par zéro.
- **Importance** : Introduit de la non-linéarité et aide à résoudre le problème de gradient vanishing.

```python
# Exemple : Appliquer la fonction ReLU à une image convoluée
def relu(x):
    return np.maximum(0, x)

relu_image = relu(convoluted_image)
plt.imshow(relu_image, cmap='gray')
plt.show()
```

---

#### Chapitre 6 : Couches de Pooling dans les CNN

##### 6.1 Qu'est-ce que le pooling ?
- **Définition** : Opération de réduction de la dimensionnalité de l'image tout en conservant les caractéristiques importantes.
- **Types de pooling** :
  - **Max Pooling** : Prend la valeur maximale dans une fenêtre de sous-échantillonnage.
  - **Average Pooling** : Calcule la moyenne des valeurs dans la fenêtre de sous-échantillonnage.

```python
# Exemple : Appliquer le max pooling à une image
def max_pooling(image, size, stride):
    output_shape = (
        (image.shape[0] - size) // stride + 1,
        (image.shape[1] - size) // stride + 1
    )
    output = np.zeros(output_shape)
    for i in range(0, output_shape[0], stride):
        for j in range(0, output_shape[1], stride):
            output[i, j] = np.max(image[i:i+size, j:j+size])
    return output

pooled_image = max_pooling(relu_image, 2, 2)
plt.imshow(pooled_image, cmap='gray')
plt.show()
```

##### 6.2 Avantages du pooling
- **Invariance spatiale** : Réduit la sensibilité aux petites translations de l'image.
- **Réduction des paramètres** : Diminue le nombre de paramètres à apprendre, réduisant ainsi le risque de surapprentissage.

```python
# Exemple : Appliquer l'average pooling à une image
def average_pooling(image, size, stride):
    output_shape = (
        (image.shape[0] - size) // stride + 1,
        (image.shape[1] - size) // stride + 1
    )
    output = np.zeros(output_shape)
    for i in range(0, output_shape[0], stride):
        for j in range(0, output_shape[1], stride):
            output[i, j] = np.mean(image[i:i+size, j:j+size])
    return output

pooled_image_avg = average_pooling(relu_image, 2, 2)
plt.imshow(pooled_image_avg, cmap='gray')
plt.show()
```

---

#### Chapitre 7 : Couches Entièrement Connectées et Classification

##### 7.1 Couches entièrement connectées
- **Définition** : Chaque neurone est connecté à tous les neurones de la couche précédente.
- **Rôle** : Effectue la classification finale basée sur les caractéristiques extraites par les couches précédentes.

```python
# Exemple : Ajouter des couches entièrement connectées à un CNN
from keras.layers import Dense, Flatten

# Ajout de couches entièrement connectées à un modèle CNN
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

##### 7.2 Fonctionnement
- **Flattening** : Transformation des matrices 2D en vecteurs 1D pour les couches entièrement connectées.
- **Softmax** : Fonction d'activation utilisée en sortie pour produire des probabilités de classification.

```python
# Exemple : Utiliser la fonction softmax pour la classification
from keras.layers import Softmax

# Ajout de la couche Softmax pour la classification
model.add(Dense(10))
model.add(Softmax())
```

---

#### Chapitre 8 : Entraînement des CNN

##### 8.1 Processus d'entraînement
- **Dataset** : Ensemble d'images avec des labels associés.
- **Forward Propagation** : Calcul des sorties du réseau à partir des entrées.
- **Backpropagation** : Ajustement des poids et des biais pour minimiser l'erreur de classification.
- **Optimisation** : Utilisation d'algorithmes comme l'Adam ou le SGD pour ajuster les paramètres.

```python
# Exemple : Entraîner un modèle CNN avec Keras
from keras.datasets import mnist
from keras.utils import to_categorical

# Charger le dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
```

##### 8.2 Hyperparamètres
- **Taux d'apprentissage** : Détermine la taille des ajustements des poids.
- **Nombre d'époques** : Nombre de fois que l'ensemble des données d'entraînement est passé à travers le réseau.
- **Batch size** : Nombre d'échantillons traités avant la mise à jour des poids.

```python
# Exemple : Modifier les hyperparamètres lors de la compilation et de l'entraînement du modèle
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
```

---

#### Chapitre 9 : Applications des CNN

##### 9.1 Domaines d'application
- **Reconnaissance d'images** : Identification d'objets dans des images.
- **Analyse de vidéos** : Détection et suivi d'objets en mouvement.
- **Systèmes de recommandation** : Suggestions personnalisées basées sur les préférences visuelles.
- **Traitement du langage naturel** : Applications comme la reconnaissance d'écriture manuscrite.

```python
# Exemple : Utiliser un modèle CNN pré-entraîné pour la reconnaissance d'images
from keras.applications import VGG16

# Charger le modèle VGG16 pré-entraîné
vgg_model = VGG16(weights='imagenet')

# Charger une image et la préparer pour le modèle
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Faire des prédictions
preds = vgg_model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
```

---

#### Chapitre 10 : Avantages et Inconvénients des CNN

##### 10.1 Avantages
- **Efficacité** : Excellente performance pour les tâches de vision par ordinateur.
- **Automatisation** : Capacité à apprendre automatiquement des caractéristiques à partir des données brutes.

##### 10.2 Inconvénients
- **Complexité** : Nécessitent de grandes quantités de données pour l'entraînement.
- **Ressources** : Requiert des ressources matérielles importantes pour l'entraînement.

```python
# Exemple : Comparer les temps d'entraînement entre un modèle CNN et un modèle traditionnel
import time
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import mnist

# Charger le dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_flat = X_train.reshape(-1,

 28*28)
X_test_flat = X_test.reshape(-1, 28*28)

# Entraîner un modèle de forêt aléatoire
start_time = time.time()
rf_model = RandomForestClassifier()
rf_model.fit(X_train_flat, y_train)
rf_training_time = time.time() - start_time

# Entraîner un modèle CNN
start_time = time.time()
model.fit(X_train.reshape(-1, 28, 28, 1), to_categorical(y_train), epochs=10, batch_size=64)
cnn_training_time = time.time() - start_time

print(f"Temps d'entraînement RF: {rf_training_time:.2f} secondes")
print(f"Temps d'entraînement CNN: {cnn_training_time:.2f} secondes")
```

---

### Conclusion

Les réseaux de neurones convolutifs sont des outils puissants pour le traitement d'images et de vidéos, offrant des performances remarquables grâce à leur capacité à extraire automatiquement des caractéristiques hiérarchiques. Leur utilisation s'étend à de nombreux domaines, rendant les CNN essentiels pour les applications modernes de deep learning.

---

### Introduction à Keras

#### Qu'est-ce que Keras ?

Keras est une API de réseaux de neurones de haut niveau, écrite en Python et capable de s'exécuter sur des moteurs de calcul comme TensorFlow, CNTK, ou Theano. Elle permet de créer et d’entraîner des modèles de deep learning de manière simple et efficace.

#### Pourquoi utiliser Keras ?

1. **Facilité d’utilisation** : Keras est conçu pour être convivial et simple à utiliser, ce qui permet de se concentrer sur les aspects importants de la création de modèles de deep learning sans se perdre dans des détails techniques complexes.
2. **Modularité** : Keras est basé sur un modèle modulaire où chaque composant d'un modèle de deep learning est une entité distincte qui peut être combinée avec d'autres.
3. **Extensibilité** : Keras est hautement extensible et permet d'ajouter facilement de nouveaux modules ou de personnaliser ceux existants.
4. **Production** : Keras offre une compatibilité avec plusieurs backend, ce qui en fait un choix flexible pour différents environnements de production.

#### Environnement Keras

##### Architecture de Keras

- **Votre programme** : C’est le script Python que vous écrivez.
- **Keras (API NN)** : L’API qui vous permet de définir et d’entraîner vos modèles.
- **Backend (TensorFlow, Theano, CNTK)** : Le moteur de calcul qui effectue les opérations de bas niveau nécessaires pour entraîner les modèles.

##### Exemple de modèle avec Keras

```python
from keras.models import Sequential
from keras.layers import Dense

# Initialiser le modèle
model = Sequential()

# Ajouter des couches
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

# Compiler le modèle
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

---

### Cours : Créer Votre Premier Réseau de Neurones avec Keras

#### Introduction

Keras est une bibliothèque de réseaux de neurones haut niveau qui tourne sur TensorFlow. Il permet de construire et d'entraîner des modèles de Deep Learning facilement grâce à une interface simple et conviviale.

#### Importation de Keras

Keras est intégré dans TensorFlow, ce qui signifie que vous pouvez l'utiliser directement en l'important depuis TensorFlow :

```python
import tensorflow as tf
from tensorflow import keras
```

#### Installation de Keras

- **Keras intégré à TensorFlow** : Aucune installation séparée n'est nécessaire si vous utilisez TensorFlow.
- **Keras autonome** : Il est également possible d'installer Keras comme une bibliothèque autonome et de l'utiliser avec différents backends tels que TensorFlow, Theano, ou CNTK. 

Pour installer Keras de manière autonome :
```bash
pip install keras
```
Pour changer de backend, vous devez configurer le fichier `.keras/keras.json` dans votre répertoire utilisateur.

#### Création d'un Modèle Simple

Pour commencer, nous allons créer un modèle de réseau de neurones simple avec une couche d'entrée, une couche cachée et une couche de sortie.

##### 1. Importer les Bibliothèques Nécessaires

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

##### 2. Préparer les Données

Nous allons utiliser le dataset MNIST pour cette démonstration. Ce dataset contient des images de chiffres écrits à la main.

```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

##### 3. Construire le Modèle

Nous allons créer un modèle séquentiel avec une couche d'entrée, une couche cachée dense et une couche de sortie.

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
```

##### 4. Compiler le Modèle

Ensuite, nous compilons le modèle avec une fonction de perte, un optimiseur et des métriques.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

##### 5. Entraîner le Modèle

Nous entraînons le modèle sur les données d'entraînement.

```python
model.fit(x_train, y_train, epochs=5)
```

##### 6. Évaluer le Modèle

Enfin, nous évaluons le modèle sur les données de test.

```python
model.evaluate(x_test, y_test)
```

#### Changer de Backend

Keras permet de changer de backend facilement en modifiant le fichier de configuration `keras.json`.

```json
{
    "backend": "tensorflow",
    "floatx": "float32",
    "epsilon": 1e-07,
    "image_data_format": "channels_last"
}
```

### Conclusion

Keras simplifie la création et l'entraînement des modèles de Deep Learning. Son intégration avec TensorFlow permet une utilisation flexible et puissante, adaptée aux débutants comme aux experts.

---

### Construire des Modèles avec Keras

#### Introduction

Keras est une bibliothèque puissante et facile à utiliser pour construire des réseaux de neurones. Elle propose deux principaux types de modèles : le modèle Sequential et la classe Model avec l'API Fonctionnelle.

---

#### Modèle Sequential

**Caractéristiques clés** :
- **Facile à comprendre** : Le modèle Sequential est simple, ce qui en fait un excellent point de départ pour les débutants.
- **Séquence de couches** : Les couches sont ajoutées une par une, formant une pile linéaire.
- **Interconnexion automatique des couches** : Chaque couche est automatiquement connectée à la suivante.
- **Rapide et simple** : Idéal pour des architectures de réseaux de neurones simples.

---

#### Classe Model avec API Fonctionnelle

**Caractéristiques clés** :
- **Réseaux de neurones complexes** : Convient aux architectures de réseaux plus complexes.
- **Couches comme unités fonctionnelles** : Permet de traiter les couches comme des fonctions pouvant être connectées dans une structure de graphe.
- **Connexions de couches définies par l'utilisateur** : Donne un contrôle total sur les connexions entre les couches.
- **Détaillé et puissant** : Plus flexible et puissant que le modèle Sequential.

---

### Règles de base pour les couches cachées

**Nombre de couches cachées** :
- **0** : Représente uniquement des fonctions linéairement séparables.
- **1** : Peut mapper des fonctions continues d'un espace à un autre.
- **2** : Capable de représenter des frontières de décision arbitraires.
- **3 ou plus** : Peut apprendre des représentations complexes.

**Références** :
- Heaton, Jeff : [Heaton Research](http://www.heatonresearch.com/2017/06/01/hidden-layers.html)

---

### Règles de base pour les neurones dans les couches cachées

**Lignes directrices** :
- **>= taille de la couche d'entrée ET <= taille de la couche de sortie**
- **(2/3 * taille des couches d'entrée) + taille de la couche de sortie**
- **< 2 * taille de la couche d'entrée**

**Références** :
- Heaton, Jeff : [Heaton Research](http://www.heatonresearch.com/2017/06/01/hidden-layers.html)

---

### Visualisation

**Outils** :
- **plot_model()** : 
  - Fournit une représentation graphique du modèle.
  - Montre les connexions entre les couches.
  - Produit un fichier image de qualité pour les présentations.
- **summary()** :
  - Imprime un résumé du modèle.
  - Affiche les informations sur les couches, les formes, et le nombre de paramètres entraînables.

---

### Callbacks

**Fonctions** :
- **Collecter des informations d'entraînement** : Surveiller les progrès de l'entraînement à distance.
- **Ajuster les paramètres** : Ajuster les paramètres pendant l'entraînement.
- **Créer des points de contrôle** : Sauvegarder le modèle à des intervalles spécifiques.
- **Arrêter l

'entraînement prématurément** : Arrêter l'entraînement selon certaines conditions.
- **API pour callbacks définis par l'utilisateur** : Permet l'intégration de callbacks personnalisés dans le processus d'entraînement.

---

### Sauvegarder et Restaurer des Modèles

**Méthodes** :
- **.save(filepath)** : Sauvegarde le modèle dans un fichier HDF5.
- **.load_model(filepath)** : Charge un modèle depuis un fichier HDF5.
- **.model_to_json / .model_to_yaml** : Sauvegarde uniquement l'architecture de la couche du modèle en une chaîne JSON ou YAML.
- **.model_from_json / .model_from_yaml** : Charge l'architecture de la couche du modèle depuis une chaîne JSON ou YAML.
- **.save_weights(filepath)** : Sauvegarde uniquement les poids du modèle dans un fichier HDF5.
- **.load_weights(filepath)** : Charge uniquement les poids du modèle depuis un fichier HDF5.

---

### Exemples Pratiques

**Modèle Sequential** :
1. **Couche d'entrée** : La couche de départ du modèle.
2. **Couches cachées** : Couches intermédiaires qui apprennent à représenter les données.
3. **Couche de sortie** : La couche finale qui produit la sortie.

**API Fonctionnelle** :
1. **Entrées multiples** : Les modèles peuvent gérer plusieurs sources d'entrée.
2. **Couches itératives** : Les couches peuvent être appliquées de manière itérative.
3. **Architectures complexes** : Permet la création de structures de réseaux complexes avec des sorties multiples.

---

### Conclusion

Keras fournit des outils robustes pour construire des réseaux de neurones, que vous travailliez avec des modèles simples utilisant l'API Sequential ou des architectures plus complexes avec l'API Fonctionnelle. Comprendre les lignes directrices pour configurer les couches cachées et les neurones, utiliser les outils de visualisation, les callbacks, et sauvegarder/restaurer les modèles sont des compétences essentielles pour développer des modèles de deep learning efficaces.

**Méthodes de support** :
- **Visualisation** : Outils pour visualiser l'architecture du modèle.
- **Callbacks** : Mécanismes pour contrôler et surveiller l'entraînement.
- **Sauvegarde et restauration** : Méthodes pour sauvegarder et charger les architectures et les poids des modèles.

---

### Introduction aux Réseaux de Neurones Convolutionnels (CNN) avec Keras

#### 1. Vue d'ensemble des Réseaux de Neurones Convolutionnels

Les réseaux de neurones convolutionnels (CNN) sont une classe de réseaux de neurones profonds particulièrement efficaces pour analyser les images visuelles. Ils sont largement utilisés dans la reconnaissance d'images et de vidéos, les systèmes de recommandation et le traitement du langage naturel.

#### 2. Comprendre les Couches dans Keras

Keras est une bibliothèque de réseaux de neurones open-source puissante et facile à utiliser, écrite en Python. Elle peut fonctionner sur TensorFlow, Microsoft Cognitive Toolkit, Theano ou PlaidML. Elle permet un prototypage facile et rapide, prend en charge à la fois les réseaux convolutionnels et récurrents, et fonctionne sans problème sur CPU et GPU.

#### 3. Les Couches Communes dans Keras

- **Couches Denses** : Couches entièrement connectées où chaque neurone est connecté à tous les neurones de la couche précédente.
- **Couches de Dropout** : Technique de régularisation où des neurones sélectionnés aléatoirement sont ignorés pendant l'entraînement pour éviter le surapprentissage.
- **Couches de Reshape** : Modifient la forme de l'entrée sans en altérer les données.
- **Couches de Flatten** : Aplatissent l'entrée, la convertissant en un tableau à 1 dimension.
- **Couches de Permute** : Permutent les dimensions de l'entrée selon un schéma donné.
- **Couches de RepeatVector** : Répètent l'entrée un certain nombre de fois.

#### 4. Les Couches Convolutionnelles

Les couches convolutionnelles sont les blocs de construction fondamentaux d'un CNN. Elles appliquent une opération de convolution à l'entrée, passant le résultat à la couche suivante.

- **Couches Convolutionnelles** : Effectuent des convolutions, qui combinent des filtres apprenables avec des données d'entrée pour produire des cartes de caractéristiques.
- **Couches de Pooling** : Réduisent la dimensionnalité de chaque carte de caractéristiques tout en conservant les informations les plus importantes. Les types courants incluent le Max Pooling et le Average Pooling.

#### 5. Construire un Réseau de Neurones Convolutionnel avec Keras

Construisons un CNN simple en utilisant Keras.

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Initialiser le CNN
model = Sequential()

# Étape 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Étape 2 - Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Ajouter une deuxième couche convolutionnelle
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Étape 3 - Flattening
model.add(Flatten())

# Étape 4 - Connection Complète
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Compiler le CNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

#### 6. Entraîner le CNN

Pour entraîner le CNN, vous devez préparer votre jeu de données, ce qui implique :

- Diviser les données en ensembles d'entraînement et de test.
- Normaliser les images.
- Appliquer des techniques d'augmentation de données pour augmenter la diversité de vos données d'entraînement sans collecter de nouvelles données.

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

model.fit_generator(training_set,
                     steps_per_epoch=8000/32,
                     epochs=25,
                     validation_data=test_set,
                     validation_steps=2000/32)
```

#### 7. Évaluer le Modèle

Après l'entraînement, évaluez la performance du modèle sur le jeu de test pour comprendre sa capacité à se généraliser à de nouvelles données.

```python
score = model.evaluate(test_set)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

#### 8. Conclusion

Les réseaux de neurones convolutionnels sont un outil puissant pour la reconnaissance d'images et d'autres tâches impliquant des données de type grille. Keras fournit une API de haut niveau, facile à utiliser pour construire et entraîner ces réseaux de manière efficace.

#### Références
- [Documentation Keras](https://keras.io)
- [Building Convolutional NN with Keras (Vidéo)]
- [Employing Layers in Keras Models (Diapositives)]

---

### Introduction aux Réseaux Neuronaux Convolutifs (CNN)

#### Définition
Un réseau neuronal convolutif (CNN, ou ConvNet) est une classe de réseaux neuronaux profonds, feed-forward, qui a été appliquée avec succès à l'analyse des images visuelles.

#### Problème de la Dimensionalité
- **Entrées** : Les images peuvent contenir des millions de pixels, ce qui entraîne un nombre élevé de poids à entraîner.
- **Exemple** : Une image de 8 millions de pixels avec 1000 neurones entraîne 8 milliards de poids, et avec 3 couleurs, on atteint 24 milliards de poids.

#### Invariance par Translation
- **Objectif** : Réduire le nombre de poids à entraîner tout en maintenant la capacité de détecter des objets généraux dans les images.

---

### Composants d'un CNN

#### Convolution
- **Fonctionnement** : La convolution extrait des caractéristiques des images tout en préservant les relations spatiales des caractéristiques (comme les bords et les éléments composites tels que les yeux ou le nez).
- **Hyperparamètres clés** :
  - **Taille du noyau (Kernel Size)** : Détermine les pixels liés.
  - **Nombre de filtres** : Détecte différentes caractéristiques.
  - **Stride** : Distance de déplacement du filtre (valeurs plus grandes réduisent la taille de la carte de caractéristiques et l'information transmise à la couche suivante).
  - **Padding** : Ajoute des pixels autour de l'image pour préserver les dimensions après la convolution.

#### Activation Non-Linéaire (ReLU)
- **Fonction** : Ajoutée après une couche de convolution pour introduire la non-linéarité, empêchant le problème du gradient qui disparaît.
- **Formule** : \( y = \max(0,

 x) \)

#### Pooling (Sous-échantillonnage)
- **Objectif** : Réduire la dimensionnalité tout en maintenant les caractéristiques importantes.
- **Types** :
  - **Max Pooling** : Prend la valeur maximale dans une fenêtre de taille définie (ex. 2x2).

---

### Exemple Pratique avec MNIST et Fashion MNIST

#### MNIST
- **Description** : Jeu de données contenant 60 000 images d'entraînement et 10 000 images de test de chiffres manuscrits (28x28 pixels, en niveaux de gris).
- **Performance** : Un CNN basique peut atteindre une précision supérieure à 99 %.

#### Fashion MNIST
- **Description** : Jeu de données similaire à MNIST mais contenant des images de vêtements avec 10 classes différentes.
- **Défi** : Un CNN basique peut atteindre une précision de 90 %, et plus avec un ajustement.

---

### Apprentissage par Transfert

#### Concept
- **Objectif** : Utiliser un modèle pré-entraîné sur un large jeu de données pour améliorer la performance sur un problème spécifique.
- **Étapes** :
  - Utiliser un modèle existant pour la détection de caractéristiques.
  - Remplacer le classificateur final par un classificateur adapté à notre problème.
  - Entraîner ce nouveau classificateur sur notre jeu de données spécifique.

#### Exemple avec Inception V3
- **Problèmes de Formation** : Les grands modèles comme Inception nécessitent beaucoup de données et de temps pour l'entraînement.
- **Solution** : Apprentissage par transfert pour tirer parti de la puissance de ces modèles pré-entraînés.

---

### Structure des Dossiers de Données

#### Organisation
- **Emplacement du Programme** : Le dossier `data` contient les sous-dossiers `train` et `validate`.
- **Données d'Entraînement** : 
  - `train/cats` : 1000 images de chats.
  - `train/dogs` : 1000 images de chiens.
- **Données de Validation** :
  - `validate/cats` : 400 images de chats.
  - `validate/dogs` : 400 images de chiens.
- **Source des Données** : Les données peuvent être téléchargées depuis [Kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) (train.zip).

---

### Synthèse

- **Résumé** : Les CNNs résolvent les problèmes d'analyse d'images en extrayant des cartes de caractéristiques et en utilisant des couches de convolution, de non-linéarité et de pooling.
- **Exemples** : Démonstration avec Fashion MNIST, et utilisation de l'apprentissage par transfert avec Inception V3.


# Annexe : 

### Livre sur les Réseaux de Neurones Convolutifs (CNN) avec Keras

---

#### Introduction

Les réseaux de neurones convolutifs (CNN) sont une classe de réseaux neuronaux profonds particulièrement efficaces pour analyser les images visuelles. Ils sont largement utilisés dans la reconnaissance d'images et de vidéos, les systèmes de recommandation et le traitement du langage naturel. Ce livre présente une vue d'ensemble des CNN, les concepts clés et des exemples pratiques avec Keras pour vous aider à démarrer avec la création de modèles de deep learning.

---

### Chapitre 1 : Introduction aux Images Numériques

#### 1.1 Qu'est-ce qu'une image numérique ?
Une image numérique est une représentation visuelle d'une scène sous forme de grille de pixels. Chaque pixel contient des informations sur la couleur ou la luminosité.

**Exemple de code : Afficher une image en niveaux de gris**
```python
import matplotlib.pyplot as plt
import numpy as np

image = np.array([
    [0, 30, 60, 90, 120, 150, 180, 210],
    [30, 60, 90, 120, 150, 180, 210, 240],
    [60, 90, 120, 150, 180, 210, 240, 255],
    [90, 120, 150, 180, 210, 240, 255, 255],
    [120, 150, 180, 210, 240, 255, 255, 255],
    [150, 180, 210, 240, 255, 255, 255, 255],
    [180, 210, 240, 255, 255, 255, 255, 255],
    [210, 240, 255, 255, 255, 255, 255, 255]
])

plt.imshow(image, cmap='gray')
plt.colorbar()
plt.show()
```

#### 1.2 Types d'images numériques
- **Images en niveaux de gris** : Chaque pixel est une valeur de luminosité allant de 0 (noir) à 255 (blanc).
- **Images en couleurs** : Chaque pixel est représenté par trois valeurs correspondant aux canaux de couleur rouge, vert et bleu (RVB).

**Exemple de code : Afficher une image en couleurs**
```python
import matplotlib.pyplot as plt
import numpy as np

image = np.zeros((8, 8, 3), dtype=np.uint8)
image[..., 0] = np.linspace(0, 255, 8)  # Canal rouge
image[..., 1] = np.linspace(0, 255, 8)[:, np.newaxis]  # Canal vert
image[..., 2] = 128  # Canal bleu

plt.imshow(image)
plt.show()
```

#### 1.3 Résolution d'une image
La résolution d'une image est déterminée par le nombre de pixels horizontaux et verticaux.

**Exemple de code : Créer une image de 128x128 pixels**
```python
import numpy as np

image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
print(f"Nombre de pixels dans l'image : {image.size}")
```

---

### Chapitre 2 : Les Bases des Réseaux de Neurones

#### 2.1 Qu'est-ce qu'un réseau de neurones ?
Un réseau de neurones est un modèle mathématique inspiré du cerveau humain, conçu pour reconnaître des motifs et apprendre à partir de données.

#### 2.2 Neurone artificiel
Chaque neurone reçoit des entrées, les transforme à l'aide de poids et de biais, applique une fonction d'activation, puis produit une sortie.

**Exemple de code : Neurone simple avec fonction d'activation sigmoïde**
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

inputs = np.array([1.0, 2.0, 3.0])
weights = np.array([0.2, 0.8, -0.5])
bias = 2.0

output = sigmoid(np.dot(inputs, weights) + bias)
print(f"Sortie du neurone : {output}")
```

#### 2.3 Couches d'un réseau de neurones
- **Couche d'entrée** : Reçoit les données brutes.
- **Couches cachées** : Effectuent des transformations et extraient des caractéristiques.
- **Couche de sortie** : Produit le résultat final, par exemple, la classe prédite d'une image.

**Exemple de code : Réseau de neurones simple avec une couche cachée**
```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=3, activation='relu'))  # Couche cachée
model.add(Dense(1, activation='sigmoid'))  # Couche de sortie

model.summary()
```

---

### Chapitre 3 : Introduction aux Réseaux de Neurones Convolutifs (CNN)

#### 3.1 Historique des CNN
Les CNN sont inspirés par le fonctionnement du cortex visuel des mammifères et ont été introduits par Yann LeCun et al. en 1998 pour la classification du dataset MNIST.

#### 3.2 Structure d'un CNN
- **Couche d'entrée** : Reçoit l'image sous forme de matrice de pixels.
- **Couches de convolution** : Appliquent des filtres pour extraire des caractéristiques locales.
- **Couches de ReLU** : Introduisent de la non-linéarité en remplaçant les valeurs négatives par zéro.
- **Couches de pooling** : Réduisent la dimensionnalité de l'image tout en conservant les informations essentielles.
- **Couches entièrement connectées** : Effectuent la classification en se basant sur les caractéristiques extraites.

**Exemple de code : Construire un simple CNN avec Keras**
```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # Couche de convolution
model.add(MaxPooling2D((2, 2)))  # Couche de pooling
model.add(Flatten())  # Couche de mise à plat
model.add(Dense(64, activation='relu'))  # Couche entièrement connectée
model.add(Dense(10, activation='softmax'))  # Couche de sortie

model.summary()
```

---

### Chapitre 4 : Convolution dans les CNN

#### 4.1 Qu'est-ce que la convolution ?
La convolution est une opération mathématique qui combine deux fonctions pour produire une troisième fonction. Elle est utilisée dans les CNN pour appliquer des filtres sur les images afin d'extraire des caractéristiques spécifiques.

**Exemple de code : Appliquer un filtre de convolution à une image**
```python
from scipy.ndimage import convolve

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

convoluted_image = convolve(image, kernel)
plt.imshow(convoluted_image, cmap='gray')
plt.show()
```

#### 4.2 Filtre (Kernel)
Un filtre est une petite matrice utilisée pour balayer l'image et effectuer des opérations de convolution.

**Exemple de code : Définir et appliquer un filtre 3x3**
```python
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

convoluted_image = convolve(image, kernel)
plt.imshow(convoluted_image, cmap='gray')
plt.show()
```

#### 4.3 Stride
Le stride est le nombre de pixels par lesquels le filtre se déplace sur l'image. Un stride plus grand réduit la dimension de l'image de sortie.

**Exemple de code : Appliquer la convolution avec différents strides**
```python
def apply_convolution(image, kernel, stride):
    output_shape = (
        (image.shape[0] - kernel.shape[0]) // stride + 1,
        (image.shape[1] - kernel.shape[1]) // stride + 1
    )
    output = np.zeros(output_shape)
    for i in range(0, output_shape[0], stride):
        for j in range(0, output_shape[1], stride):
            output[i, j] = np.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return output

convoluted_image_stride1 = apply_convolution(image, kernel, 1)
convoluted_image_stride2 = apply_convolution(image, kernel, 2)

plt.subplot(1, 2, 1)
plt.title('Stride 1')
plt.imshow(convoluted_image_stride1, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Stride 2')
plt.imshow(convoluted_image_stride2, cmap='gray')

plt.show()
```

#### 4.4 Padding
Le padding consiste à ajouter des bordures de pixels autour de l'image pour conserver sa dimension après la convolution.

**Exemple de code : Appliquer la convolution avec et sans padding**
```python
def apply_padding(image, padding_size):
    padded_image = np.pad(image

, pad_width=padding_size, mode='constant', constant_values=0)
    return padded_image

padded_image = apply_padding(image, 1)
convoluted_image_padded = convolve(padded_image, kernel)

plt.subplot(1, 2, 1)
plt.title('Sans Padding')
plt.imshow(convoluted_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Avec Padding')
plt.imshow(convoluted_image_padded, cmap='gray')

plt.show()
```

---

### Chapitre 5 : Couches de ReLU dans les CNN

#### 5.1 Fonction d'activation ReLU
La fonction d'activation ReLU remplace toutes les valeurs négatives par zéro, introduisant ainsi de la non-linéarité et aidant à résoudre le problème de gradient vanishing.

**Exemple de code : Appliquer la fonction ReLU à une image convoluée**
```python
def relu(x):
    return np.maximum(0, x)

relu_image = relu(convoluted_image)
plt.imshow(relu_image, cmap='gray')
plt.show()
```

---

### Chapitre 6 : Couches de Pooling dans les CNN

#### 6.1 Qu'est-ce que le pooling ?
Le pooling est une opération de réduction de la dimensionnalité de l'image tout en conservant les caractéristiques importantes. Les types courants incluent le max pooling et l'average pooling.

**Exemple de code : Appliquer le max pooling à une image**
```python
def max_pooling(image, size, stride):
    output_shape = (
        (image.shape[0] - size) // stride + 1,
        (image.shape[1] - size) // stride + 1
    )
    output = np.zeros(output_shape)
    for i in range(0, output_shape[0], stride):
        for j in range(0, output_shape[1], stride):
            output[i, j] = np.max(image[i:i+size, j:j+size])
    return output

pooled_image = max_pooling(relu_image, 2, 2)
plt.imshow(pooled_image, cmap='gray')
plt.show()
```

#### 6.2 Avantages du pooling
- **Invariance spatiale** : Réduit la sensibilité aux petites translations de l'image.
- **Réduction des paramètres** : Diminue le nombre de paramètres à apprendre, réduisant ainsi le risque de surapprentissage.

**Exemple de code : Appliquer l'average pooling à une image**
```python
def average_pooling(image, size, stride):
    output_shape = (
        (image.shape[0] - size) // stride + 1,
        (image.shape[1] - size) // stride + 1
    )
    output = np.zeros(output_shape)
    for i in range(0, output_shape[0], stride):
        for j in range(0, output_shape[1], stride):
            output[i, j] = np.mean(image[i:i+size, j:j+size])
    return output

pooled_image_avg = average_pooling(relu_image, 2, 2)
plt.imshow(pooled_image_avg, cmap='gray')
plt.show()
```

---

### Chapitre 7 : Couches Entièrement Connectées et Classification

#### 7.1 Couches entièrement connectées
Les couches entièrement connectées, où chaque neurone est connecté à tous les neurones de la couche précédente, jouent un rôle clé dans la classification finale basée sur les caractéristiques extraites par les couches précédentes.

**Exemple de code : Ajouter des couches entièrement connectées à un CNN**
```python
from keras.layers import Dense, Flatten

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

#### 7.2 Fonctionnement
La mise à plat (flattening) transforme les matrices 2D en vecteurs 1D pour les couches entièrement connectées. La fonction softmax est utilisée en sortie pour produire des probabilités de classification.

**Exemple de code : Utiliser la fonction softmax pour la classification**
```python
from keras.layers import Softmax

model.add(Dense(10))
model.add(Softmax())
```

---

### Chapitre 8 : Entraînement des CNN

#### 8.1 Processus d'entraînement
L'entraînement d'un CNN implique plusieurs étapes clés :
- **Dataset** : Ensemble d'images avec des labels associés.
- **Forward Propagation** : Calcul des sorties du réseau à partir des entrées.
- **Backpropagation** : Ajustement des poids et des biais pour minimiser l'erreur de classification.
- **Optimisation** : Utilisation d'algorithmes comme l'Adam ou le SGD pour ajuster les paramètres.

**Exemple de code : Entraîner un modèle CNN avec Keras**
```python
from keras.datasets import mnist
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
```

#### 8.2 Hyperparamètres
Les hyperparamètres clés incluent le taux d'apprentissage, le nombre d'époques et la taille des lots.

**Exemple de code : Modifier les hyperparamètres lors de la compilation et de l'entraînement du modèle**
```python
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
```

---

### Chapitre 9 : Applications des CNN

#### 9.1 Domaines d'application
Les CNN sont utilisés dans divers domaines, notamment :
- **Reconnaissance d'images** : Identification d'objets dans des images.
- **Analyse de vidéos** : Détection et suivi d'objets en mouvement.
- **Systèmes de recommandation** : Suggestions personnalisées basées sur les préférences visuelles.
- **Traitement du langage naturel** : Applications comme la reconnaissance d'écriture manuscrite.

**Exemple de code : Utiliser un modèle CNN pré-entraîné pour la reconnaissance d'images**
```python
from keras.applications import VGG16

vgg_model = VGG16(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = vgg_model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
```

---

### Chapitre 10 : Avantages et Inconvénients des CNN

#### 10.1 Avantages
Les CNN offrent une excellente performance pour les tâches de vision par ordinateur et la capacité à apprendre automatiquement des caractéristiques à partir des données brutes.

#### 10.2 Inconvénients
Les CNN nécessitent de grandes quantités de données pour l'entraînement et requièrent des ressources matérielles importantes.

**Exemple de code : Comparer les temps d'entraînement entre un modèle CNN et un modèle traditionnel**
```python
import time
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_flat = X_train.reshape(-1, 28*28)
X_test_flat = X_test.reshape(-1, 28*28)

start_time = time.time()
rf_model = RandomForestClassifier()
rf_model.fit(X_train_flat, y_train)
rf_training_time = time.time() - start_time

start_time = time.time()
model.fit(X_train.reshape(-1, 28, 28, 1), to_categorical(y_train), epochs=10, batch_size=64)
cnn_training_time = time.time() - start_time

print(f"Temps d'entraînement RF: {rf_training_time:.2f} secondes")
print(f"Temps d'entraînement CNN: {cnn_training_time:.2f} secondes")
```

---

### Conclusion

Les réseaux de neurones convolutifs sont des outils puissants pour le traitement d'images et de vidéos, offrant des performances remarquables grâce à leur capacité à extraire automatiquement des caractéristiques hiérarchiques. Leur utilisation s'étend à de nombreux domaines, rendant les CNN essentiels pour les applications modernes de deep learning.

---

Les informations détaillées ci-dessus sont basées sur les documents fournis, notamment le fichier PDF "3-How_CNNs_work" et le fichier PowerPoint "cours_3_cnn.pptx"  .

---

### Références
- [Documentation Keras](https://keras.io)
- [Building Convolutional NN with Keras (Vidéo)]
- [Employing Layers in Keras Models (Diapositives)]
