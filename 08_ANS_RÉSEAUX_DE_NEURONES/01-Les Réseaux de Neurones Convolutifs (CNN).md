# Les Réseaux de Neurones Convolutifs (CNN)

### Introduction
- Dans le cadre de ce cours sur l'apprentissage non supervisé, il est essentiel de comprendre les réseaux de neurones convolutifs (CNN). Une solide compréhension des CNN est indispensable, surtout pour les applications en imagerie. Les CNN sont des outils puissants pour extraire et représenter les caractéristiques complexes des images, une compétence clé pour appliquer efficacement des techniques non supervisées comme les autoencodeurs et le clustering. Nous aurons une partie dédiée à l'imagerie dans ce cours, où nous explorerons comment appliquer des techniques d'apprentissage non supervisé aux images.

# PLAN
- [**Chapitre 1 : Introduction aux Images Numériques**](#chapitre-1--introduction-aux-images-numériques)
  - [1.1 Qu'est-ce qu'une image numérique ?](#11-quest-ce-quune-image-numerique)
  - [1.2 Types d'images numériques](#12-types-dimages-numériques)
  - [1.3 Résolution d'une image](#13-résolution-dune-image)

- [**Chapitre 2 : Les Bases des Réseaux de Neurones**](#chapitre-2--les-bases-des-réseaux-de-neurones)
  - [2.1 Qu'est-ce qu'un réseau de neurones ?](#21-quest-ce-quun-réseau-de-neurones)
  - [2.2 Neurone artificiel](#22-neurone-artificiel)
  - [2.3 Couches d'un réseau de neurones](#23-couches-dun-réseau-de-neurones)

- [**Chapitre 3 : Introduction aux Réseaux de Neurones Convolutifs (CNN)**](#chapitre-3--introduction-aux-réseaux-de-neurones-convolutifs-cnn)
  - [3.1 Historique des CNN](#31-historique-des-cnn)
  - [3.2 Structure d'un CNN](#32-structure-dun-cnn)

- [**Chapitre 4 : Convolution dans les CNN**](#chapitre-4--convolution-dans-les-cnn)
  - [4.1 Qu'est-ce que la convolution ?](#41-quest-ce-que-la-convolution)
  - [4.2 Filtre (Kernel)](#42-filtre-kernel)
  - [4.3 Stride](#43-stride)
  - [4.4 Padding](#44-padding)

- [**Chapitre 5 : Couches de ReLU dans les CNN**](#chapitre-5--couches-de-relu-dans-les-cnn)
  - [5.1 Fonction d'activation ReLU](#51-fonction-dactivation-relu)

- [**Chapitre 6 : Couches de Pooling dans les CNN**](#chapitre-6--couches-de-pooling-dans-les-cnn)
  - [6.1 Qu'est-ce que le pooling ?](#61-quest-ce-que-le-pooling)
  - [6.2 Avantages du pooling](#62-avantages-du-pooling)

- [**Chapitre 7 : Couches Entièrement Connectées et Classification**](#chapitre-7--couches-entièrement-connectées-et-classification)
  - [7.1 Couches entièrement connectées](#71-couches-entièrement-connectées)
  - [7.2 Fonctionnement](#72-fonctionnement)

- [**Chapitre 8 : Entraînement des CNN**](#chapitre-8--entraînement-des-cnn)
  - [8.1 Processus d'entraînement](#81-processus-dentraînement)
  - [8.2 Hyperparamètres](#82-hyperparamètres)

- [**Chapitre 9 : Applications des CNN**](#chapitre-9--applications-des-cnn)
  - [9.1 Domaines d'application](#91-domaines-dapplication)

- [**Chapitre 10 : Avantages et Inconvénients des CNN**](#chapitre-10--avantages-et-inconvénients-des-cnn)
  - [10.1 Avantages](#101-avantages)
  - [10.2 Inconvénients](#102-inconvénients)

- [**Conclusion de la première partie**](#conclusion-de-la-première-partie)

- [**Chapitre 11 : Introduction à Keras**](#chapitre-11--introduction-à-keras)
  - [Qu'est-ce que Keras ?](#111-quest-ce-que-keras)
  - [Pourquoi utiliser Keras ?](#112-pourquoi-utiliser-keras)
  - [Environnement Keras](#113-environnement-keras)

- [**Chapitre 12 : Créer Votre Premier Réseau de Neurones avec Keras**](#chapitre-12--créer-votre-premier-réseau-de-neurones-avec-keras)
  - [Introduction](#121-introduction)
  - [Importation de Keras](#122-importation-de-keras)
  - [Installation de Keras](#123-installation-de-keras)
  - [Création d'un Modèle Simple](#124-création-dun-modèle-simple)
  - [Changer de Backend](#125-changer-de-backend)

- [**Chapitre 13 : Construire des Modèles avec Keras**](#chapitre-13--construire-des-modèles-avec-keras)
  - [Modèle Sequential](#131-modèle-sequential)
  - [Classe Model avec API Fonctionnelle](#132-classe-model-avec-api-fonctionnelle)

- [**Chapitre 14 : Introduction aux Réseaux de Neurones Convolutionnels (CNN) avec Keras**](#chapitre-14--introduction-aux-réseaux-de-neurones-convolutionnels-cnn-avec-keras)
  - [Vue d'ensemble des Réseaux de Neurones Convolutionnels](#141-vue-densemble-des-réseaux-de-neurones-convolutionnels)
  - [Comprendre les Couches dans Keras](#142-comprendre-les-couches-dans-keras)
  - [Les Couches Communes dans Keras](#143-les-couches-communes-dans-keras)
  - [Les Couches Convolutionnelles](#144-les-couches-convolutionnelles)
  - [Construire un Réseau de Neurones Convolutionnel avec Keras](#145-construire-un-réseau-de-neurones-convolutionnel-avec-keras)
  - [Entraîner le CNN](#146-entraîner-le-cnn)
  - [Évaluer le Modèle](#147-évaluer-le-modèle)

- [**Chapitre 15 : Composants d'un CNN**](#chapitre-15--composants-dun-cnn)
  - [Convolution](#151-convolution)
  - [Activation Non-Linéaire (ReLU)](#152-activation-non-linéaire-relu)
  - [Pooling (Sous-échantillonnage)](#153-pooling-sous-échantillonnage)

- [**Chapitre 16 : Exemple Pratique avec MNIST et Fashion MNIST**](#chapitre-16--exemple-pratique-avec-mnist-et-fashion-mnist)
  - [MNIST](#161-mnist)
  - [Fashion MNIST](#162-fashion-mnist)

- [**Chapitre 17 : Apprentissage par Transfert**](#chapitre-17--apprentissage-par-transfert)
  - [Concept](#171-concept)
  - [Exemple avec Inception V3](#172-exemple-avec-inception-v3)

- [**Structure des Dossiers de Données**](#structure-des-dossiers-de-données)
- [**Synthèse**](#synthèse)


# Chapitre 1 : Introduction aux Images Numériques

### 1.1 Qu'est-ce qu'une image numérique ?
[Retour en haut](#plan)
- **Définition** : Une image numérique est une représentation visuelle d'une scène sous forme de grille de pixels.
- **Pixel** : L'élément de base d'une image numérique. Chaque pixel contient des informations sur la couleur ou la luminosité.

#### Exercices :

1. **Afficher une image en niveaux de gris** :
   ```python
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

2. **Convertir une image couleur en niveaux de gris** :
   ```python
   import cv2

   # Charger une image couleur
   image_color = cv2.imread('path/to/your/image.jpg')

   # Convertir l'image en niveaux de gris
   image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

   # Afficher l'image en niveaux de gris
   plt.imshow(image_gray, cmap='gray')
   plt.colorbar()
   plt.show()
   ```

3. **Créer une image binaire à partir d'une image en niveaux de gris** :
   ```python
   # Définir un seuil pour la binarisation
   threshold = 128

   # Créer une image binaire
   image_binary = (image_gray > threshold).astype(np.uint8) * 255

   # Afficher l'image binaire
   plt.imshow(image_binary, cmap='gray')
   plt.colorbar()
   plt.show()
   ```

4. **Manipuler les pixels d'une image** :
   ```python
   # Créer une image 8x8 avec des valeurs aléatoires
   image_random = np.random.randint(0, 256, (8, 8))

   # Afficher l'image aléatoire
   plt.imshow(image_random, cmap='gray')
   plt.colorbar()
   plt.show()

   # Modifier un pixel spécifique
   image_random[4, 4] = 255

   # Afficher l'image modifiée
   plt.imshow(image_random, cmap='gray')
   plt.colorbar()
   plt.show()
   ```

### 1.2 Types d'images numériques
[Retour en haut](#plan)
- **Images en niveaux de gris** : Chaque pixel est une valeur de luminosité allant de 0 (noir) à 255 (blanc).
- **Images en couleurs** : Chaque pixel est représenté par trois valeurs correspondant aux canaux de couleur rouge, vert et bleu (RVB).

#### Exemples :

1. **Afficher une image en couleurs**
```python
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

2. **Convertir une image en niveaux de gris**
```python
# Charger une image en couleurs et la convertir en niveaux de gris
from skimage import color, data

image = data.astronaut()  # Charger une image de démonstration
gray_image = color.rgb2gray(image)

plt.imshow(gray_image, cmap='gray')
plt.colorbar()
plt.show()
```

#### Exercices :

1. **Exercice 1 : Créer une image en niveaux de gris**
   - Créez une image de 10x10 pixels avec une transition de noir à blanc.
   - Affichez l'image en utilisant `matplotlib`.

2. **Exercice 2 : Manipuler les canaux de couleur**
   - Créez une image de 10x10 pixels.
   - Définissez le canal rouge avec une valeur croissante de 0 à 255.
   - Définissez le canal vert avec une valeur décroissante de 255 à 0.
   - Définissez le canal bleu avec une valeur constante de 128.
   - Affichez l'image.

3. **Exercice 3 : Conversion de couleur à niveaux de gris**
   - Chargez une image en couleurs à partir d'un fichier local ou utilisez une image de démonstration de `skimage`.
   - Convertissez l'image en niveaux de gris et affichez les deux images côte à côte pour comparer.

### 1.3 Résolution d'une image
[Retour en haut](#plan)
- **Définition** : La résolution d'une image est déterminée par le nombre de pixels horizontaux et verticaux.
- **Exemple** : Une image de 128 x 128 pixels contient 16 384 pixels.

#### Exemples :

1. **Créer une image de 128x128 pixels**
```python
import numpy as np

# Créer une image de 128x128 pixels avec des valeurs aléatoires
image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
print(f"Nombre de pixels dans l'image : {image.size}")
```

2. **Redimensionner une image**
```python
from skimage.transform import resize
from skimage import data

# Charger une image de démonstration
image = data.camera()

# Redimensionner l'image à 64x64 pixels
resized_image = resize(image, (64, 64), anti_aliasing=True)

plt.subplot(1, 2, 1)
plt.title('Originale')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Redimensionnée')
plt.imshow(resized_image, cmap='gray')

plt.show()
```

#### Exercices :

1. **Exercice 1 : Calculer la résolution d'une image**
   - Chargez une image à partir d'un fichier local.
   - Affichez les dimensions de l'image (hauteur, largeur) et calculez le nombre total de pixels.

2. **Exercice 2 : Redimensionner une image**
   - Chargez une image de démonstration.
   - Redimensionnez l'image à différentes résolutions (32x32, 64x64, 128x128).
   - Affichez les images redimensionnées côte à côte pour observer les différences de résolution.

3. **Exercice 3 : Créer une image à haute résolution**
   - Créez une image de 256x256 pixels avec des valeurs aléatoires.
   - Affichez l'image en utilisant `matplotlib`.

---

#### Chapitre 2 : Les Bases des Réseaux de Neurones

##### 2.1 Qu'est-ce qu'un réseau de neurones ?
[Retour en haut](#plan)
- **Définition** : Un réseau de neurones est un modèle mathématique inspiré du cerveau humain, conçu pour reconnaître des motifs et apprendre à partir de données. Un réseau de neurones est constitué d'unités appelées neurones artificiels, organisés en couches (couches d'entrée, couches cachées et couche de sortie).

**Exercice 1 : Compréhension des Réseaux de Neurones**
1. Décrivez en vos propres mots ce qu'est un réseau de neurones et ses principales composantes.
2. Pourquoi les réseaux de neurones sont-ils inspirés du cerveau humain ?
3. Quels sont les avantages des réseaux de neurones par rapport aux autres méthodes d'apprentissage automatique ?

##### 2.2 Neurone artificiel
[Retour en haut](#plan)
- **Fonctionnement** : Chaque neurone reçoit des entrées, les transforme à l'aide de poids et de biais, applique une fonction d'activation, puis produit une sortie. 
- **Fonction d'activation** : Une fonction mathématique qui introduit de la non-linéarité, essentielle pour la modélisation de relations complexes. Les fonctions d'activation courantes incluent la sigmoïde, ReLU (Rectified Linear Unit), et tanh.

**Exemple détaillé : Neurone simple avec fonction d'activation sigmoïde**
```python
import numpy as np

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

**Exercice 2 : Implémentation d'un Neurone**
1. Implémentez un neurone artificiel avec une fonction d'activation tanh.
2. Modifiez le code pour tester avec différentes valeurs de poids et biais.
3. Expliquez comment les changements de poids et de biais affectent la sortie du neurone.

**Exemple avancé : Comparaison des fonctions d'activation**
```python
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Créer une gamme de valeurs d'entrée
x = np.linspace(-10, 10, 100)

# Calculer les sorties pour chaque fonction d'activation
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

# Tracer les résultats
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_tanh, label='Tanh')
plt.plot(x, y_relu, label='ReLU')
plt.legend()
plt.title('Comparaison des fonctions d\'activation')
plt.xlabel('Entrée')
plt.ylabel('Sortie')
plt.show()
```

**Exercice 3 : Analyse des Fonctions d'Activation**
1. Tracez et comparez les fonctions d'activation ReLU, Sigmoid et Tanh.
2. Pour quelles applications chaque fonction d'activation est-elle la plus adaptée ?
3. Pourquoi est-il important d'introduire de la non-linéarité dans un réseau de neurones ?

**Exercice 4 : Création d'un Réseau de Neurones Simple**
1. Créez un réseau de neurones à une couche cachée en utilisant la bibliothèque Keras.
2. Utilisez une fonction d'activation ReLU pour la couche cachée et une fonction d'activation softmax pour la couche de sortie.
3. Entraînez le réseau sur un ensemble de données simple (par exemple, MNIST).

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical

# Charger les données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Créer le modèle
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(28*28,)))
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
```

**Exercice 5 : Expérimentation avec des Hyperparamètres**
1. Modifiez les hyperparamètres du modèle (nombre de neurones, taux d'apprentissage, nombre d'époques) et observez l'impact sur la performance.
2. Documentez vos observations et tirez des conclusions sur l'importance des hyperparamètres.

##### 2.3 Couches d'un réseau de neurones
[Retour en haut](#plan)

- **Couche d'entrée** : La couche d'entrée est la première couche d'un réseau de neurones. Elle reçoit les données brutes et les passe aux couches suivantes. Le nombre de neurones dans la couche d'entrée correspond au nombre de caractéristiques dans les données d'entrée.

- **Couches cachées** : Les couches cachées se situent entre la couche d'entrée et la couche de sortie. Elles effectuent des transformations sur les données d'entrée pour extraire des caractéristiques et des informations pertinentes. Un réseau peut avoir plusieurs couches cachées, chacune avec un nombre différent de neurones et des fonctions d'activation variées.

- **Couche de sortie** : La couche de sortie est la dernière couche d'un réseau de neurones. Elle produit le résultat final, par exemple, la classe prédite d'une image ou une valeur de régression. Le nombre de neurones dans la couche de sortie correspond au nombre de classes dans le problème de classification ou à la dimension de la sortie dans un problème de régression.

**Exemple : Réseau de neurones simple avec une couche cachée**
```python
from keras.models import Sequential
from keras.layers import Dense

# Créer le modèle
model = Sequential()
model.add(Dense(4, input_dim=3, activation='relu'))  # Couche cachée avec 4 neurones
model.add(Dense(1, activation='sigmoid'))  # Couche de sortie avec 1 neurone

# Afficher le résumé du modèle
model.summary()
```

**Exercice 1 : Création d'un réseau de neurones simple**
1. Créez un réseau de neurones avec deux couches cachées. La première couche cachée doit avoir 8 neurones avec une fonction d'activation ReLU, et la deuxième couche cachée doit avoir 4 neurones avec une fonction d'activation tanh.
2. Utilisez une fonction d'activation softmax pour la couche de sortie avec 3 neurones.

```python
from keras.models import Sequential
from keras.layers import Dense

# Créer le modèle
model = Sequential()
model.add(Dense(8, input_dim=3, activation='relu'))  # Première couche cachée
model.add(Dense(4, activation='tanh'))  # Deuxième couche cachée
model.add(Dense(3, activation='softmax'))  # Couche de sortie

# Afficher le résumé du modèle
model.summary()
```

**Exercice 2 : Expérimentation avec des fonctions d'activation**
1. Modifiez le réseau ci-dessus pour utiliser une fonction d'activation sigmoid dans la première couche cachée et une fonction d'activation ReLU dans la deuxième couche cachée.
2. Observez comment cela affecte les résultats de l'entraînement et la performance du modèle sur un ensemble de données simple (par exemple, MNIST).

**Exemple de réseau de neurones avec des couches multiples et une fonction de perte personnalisée**
```python
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error

# Créer le modèle
model = Sequential()
model.add(Dense(16, input_dim=5, activation='relu'))  # Première couche cachée
model.add(Dense(8, activation='relu'))  # Deuxième couche cachée
model.add(Dense(4, activation='relu'))  # Troisième couche cachée
model.add(Dense(1, activation='linear'))  # Couche de sortie

# Compiler le modèle avec une fonction de perte personnalisée
model.compile(optimizer='adam', loss=mean_squared_error, metrics=['mse'])

# Afficher le résumé du modèle
model.summary()
```

**Exercice 3 : Compréhension de la structure d'un réseau de neurones**
1. Dessinez un diagramme de votre réseau de neurones avec les couches d'entrée, cachées et de sortie.
2. Expliquez pourquoi vous avez choisi certaines fonctions d'activation pour les couches cachées et la couche de sortie.
3. Discutez de l'impact possible de l'ajout de plus de couches cachées ou de neurones dans les couches existantes sur la performance du modèle.

---

#### Chapitre 3 : Introduction aux Réseaux de Neurones Convolutifs (CNN)

##### 3.1 Historique des CNN
[Retour en haut](#plan)

- **Origines biologiques** : Les CNN sont inspirés par le fonctionnement du cortex visuel des mammifères, basé sur les recherches de Hubel et Wiesel dans les années 1960. Ils ont découvert que les neurones dans le cortex visuel des chats réagissent à des motifs visuels spécifiques comme des bords et des lignes de différentes orientations.
  
- **Développement en informatique** : Les CNN ont été popularisés par Yann LeCun et ses collègues en 1998 avec la publication de LeNet-5, un modèle de CNN utilisé pour la classification des chiffres manuscrits dans le dataset MNIST. Depuis, les CNN sont devenus un outil fondamental dans le domaine de la vision par ordinateur.

##### 3.2 Structure d'un CNN
[Retour en haut](#plan)

Un CNN est composé de plusieurs types de couches qui permettent l'extraction automatique de caractéristiques à partir des images d'entrée. Voici une description des différentes couches typiques d'un CNN :

- **Couche d'entrée** : Reçoit l'image sous forme de matrice de pixels. La taille de cette matrice dépend de la résolution de l'image et du nombre de canaux de couleur (par exemple, 28x28x1 pour une image en niveaux de gris de 28x28 pixels).

- **Couches de convolution** : Appliquent des filtres (ou noyaux) sur l'image pour extraire des caractéristiques locales comme des bords, des textures et des motifs. Chaque filtre produit une carte de caractéristiques qui capture une certaine caractéristique de l'image.

- **Couches de ReLU** : Introduisent de la non-linéarité en remplaçant les valeurs négatives par zéro. Cela permet au réseau d'apprendre des relations complexes entre les caractéristiques.

- **Couches de pooling** : Réduisent la dimensionnalité de l'image en sous-échantillonnant les cartes de caractéristiques. Cela aide à réduire le nombre de paramètres et à contrôler le surapprentissage. Le max pooling et le average pooling sont les types de pooling les plus courants.

- **Couches entièrement connectées** : Après les couches de convolution et de pooling, les cartes de caractéristiques sont aplaties et passées à travers des couches de neurones entièrement connectés, similaires à ceux des réseaux de neurones classiques. Ces couches effectuent la classification finale basée sur les caractéristiques extraites.

**Exemple : Construire un simple CNN avec Keras**
```python
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

**Exercice 1 : Créer et entraîner un CNN simple sur le dataset MNIST**
1. Chargez le dataset MNIST.
2. Normalisez les images pour avoir des valeurs entre 0 et 1.
3. Créez un modèle CNN avec une ou deux couches de convolution, des couches de pooling, et des couches entièrement connectées.
4. Compilez et entraînez le modèle.
5. Évaluez la performance du modèle sur les données de test.

```python
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Charger le dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Créer le modèle
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # Couche de convolution
model.add(MaxPooling2D((2, 2)))  # Couche de pooling
model.add(Flatten())  # Couche de mise à plat
model.add(Dense(64, activation='relu'))  # Couche entièrement connectée
model.add(Dense(10, activation='softmax'))  # Couche de sortie

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
```

**Exercice 2 : Expérimenter avec différentes architectures de CNN**
1. Ajoutez une deuxième couche de convolution et de pooling à l'architecture précédente.
2. Changez les tailles des filtres et les nombres de filtres dans chaque couche de convolution.
3. Observez comment ces modifications affectent les performances du modèle.

**Exemple avancé : Utilisation de dropout pour éviter le surapprentissage**
Le dropout est une technique de régularisation où des neurones sélectionnés aléatoirement sont ignorés pendant l'entraînement. Cela aide à prévenir le surapprentissage en rendant le réseau moins sensible aux valeurs spécifiques des poids des neurones.

```python
from keras.layers import Dropout

# Créer le modèle avec dropout
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))  # Ajout de dropout après la couche de pooling
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # Ajout de dropout après la couche dense
model.add(Dense(10, activation='softmax'))

# Compiler et entraîner le modèle comme précédemment
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
```

---

#### Chapitre 4 : Convolution dans les CNN

##### 4.1 Qu'est-ce que la convolution ?
[Retour en haut](#plan)

- **Définition** : La convolution est une opération mathématique qui combine deux fonctions pour produire une troisième fonction. Dans le contexte des réseaux de neurones convolutionnels (CNN), cette opération est utilisée pour extraire des caractéristiques spécifiques d'une image en appliquant des filtres (ou noyaux) sur celle-ci.
- **Application dans les CNN** : Les CNN utilisent des filtres pour détecter divers motifs dans une image, tels que les bords, les textures et d'autres caractéristiques locales. Cela permet au réseau de capturer des informations essentielles à différents niveaux de profondeur.

```python
# Exemple : Appliquer un filtre de convolution à une image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Créer une image simple pour l'exemple
image = np.array([
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

# Définir un filtre (kernel) simple
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Appliquer la convolution
convoluted_image = convolve(image, kernel)
plt.imshow(convoluted_image, cmap='gray')
plt.colorbar()
plt.show()
```

**Exercice 1 : Appliquer différents filtres à une image**
1. Créez une image simple ou utilisez une image de votre choix.
2. Définissez différents filtres, par exemple, des filtres de détection des bords, de flou ou de renforcement des contours.
3. Appliquez chaque filtre à l'image et affichez le résultat.

```python
# Filtre de flou
blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9

# Filtre de détection des bords
edge_kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

# Appliquer les filtres
blurred_image = convolve(image, blur_kernel)
edges_image = convolve(image, edge_kernel)

# Afficher les résultats
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Blurred Image')
plt.imshow(blurred_image, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Edges Image')
plt.imshow(edges_image, cmap='gray')
plt.show()
```

##### 4.2 Filtre (Kernel)
[Retour en haut](#plan)

- **Définition** : Un filtre, ou kernel, est une petite matrice utilisée pour balayer l'image et effectuer des opérations de convolution. Les éléments du filtre sont des poids qui sont ajustés pendant l'entraînement pour extraire les caractéristiques les plus utiles pour la tâche de classification.
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

**Exercice 2 : Explorer l'effet de différents kernels sur une image**
1. Essayez différents types de kernels, comme les kernels de sharpening, de flou, ou de détection des contours.
2. Appliquez ces kernels à une image et observez les différences dans les résultats.

```python
# Filtre de sharpening
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# Filtre de détection des contours
edge_kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

# Appliquer les filtres
sharpened_image = convolve(image, sharpen_kernel)
edges_image = convolve(image, edge_kernel)

# Afficher les résultats
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Sharpened Image')
plt.imshow(sharpened_image, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Edges Image')
plt.imshow(edges_image, cmap='gray')
plt.show()
```

##### 4.3 Stride
[Retour en haut](#plan)

- **Définition** : Le stride est le nombre de pixels par lesquels le filtre se déplace sur l'image. Un stride plus grand réduit la dimension de l'image de sortie, car le filtre est appliqué moins souvent.
- **Impact sur la dimension de l'image convoluée** : Un stride plus grand réduit la dimension de l'image de sortie.

```python
# Exemple : Appliquer la convolution avec différents strides
def apply_convolution(image, kernel, stride):
    output_shape = (
        (image.shape[0] - kernel.shape[0]) // stride + 1,
        (image.shape[1] - kernel.shape[1]) // stride + 1
    )
    output = np.zeros(output_shape)
    for i in range(0, output_shape[0]):
        for j in range(0, output_shape[1]):
            output[i, j] = np.sum(image[i*stride:i*stride+kernel.shape[0], j*stride:j*stride+kernel.shape[1]] * kernel)
    return output

convoluted_image_stride1 = apply_convolution(image, kernel, 1)
convoluted_image_stride2 = apply_convolution(image, kernel, 2)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Stride 1')
plt.imshow(convoluted_image_stride1, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Stride 2')
plt.imshow(convoluted_image_stride2, cmap='gray')
plt.show()
```

**Exercice 3 : Expérimenter avec différents strides**
1. Changez le stride utilisé dans l'application du filtre.
2. Observez comment la taille de l'image convoluée change en fonction du stride utilisé.
3. Comparez les résultats visuellement.

##### 4.4 Padding
[Retour en haut](#plan)

- **Définition** : Le padding consiste à ajouter des bordures de pixels autour de l'image pour conserver sa dimension après la convolution. Il existe principalement deux types de padding :
  - **Valid padding** : Pas de padding, ce qui réduit la taille de l'image après la convolution.
  - **Same padding** : Padding ajouté pour que la taille de l'image reste la même après la convolution.

```python
# Exemple : Appliquer la convolution avec et sans padding
def apply_padding(image, padding_size):
    padded_image = np.pad(image, pad_width=padding_size, mode='constant', constant_values=0)
    return padded_image

padded_image = apply_padding(image, 1)
convoluted_image_padded = convolve(padded_image, kernel)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Sans Padding')
plt.imshow(convoluted_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Avec Padding')
plt.imshow(convoluted_image_padded, cmap='gray')
plt.show()
```

**Exercice 4 : Comparer l'effet du padding sur la convolution**
1. Appliquez la convolution à une image avec et sans padding.
2. Observez les différences dans les dimensions des images résultantes.
3. Discutez de l'importance du padding dans le maintien des caractéristiques spatiales.


---


#### Chapitre 5 : Couches de ReLU dans les CNN

##### 5.1 Fonction d'activation ReLU
[Retour en haut](#plan)

- **Définition** : ReLU (Rectified Linear Unit) est une fonction d'activation utilisée dans les réseaux de neurones, qui remplace toutes les valeurs négatives par zéro et laisse les valeurs positives inchangées. La formule de la fonction ReLU est la suivante :

$$
\text{ReLU}(x) = \max(0, x)
$$


- **Importance** : La fonction ReLU introduit de la non-linéarité dans le modèle, ce qui est essentiel pour permettre au réseau de neurones de modéliser des relations complexes. De plus, elle aide à résoudre le problème du gradient vanishing en fournissant des gradients plus importants pour les valeurs positives.

```python
# Exemple : Appliquer la fonction ReLU à une image convoluée
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# Image convoluée d'exemple
convoluted_image = np.array([
    [-1, 2, -3],
    [4, -5, 6],
    [-7, 8, -9]
])

relu_image = relu(convoluted_image)

plt.imshow(relu_image, cmap='gray')
plt.colorbar()
plt.show()
```

**Exercice 1 : Expérimenter avec ReLU**
1. Créez une matrice 3x3 avec des valeurs négatives et positives.
2. Appliquez la fonction ReLU à cette matrice.
3. Affichez la matrice résultante et observez comment les valeurs négatives ont été remplacées par zéro.

```python
# Matrice d'exemple
matrix = np.array([
    [-2, -1, 0],
    [1, 2, 3],
    [-3, -2, -1]
])

# Appliquer ReLU
relu_matrix = relu(matrix)

print("Matrice originale :")
print(matrix)
print("Matrice après application de ReLU :")
print(relu_matrix)
```

**Exercice 2 : Comparer ReLU avec d'autres fonctions d'activation**
1. Implémentez les fonctions d'activation Sigmoid et Tanh.
2. Appliquez Sigmoid, Tanh et ReLU à la même matrice d'exemple.
3. Comparez les résultats et discutez de l'impact de chaque fonction d'activation sur les valeurs de la matrice.

```python
# Fonction Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Fonction Tanh
def tanh(x):
    return np.tanh(x)

# Appliquer Sigmoid et Tanh
sigmoid_matrix = sigmoid(matrix)
tanh_matrix = tanh(matrix)

print("Matrice après application de Sigmoid :")
print(sigmoid_matrix)
print("Matrice après application de Tanh :")
print(tanh_matrix)
```

**Exercice 3 : Visualisation des fonctions d'activation**
1. Tracez les courbes des fonctions ReLU, Sigmoid et Tanh sur une plage de valeurs allant de -10 à 10.
2. Comparez visuellement les différentes fonctions d'activation et leur comportement.

```python
# Tracer les courbes des fonctions d'activation
x = np.linspace(-10, 10, 100)

plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.plot(x, relu(x))
plt.title('ReLU')

plt.subplot(1, 3, 2)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')

plt.subplot(1, 3, 3)
plt.plot(x, tanh(x))
plt.title('Tanh')

plt.show()
```


### Comprendre ReLU dans un contexte CNN

Lorsqu'une image passe par une couche de convolution, elle est transformée en une série de cartes de caractéristiques. L'application de la fonction d'activation ReLU permet d'introduire de la non-linéarité après chaque couche de convolution. Cela permet au réseau de neurones de mieux capturer les motifs complexes et d'améliorer la performance globale du modèle.

```python
# Exemple : Pipeline simple de CNN avec ReLU dans Keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

# Ajouter une couche de convolution
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Ajouter une couche de pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Ajouter une couche de mise à plat
model.add(Flatten())

# Ajouter une couche entièrement connectée
model.add(Dense(128, activation='relu'))

# Ajouter une couche de sortie
model.add(Dense(10, activation='softmax'))

# Afficher le résumé du modèle
model.summary()
```

**Exercice 4 : Construire et entraîner un simple CNN**
1. Utilisez le dataset MNIST pour construire un simple CNN en utilisant Keras.
2. Appliquez des couches de convolution suivies de la fonction d'activation ReLU.
3. Entraînez le modèle et évaluez sa performance sur le jeu de test.

```python
from keras.datasets import mnist
from keras.utils import to_categorical

# Charger les données MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Précision du modèle sur le jeu de test : {accuracy * 100:.2f}%")
```

---

#### Chapitre 6 : Couches de Pooling dans les CNN

##### 6.1 Qu'est-ce que le pooling ?
[Retour en haut](#plan)

- **Définition** : Le pooling est une opération de réduction de la dimensionnalité de l'image tout en conservant les caractéristiques importantes.
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
[Retour en haut](#plan)

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

**Exercice 1 : Expérimenter avec le Max Pooling**
1. Créez une matrice 4x4 avec des valeurs variées.
2. Appliquez le Max Pooling avec une fenêtre de 2x2 et un stride de 2.
3. Affichez la matrice résultante.

```python
# Matrice d'exemple
matrix = np.array([
    [1, 3, 2, 4],
    [5, 6, 7, 8],
    [9, 2, 0, 1],
    [4, 3, 5, 6]
])

# Appliquer Max Pooling
max_pooled_matrix = max_pooling(matrix, 2, 2)

print("Matrice originale :")
print(matrix)
print("Matrice après application de Max

 Pooling :")
print(max_pooled_matrix)
```

**Exercice 2 : Comparer Max Pooling et Average Pooling**
1. Appliquez à la même matrice d'exemple le Average Pooling avec une fenêtre de 2x2 et un stride de 2.
2. Comparez les résultats des deux techniques de pooling.

```python
# Appliquer Average Pooling
average_pooled_matrix = average_pooling(matrix, 2, 2)

print("Matrice après application de Max Pooling :")
print(max_pooled_matrix)
print("Matrice après application de Average Pooling :")
print(average_pooled_matrix)
```

### Comprendre le Pooling dans un contexte CNN

Le pooling est utilisé pour réduire la dimensionnalité des cartes de caractéristiques produites par les couches de convolution. Cela permet de diminuer le nombre de paramètres dans les couches suivantes du réseau, ce qui rend le modèle plus efficace et réduit le risque de surapprentissage.

```python
# Exemple : Pipeline simple de CNN avec Max Pooling dans Keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

# Ajouter une couche de convolution
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Ajouter une couche de max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Ajouter une couche de mise à plat
model.add(Flatten())

# Ajouter une couche entièrement connectée
model.add(Dense(128, activation='relu'))

# Ajouter une couche de sortie
model.add(Dense(10, activation='softmax'))

# Afficher le résumé du modèle
model.summary()
```

**Exercice 3 : Construire et entraîner un CNN avec Max Pooling**
1. Utilisez le dataset MNIST pour construire un CNN en utilisant Keras.
2. Appliquez des couches de convolution suivies de Max Pooling.
3. Entraînez le modèle et évaluez sa performance sur le jeu de test.

```python
from keras.datasets import mnist
from keras.utils import to_categorical

# Charger les données MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Précision du modèle sur le jeu de test : {accuracy * 100:.2f}%")
```

---

#### Chapitre 7 : Couches Entièrement Connectées et Classification

##### 7.1 Couches entièrement connectées
[Retour en haut](#plan)

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
[Retour en haut](#plan)

- **Flattening** : Transformation des matrices 2D en vecteurs 1D pour les couches entièrement connectées.
- **Softmax** : Fonction d'activation utilisée en sortie pour produire des probabilités de classification.

```python
# Exemple : Utiliser la fonction softmax pour la classification
from keras.layers import Softmax

# Ajout de la couche Softmax pour la classification
model.add(Dense(10))
model.add(Softmax())
```

**Exercice 1 : Comprendre le Flattening**
1. Créez une matrice 2D de 3x3.
2. Transformez cette matrice en un vecteur 1D en utilisant l'opération de flattening.
3. Affichez la matrice originale et le vecteur 1D.

```python
# Matrice d'exemple
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Appliquer Flattening
flattened_matrix = matrix.flatten()

print("Matrice originale :")
print(matrix)
print("Vecteur après Flattening :")
print(flattened_matrix)
```

**Exercice 2 : Construire un réseau de neurones entièrement connecté**
1. Créez un modèle de réseau de neurones simple avec une couche de flattening et deux couches entièrement connectées.
2. Utilisez le dataset MNIST pour entraîner le modèle.
3. Évaluez la performance du modèle sur le jeu de test.

```python
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical

# Charger les données MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Créer le modèle
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Précision du modèle sur le jeu de test : {accuracy * 100:.2f}%")
```

**Exercice 3 : Visualiser les poids de la couche entièrement connectée**
1. Après l'entraînement du modèle, récupérez les poids de la première couche entièrement connectée.
2. Visualisez ces poids sous forme de matrice de pixels.
3. Discutez de ce que ces poids représentent.

```python
# Récupérer les poids de la première couche entièrement connectée
weights, biases = model.layers[1].get_weights()

# Visualiser les poids
plt.imshow(weights, cmap='viridis')
plt.colorbar()
plt.title('Poids de la première couche entièrement connectée')
plt.show()
```


---

#### Chapitre 8 : Entraînement des CNN

##### 8.1 Processus d'entraînement
[Retour en haut](#plan)
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
[Retour en haut](#plan)
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
[Retour en haut](#plan)

Les réseaux de neurones convolutifs (CNN) ont révolutionné de nombreux domaines grâce à leur capacité à traiter et analyser les données visuelles. Voici quelques-uns des principaux domaines d'application :

- **Reconnaissance d'images** : Identification et classification d'objets dans des images, utilisée dans des applications telles que la sécurité, l'agriculture, la santé, et les véhicules autonomes.
- **Analyse de vidéos** : Détection et suivi d'objets en mouvement, essentielle pour la surveillance, la reconnaissance faciale, et les sports.
- **Systèmes de recommandation** : Suggestions personnalisées basées sur les préférences visuelles, couramment utilisées dans les plateformes de commerce électronique et de streaming.
- **Traitement du langage naturel** : Applications comme la reconnaissance d'écriture manuscrite et la traduction automatique, où les images de textes sont converties en données numériques.

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

**Exercice 1 : Utiliser un modèle pré-entraîné pour classifier des images**
1. Choisissez une image de votre choix et utilisez le modèle VGG16 pour prédire son contenu.
2. Affichez l'image et les prédictions faites par le modèle.

```python
# Charger une nouvelle image et la préparer pour le modèle
img_path = 'your_image.jpg'  # Remplacez par le chemin de votre image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Faire des prédictions
preds = vgg_model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

# Afficher l'image et les prédictions
plt.imshow(image.load_img(img_path, target_size=(224, 224)))
plt.title("Predictions: " + str(decode_predictions(preds, top=3)[0]))
plt.show()
```

**Exercice 2 : Implémenter une application simple de reconnaissance d'images**
1. Créez une interface simple qui permet de charger une image et d'afficher les prédictions faites par un modèle CNN pré-entraîné.
2. Utilisez une bibliothèque d'interface graphique en Python comme Tkinter pour créer cette interface.

```python
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

def load_image():
    img_path = filedialog.askopenfilename()
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = vgg_model.predict(x)
    result = decode_predictions(preds, top=3)[0]
    print('Predicted:', result)
    img = ImageTk.PhotoImage(Image.open(img_path))
    panel = tk.Label(root, image=img)
    panel.image = img
    panel.pack()
    label.config(text=str(result))

root = tk.Tk()
root.title("Image Recognition with CNN")

btn = tk.Button(root, text="Load Image", command=load_image)
btn.pack()

label = tk.Label(root, text="")
label.pack()

root.mainloop()
```

**Exercice 3 : Comparer les performances de différents modèles pré-entraînés**
1. Testez plusieurs modèles pré-entraînés disponibles dans Keras, comme ResNet50, InceptionV3, et MobileNet.
2. Comparez les performances de ces modèles en termes de précision et de temps de prédiction pour différentes images.

```python
from keras.applications import ResNet50, InceptionV3, MobileNet

# Charger les modèles pré-entraînés
resnet_model = ResNet50(weights='imagenet')
inception_model = InceptionV3(weights='imagenet')
mobilenet_model = MobileNet(weights='imagenet')

models = {
    "VGG16": vgg_model,
    "ResNet50": resnet_model,
    "InceptionV3": inception_model,
    "MobileNet": mobilenet_model
}

for name, model in models.items():
    start_time = time.time()
    preds = model.predict(x)
    end_time = time.time()
    print(f"Model: {name}")
    print('Predicted:', decode_predictions(preds, top=3)[0])
    print(f"Time taken: {end_time - start_time:.2f} seconds\n")
```

---

#### Chapitre 10 : Avantages et Inconvénients des CNN

##### 10.1 Avantages
[Retour en haut](#plan)

- **Efficacité** : Les CNN offrent une excellente performance pour les tâches de vision par ordinateur en raison de leur capacité à apprendre automatiquement des caractéristiques hiérarchiques à partir des données brutes.
- **Flexibilité** : Ils peuvent être appliqués à diverses tâches de traitement d'images et de vidéos, y compris la classification, la segmentation, et la détection d'objets.
- **Automatisation** : Les CNN éliminent la nécessité de l'ingénierie manuelle des caractéristiques, rendant le processus d'apprentissage plus automatique et efficace.

##### 10.2 Inconvénients
[Retour en haut](#plan)

- **Complexité** : Les CNN nécessitent de grandes quantités de données pour l'entraînement, ce qui peut être un défi pour certains domaines où les données sont limitées.
- **Ressources** : Ils requièrent des ressources matérielles importantes, notamment des GPU puissants, pour entraîner des modèles en temps raisonnable.
- **Interprétabilité** : Les modèles de CNN sont souvent considérés comme des "boîtes noires" en raison de leur complexité, ce qui peut rendre difficile l'interprétation des décisions du modèle.

**Exercice 1 : Analyser les avantages et les inconvénients des CNN**
1. Réfléchissez aux situations dans lesquelles les avantages des CNN surpassent leurs inconvénients, et vice versa.
2. Écrivez un court essai discutant de l'impact des CNN dans différents domaines d'application, en mettant l'accent sur les avantages et les défis.

```python
# Exemple de réflexion
avantages = [
    "Les CNN offrent une excellente performance pour les tâches de vision par ordinateur.",
    "Ils peuvent apprendre automatiquement des caractéristiques hiérarchiques.",
    "Ils sont flexibles et peuvent être appliqués à diverses tâches de traitement d'images."
]

inconvenients = [
    "Les CNN nécessitent de grandes quantités de données pour l'entraînement.",
    "Ils requièrent des ressources matérielles importantes.",
    "Les modèles de CNN sont souvent difficiles à interpréter."
]

print("Avantages des CNN:")
for avantage in avantages:
    print("-", avantage)

print("\nInconvénients des CNN:")
for inconvenient in inconvenients:
    print("-", inconvenient)
```

**Exercice 2 : Comparer les temps d'entraînement entre un modèle CNN et un modèle traditionnel**
1. Entraînez un modèle de forêt aléatoire (RandomForest) et un modèle CNN sur le dataset MNIST.
2. Comparez les temps d'entraînement et les performances des deux modèles.

```python
import time
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import mnist
from keras.utils import to_categorical

# Charger le dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_flat = X_train.reshape(-1, 28*28)
X_test_flat = X_test.reshape(-1, 28*28)

# Entraîner un modèle de forêt aléatoire
start_time = time.time()
rf_model = RandomForestClassifier()
rf_model.fit(X_train_flat, y_train)
rf_training_time = time.time() - start_time

# Entraîner un modèle CNN
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',

 metrics=['accuracy'])

start_time = time.time()
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
cnn_training_time = time.time() - start_time

print(f"Temps d'entraînement RF: {rf_training_time:.2f} secondes")
print(f"Temps d'entraînement CNN: {cnn_training_time:.2f} secondes")
```

**Exercice 3 : Explorer les contraintes de ressources des CNN**
1. Utilisez un environnement avec des ressources limitées (par exemple, un ordinateur portable sans GPU) pour entraîner un modèle CNN simple.
2. Notez les limitations rencontrées, telles que le temps d'entraînement et la consommation de mémoire.
3. Discutez des solutions possibles pour surmonter ces limitations, telles que l'utilisation de services de cloud computing ou de modèles pré-entraînés.

```python
# Réflexion sur les contraintes de ressources
limitations = [
    "Temps d'entraînement plus long en l'absence de GPU.",
    "Consommation de mémoire élevée pour les grands modèles.",
    "Difficulté à entraîner des modèles complexes sur des machines avec des ressources limitées."
]

solutions = [
    "Utiliser des services de cloud computing pour l'entraînement de modèles.",
    "Utiliser des modèles pré-entraînés pour réduire les besoins en données et en ressources.",
    "Optimiser les modèles en réduisant le nombre de paramètres et en utilisant des techniques de compression."
]

print("Limitations rencontrées lors de l'entraînement des CNN sur des ressources limitées:")
for limitation in limitations:
    print("-", limitation)

print("\nSolutions possibles pour surmonter ces limitations:")
for solution in solutions:
    print("-", solution)
```



---



### Conclusion de la première partie
[Retour en haut](#plan)

- Les réseaux de neurones convolutifs sont des outils puissants pour le traitement d'images et de vidéos, offrant des performances remarquables grâce à leur capacité à extraire automatiquement des caractéristiques hiérarchiques. Leur utilisation s'étend à de nombreux domaines, rendant les CNN essentiels pour les applications modernes de deep learning. Toutefois, il est important de reconnaître leurs défis et de développer des stratégies pour les surmonter, afin de tirer pleinement parti de leur potentiel.

- La première partie de ce cours a couvert les bases des réseaux de neurones convolutifs (CNN), leur structure, les composants principaux, et les processus d'entraînement. Nous avons également exploré diverses applications des CNN et comparé leurs avantages et inconvénients. Maintenant, nous allons introduire Keras, une bibliothèque puissante et facile à utiliser pour construire des modèles de deep learning.

---


### Chapitre 11 : Introduction à Keras

#### 11.1 Qu'est-ce que Keras ?
[Retour en haut](#plan)

- Keras est une API de réseaux de neurones de haut niveau, écrite en Python et capable de s'exécuter sur des moteurs de calcul comme TensorFlow, CNTK, ou Theano.
- Elle permet de créer et d’entraîner des modèles de deep learning de manière simple et efficace.
- Keras est aussi une bibliothèque extrêmement puissante qui simplifie le processus de création et d'entraînement de modèles de deep learning.
- Avec des fonctionnalités comme la modularité, l'extensibilité, et une interface conviviale, Keras est un excellent outil pour les débutants comme pour les experts en deep learning.
- Les exercices pratiques ci-dessous à faire à la maison vous aideront à vous familiariser avec les concepts clés de Keras et à expérimenter avec différents aspects de la création de modèles.


#### 11.2 Pourquoi utiliser Keras ?
[Retour en haut](#plan)

1. **Facilité d’utilisation** : Keras est conçu pour être convivial et simple à utiliser, ce qui permet de se concentrer sur les aspects importants de la création de modèles de deep learning sans se perdre dans des détails techniques complexes.
2. **Modularité** : Keras est basé sur un modèle modulaire où chaque composant d'un modèle de deep learning est une entité distincte qui peut être combinée avec d'autres.
3. **Extensibilité** : Keras est hautement extensible et permet d'ajouter facilement de nouveaux modules ou de personnaliser ceux existants.
4. **Production** : Keras offre une compatibilité avec plusieurs backend, ce qui en fait un choix flexible pour différents environnements de production.

#### 11.3 Environnement Keras

##### 11.3.1 Architecture de Keras
[Retour en haut](#plan)

- **Votre programme** : C’est le script Python que vous écrivez.
- **Keras (API NN)** : L’API qui vous permet de définir et d’entraîner vos modèles.
- **Backend (TensorFlow, Theano, CNTK)** : Le moteur de calcul qui effectue les opérations de bas niveau nécessaires pour entraîner les modèles.

##### 11.3.2 Exemple de modèle avec Keras
[Retour en haut](#plan)

Voici un exemple de création et d'entraînement d'un modèle simple avec Keras. 

**Étape 1 : Importer les bibliothèques nécessaires**

```python
from keras.models import Sequential
from keras.layers import Dense
```

**Étape 2 : Initialiser le modèle**

```python
model = Sequential()
```

**Étape 3 : Ajouter des couches au modèle**

```python
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

**Étape 4 : Compiler le modèle**

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

**Étape 5 : Entraîner le modèle**

```python
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

### Exercices à faire à la maison

**Exercice 1 : Créer un modèle simple**

1. Créez un modèle Keras similaire à l'exemple ci-dessus, mais avec une couche supplémentaire de 32 neurones après la première couche cachée.
2. Changez la fonction d'activation de la couche de sortie en `sigmoid`.
3. Compilez et entraînez le modèle avec des données fictives.

```python
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Données fictives
x_train = np.random.random((1000, 100))
y_train = np.random.randint(2, size=(1000, 10))

# Créer le modèle
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=10, activation='sigmoid'))

# Compiler le modèle
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

**Exercice 2 : Expérimenter avec différents optimizers**

1. Modifiez le modèle créé dans l'Exercice 1 pour utiliser l'optimizer `adam` au lieu de `sgd`.
2. Comparez les performances en termes de précision et de temps d'entraînement.

```python
# Compiler le modèle avec optimizer 'adam'
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

**Exercice 3 : Sauvegarder et charger un modèle**

1. Sauvegardez le modèle entraîné dans l'Exercice 1 sur le disque.
2. Chargez le modèle sauvegardé et utilisez-le pour faire des prédictions sur de nouvelles données fictives.

```python
# Sauvegarder le modèle
model.save('my_model.h5')

# Charger le modèle
from keras.models import load_model
model = load_model('my_model.h5')

# Nouvelles données fictives
x_new = np.random.random((10, 100))

# Faire des prédictions
predictions = model.predict(x_new)
print(predictions)
```

**Exercice 4 : Visualiser les courbes d'entraînement**

1. Ajoutez des callbacks pour visualiser les courbes de précision et de perte pendant l'entraînement du modèle.
2. Utilisez TensorBoard pour afficher les graphiques.

```python
from keras.callbacks import TensorBoard

# Initialiser TensorBoard
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

# Entraîner le modèle avec TensorBoard
model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=[tensorboard])
```


---

### Chapitre 12 : Créer Votre Premier Réseau de Neurones avec Keras

#### 12.1 Introduction
[Retour en haut](#plan)

Keras est une bibliothèque de réseaux de neurones haut niveau qui tourne sur TensorFlow. Il permet de construire et d'entraîner des modèles de Deep Learning facilement grâce à une interface simple et conviviale. Ce chapitre vous guidera pas à pas dans la création de votre premier réseau de neurones avec Keras.

#### 12.2 Importation de Keras
[Retour en haut](#plan)

Keras est intégré dans TensorFlow, ce qui signifie que vous pouvez l'utiliser directement en l'important depuis TensorFlow :

```python
import tensorflow as tf
from tensorflow import keras
```

#### 12.3 Installation de Keras
[Retour en haut](#plan)

- **Keras intégré à TensorFlow** : Aucune installation séparée n'est nécessaire si vous utilisez TensorFlow.
- **Keras autonome** : Il est également possible d'installer Keras comme une bibliothèque autonome et de l'utiliser avec différents backends tels que TensorFlow, Theano, ou CNTK. 

Pour installer Keras de manière autonome :
```bash
pip install keras
```
Pour changer de backend, vous devez configurer le fichier `.keras/keras.json` dans votre répertoire utilisateur.

#### 12.4 Création d'un Modèle Simple
[Retour en haut](#plan)

Pour commencer, nous allons créer un modèle de réseau de neurones simple avec une couche d'entrée, une couche cachée et une couche de sortie.

##### 12.4.1 Importer les Bibliothèques Nécessaires
[Retour en haut](#plan)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

##### 12.4.2 Préparer les Données
[Retour en haut](#plan)

Nous allons utiliser le dataset MNIST pour cette démonstration. Ce dataset contient des images de chiffres écrits à la main.

```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

##### 12.4.3 Construire le Modèle
[Retour en haut](#plan)

Nous allons créer un modèle séquentiel avec une couche d'entrée, une couche cachée dense et une couche de sortie.

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
```

##### 12.4.4 Compiler le Modèle
[Retour en haut](#plan)

Ensuite, nous compilons le modèle avec une fonction de perte, un optimiseur et des métriques.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

##### 12.4.5 Entraîner le Modèle
[Retour en haut](#plan)

Nous entraînons le modèle sur les données d'entraînement.

```python
model.fit(x_train, y_train, epochs=5)
```

##### 12.4.6 Évaluer le Modèle
[Retour en haut](#plan)

Enfin, nous évaluons le modèle sur les données de test.

```python
model.evaluate(x_test, y_test)
```

#### 12.5 Changer de Backend
[Retour en haut](#plan)

Keras permet de changer de backend facilement en modifiant le fichier de configuration `keras.json`.

```json
{
    "backend": "tensorflow",
    "floatx": "float32",
    "epsilon": 1e-07,
    "image_data_format": "channels_last"
}
```

### Exercices à faire à la maison

Pour bien assimiler les concepts et techniques décrits dans ce chapitre, voici quelques exercices pratiques. Ces exercices vous permettront de renforcer votre compréhension de la création et de l'entraînement des modèles de deep learning avec Keras.

#### Exercice 1 : Créer un modèle de réseau de neurones

1. **Objectif** : Créer un modèle avec une structure différente pour comprendre l'impact des différentes architectures de réseaux.
2. **Instructions** :
    - Créez un modèle avec deux couches cachées de 64 et 32 neurones respectivement.
    - Utilisez la fonction d'activation `tanh` pour la première couche cachée et `relu` pour la seconde.
    - Changez la fonction d'activation de la couche de sortie en `softmax`.
    - Compilez et entraînez le modèle sur le dataset MNIST.

```python
from keras.models import Sequential
from keras.layers import Dense

# Créer le modèle
model = Sequential()
model.add(Dense(64, activation='tanh', input_shape=(784,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=5)
```

#### Exercice 2 : Comparer les optimizers

1. **Objectif** : Comprendre l'impact de différents optimizers sur la performance du modèle.
2. **Instructions** :
    - Utilisez le même modèle que dans l'Exercice 1.
    - Compilez et entraînez le modèle en utilisant différents optimizers (`adam`, `sgd`, `rmsprop`).
    - Comparez les performances en termes de précision et de temps d'entraînement.

```python
# Compilez le modèle avec différents optimizers et comparez les performances

# Optimizer 'adam'
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_adam = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Optimizer 'sgd'
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_sgd = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Optimizer 'rmsprop'
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_rmsprop = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
```

#### Exercice 3 : Ajouter des callbacks

1. **Objectif** : Utiliser des callbacks pour surveiller et ajuster l'entraînement du modèle.
2. **Instructions** :
    - Ajoutez des callbacks pour visualiser les courbes de précision et de perte pendant l'entraînement.
    - Utilisez `EarlyStopping` pour arrêter l'entraînement si la performance sur le jeu de validation ne s'améliore pas pendant un certain nombre d'époques.

```python
from keras.callbacks import TensorBoard, EarlyStopping

# Initialiser TensorBoard
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

# Initialiser EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Entraîner le modèle avec les callbacks
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), callbacks=[tensorboard, early_stopping])
```

#### Exercice 4 : Sauvegarder et charger un modèle

1. **Objectif** : Apprendre à sauvegarder et charger un modèle pour réutilisation ultérieure.
2. **Instructions** :
    - Sauvegardez le modèle entraîné dans l'Exercice 1 sur le disque.
    - Chargez le modèle sauvegardé et utilisez-le pour faire des prédictions sur de nouvelles données.

```python
# Sauvegarder le modèle
model.save('my_model.h5')

# Charger le modèle
from keras.models import load_model
model = load_model('my_model.h5')

# Nouvelles données fictives
x_new = np.random.random((10, 784))

# Faire des prédictions
predictions = model.predict(x_new)
print(predictions)
```

### Conclusion
[Retour en haut](#plan)

Keras simplifie la création et l'entraînement des modèles de Deep Learning. Son intégration avec TensorFlow permet une utilisation flexible et puissante, adaptée aux débutants comme aux experts. Les exercices pratiques vous aideront à renforcer vos compétences et à mieux comprendre les concepts de base de la création de réseaux de neurones avec Keras.

---

### Chapitre 13 : Construire des Modèles avec Keras

#### 13.1 Modèle Sequential
[Retour en haut](#plan)

Le modèle Sequential est l'une des approches les plus simples et directes pour construire des réseaux de neurones avec Keras. Voici ses principales caractéristiques et comment l'utiliser efficacement.

**Caractéristiques clés** :
- **Facile à comprendre** : Le modèle Sequential est simple, ce qui en fait un excellent point de départ pour les débutants.
- **Séquence de couches** : Les couches sont ajoutées une par une, formant une pile linéaire.
- **Interconnexion automatique des couches** : Chaque couche est automatiquement connectée à la suivante.
- **Rapide et simple** : Idéal pour des architectures de réseaux de neurones simples.

**Exemple de création d'un modèle Sequential :**
```python
from keras.models import Sequential
from keras.layers import Dense

# Initialiser le modèle
model = Sequential()

# Ajouter des couches
model.add(Dense(64, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Afficher le résumé du modèle
model.summary()
```

**Exercice 1 : Créer un modèle Sequential**
1. Créez un modèle Sequential avec trois couches cachées de tailles 128, 64 et 32 neurones respectivement.
2. Utilisez la fonction d'activation `relu` pour toutes les couches cachées.
3. Utilisez la fonction d'activation `softmax` pour la couche de sortie.
4. Compilez le modèle avec l'optimiseur `adam` et la fonction de perte `sparse_categorical_crossentropy`.

```python
# Créer le modèle
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=100))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Afficher le résumé du modèle
model.summary()
```

---

#### 13.2 Classe Model avec API Fonctionnelle
[Retour en haut](#plan)

L'API fonctionnelle de Keras offre plus de flexibilité et de puissance pour créer des architectures de réseaux de neurones complexes. Elle permet de construire des modèles avec des connexions arbitraires, ce qui est particulièrement utile pour les réseaux qui ne sont pas purement séquentiels.

**Caractéristiques clés** :
- **Réseaux de neurones complexes** : Convient aux architectures de réseaux plus complexes.
- **Couches comme unités fonctionnelles** : Permet de traiter les couches comme des fonctions pouvant être connectées dans une structure de graphe.
- **Connexions de couches définies par l'utilisateur** : Donne un contrôle total sur les connexions entre les couches.
- **Détaillé et puissant** : Plus flexible et puissant que le modèle Sequential.

**Exemple de création d'un modèle avec l'API fonctionnelle :**
```python
from keras.models import Model
from keras.layers import Input, Dense

# Définir l'entrée
inputs = Input(shape=(100,))

# Ajouter des couches
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

# Créer le modèle
model = Model(inputs=inputs, outputs=outputs)

# Compiler le modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Afficher le résumé du modèle
model.summary()
```

**Exercice 2 : Créer un modèle avec l'API fonctionnelle**
1. Créez un modèle avec deux entrées : une pour les caractéristiques numériques et une pour les images.
2. La première entrée doit passer par deux couches denses avec `relu` et la seconde par une couche convolutionnelle suivie d'une couche de pooling.
3. Combinez les sorties des deux branches et ajoutez une couche dense finale avec `softmax`.

```python
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate

# Première branche (caractéristiques numériques)
input_numeric = Input(shape=(10,))
x1 = Dense(64, activation='relu')(input_numeric)
x1 = Dense(32, activation='relu')(x1)

# Deuxième branche (images)
input_image = Input(shape=(28, 28, 1))
x2 = Conv2D(32, (3, 3), activation='relu')(input_image)
x2 = MaxPooling2D((2, 2))(x2)
x2 = Flatten()(x2)
x2 = Dense(64, activation='relu')(x2)

# Combiner les deux branches
combined = concatenate([x1, x2])

# Couche de sortie
outputs = Dense(10, activation='softmax')(combined)

# Créer le modèle
model = Model(inputs=[input_numeric, input_image], outputs=outputs)

# Compiler le modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Afficher le résumé du modèle
model.summary()
```

### Conclusion
[Retour en haut](#plan)

- Keras offre deux principales façons de créer des modèles de réseaux de neurones : le modèle Sequential et l'API fonctionnelle. Le modèle Sequential est idéal pour les architectures simples et linéaires, tandis que l'API fonctionnelle offre la flexibilité nécessaire pour les réseaux complexes. En comprenant les caractéristiques et les avantages de chaque approche, vous pouvez choisir celle qui convient le mieux à vos besoins et exploiter pleinement la puissance de Keras pour créer des modèles de deep learning performants et efficaces.

---

### Chapitre 14 : Introduction aux Réseaux de Neurones Convolutionnels (CNN) avec Keras

#### 14.1 Vue d'ensemble des Réseaux de Neurones Convolutionnels
[Retour en haut](#plan)

Les réseaux de neurones convolutionnels (CNN) sont une classe de réseaux de neurones profonds particulièrement efficaces pour analyser les images visuelles. Ils sont largement utilisés dans la reconnaissance d'images et de vidéos, les systèmes de recommandation et le traitement du langage naturel.

#### 14.2 Comprendre les Couches dans Keras
[Retour en haut](#plan)

Keras est une bibliothèque de réseaux de neurones open-source puissante et facile à utiliser, écrite en Python. Elle peut fonctionner sur TensorFlow, Microsoft Cognitive Toolkit, Theano ou PlaidML. Elle permet un prototypage facile et rapide, prend en charge à la fois les réseaux convolutionnels et récurrents, et fonctionne sans problème sur CPU et GPU.

#### 14.3 Les Couches Communes dans Keras
[Retour en haut](#plan)

- **Couches Denses** : Couches entièrement connectées où chaque neurone est connecté à tous les neurones de la couche précédente.
- **Couches de Dropout** : Technique de régularisation où des neurones sélectionnés aléatoirement sont ignorés pendant l'entraînement pour éviter le surapprentissage.
- **Couches de Reshape** : Modifient la forme de l'entrée sans en altérer les données.
- **Couches de Flatten** : Aplatissent l'entrée, la convertissant en un tableau à 1 dimension.
- **Couches de Permute** : Permutent les dimensions de l'entrée selon un schéma donné.
- **Couches de RepeatVector** : Répètent l'entrée un certain nombre de fois.

**Exercice 1 : Exploration des Couches**
1. Créez un modèle Sequential en utilisant les couches mentionnées ci-dessus.
2. Ajoutez une couche Dense, une couche Dropout, une couche Flatten, une couche Reshape, une couche Permute et une couche RepeatVector.
3. Compilez le modèle et affichez son résumé.

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Permute, RepeatVector

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=100))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Reshape((8, 8, 1)))
model.add(Permute((2, 1, 3)))
model.add(RepeatVector(10))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

#### 14.4 Les Couches Convolutionnelles
[Retour en haut](#plan)

Les couches convolutionnelles sont les blocs de construction fondamentaux d'un CNN. Elles appliquent une opération de convolution à l'entrée, passant le résultat à la couche suivante.

- **Couches Convolutionnelles** : Effectuent des convolutions, qui combinent des filtres apprenables avec des données d'entrée pour produire des cartes de caractéristiques.
- **Couches de Pooling** : Réduisent la dimensionnalité de chaque carte de caractéristiques tout en conservant les informations les plus importantes. Les types courants incluent le Max Pooling et le Average Pooling.

**Exercice 2 : Construire un CNN Basique**
1. Créez un modèle CNN en utilisant des couches convolutionnelles et des couches de pooling.
2. Ajoutez une couche de convolution suivie d'une couche de MaxPooling, puis répétez le processus.
3. Ajoutez des couches Flatten et Dense à la fin pour la classification.

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

#### 14.5 Construire un Réseau de Neurones Convolutionnel avec Keras
[Retour en haut](#plan)

Construisons un CNN complet et plus détaillé en utilisant Keras.

**Exemple de code : Construire un CNN avec Keras**
```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

**Exercice 3 : Ajouter des Couches à un CNN**
1. Modifiez le modèle ci-dessus pour ajouter une troisième couche convolutionnelle et une couche de Dropout supplémentaire.
2. Compilez le modèle et affichez son résumé.

```python
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.summary()
```

#### 14.6 Entraîner le CNN
[Retour en haut](#plan)

Pour entraîner le CNN, vous devez préparer votre jeu de données, ce qui implique :
- Diviser les données en ensembles d'entraînement et de test.
- Normaliser les images.
- Appliquer des techniques d'augmentation de données pour augmenter la diversité de vos données d'entraînement sans collecter de nouvelles données.

**Exemple de code : Préparer les données et entraîner le modèle**
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

model.fit(training_set,
          steps_per_epoch=8000//32,
          epochs=25,
          validation_data=test_set,
          validation_steps=2000//32)
```

**Exercice 4 : Entraîner un CNN avec Data Augmentation**
1. Ajoutez des transformations supplémentaires dans `ImageDataGenerator` pour l'augmentation des données.
2. Entraînez le modèle avec les nouvelles transformations.

```python
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
```

#### 14.7 Évaluer le Modèle
[Retour en haut](#plan)

Après l'entraînement, évaluez la performance du modèle sur le jeu de test pour comprendre sa capacité à se généraliser à de nouvelles données.

**Exemple de code : Évaluer le modèle**
```python
score = model.evaluate(test_set)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

**Exercice 5 : Évaluation du modèle**
1. Utilisez le jeu de données de test pour évaluer le modèle.
2. Affichez la précision et la perte sur le jeu de test.

```python
test_loss, test_accuracy = model.evaluate(test_set)
print(f'Loss on test data: {test_loss}')
print(f'Accuracy on test data: {test_accuracy}')
```

---

### Chapitre 15 : Composants d'un CNN

#### 15.1 Convolution
[Retour en haut](#plan)

**Fonctionnement** :
- La convolution est une opération qui applique un filtre (ou noyau) sur l'image d'entrée pour extraire des caractéristiques importantes tout en préservant les relations spatiales. 
- Les caractéristiques extraites peuvent inclure des bords, des textures, et des motifs complexes.

**Hyperparamètres clés** :
1. **Taille du noyau (Kernel Size)** :
   - Détermine la taille de la matrice de filtre appliquée à l'image.
   - Un noyau 3x3 est couramment utilisé.
   
   **Exemple de code** :
   ```python
   from keras.layers import Conv2D

   model = Sequential()
   model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
   ```

2. **Nombre de filtres** :
   - Le nombre de filtres détermine le nombre de cartes de caractéristiques générées par la couche de convolution.
   - Plus de filtres permettent d'extraire plus de types de caractéristiques.

3. **Stride** :
   - Le stride est le nombre de pixels par lequel le filtre se déplace sur l'image.
   - Un stride plus grand réduit la taille de la carte de caractéristiques mais peut perdre des informations.

   **Exemple de code** :
   ```python
   model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu'))
   ```

4. **Padding** :
   - Le padding ajoute des pixels autour de l'image pour conserver les dimensions après la convolution.
   - Types de padding : "valid" (sans padding) et "same" (padding pour conserver la taille d'origine).

   **Exemple de code** :
   ```python
   model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
   ```

**Exercice 1 : Expérimenter avec la Convolution**
1. Créez un modèle CNN avec différentes tailles de noyaux (3x3, 5x5) et comparez les résultats.
2. Expérimentez avec différents nombres de filtres (32, 64) et observez l'impact sur les performances.
3. Testez différents strides (1, 2) et notez les changements dans la taille de la carte de caractéristiques.

```python
# Exercice pratique
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

#### 15.2 Activation Non-Linéaire (ReLU)
[Retour en haut](#plan)

**Fonction** :
- La fonction d'activation ReLU (Rectified Linear Unit) remplace toutes les valeurs négatives par zéro, introduisant ainsi de la non-linéarité dans le modèle. Cela permet au réseau de mieux capturer des relations complexes.

**Formule** : 
$$
y = \max(0, x)
$$


**Exemple de code** :
```python
def relu(x):
    return np.maximum(0, x)

# Appliquer ReLU à une matrice d'exemple
matrix = np.array([[-1, 2], [-3, 4]])
relu_matrix = relu(matrix)
print(relu_matrix)
```

**Exercice 2 : Comparer ReLU avec d'autres Fonctions d'Activation**
1. Implémentez les fonctions d'activation Sigmoid et Tanh.
2. Appliquez ReLU, Sigmoid, et Tanh à la même matrice d'exemple.
3. Comparez les résultats et discutez de l'impact de chaque fonction d'activation.

```python
import numpy as np

# Fonctions d'activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Matrice d'exemple
matrix = np.array([[-1, 2], [-3, 4]])

# Appliquer les fonctions d'activation
relu_matrix = relu(matrix)
sigmoid_matrix = sigmoid(matrix)
tanh_matrix = tanh(matrix)

print("ReLU :\n", relu_matrix)
print("Sigmoid :\n", sigmoid_matrix)
print("Tanh :\n", tanh_matrix)
```

#### 15.3 Pooling (Sous-échantillonnage)
[Retour en haut](#plan)

**Objectif** :
- Le pooling réduit la dimensionnalité de l'image tout en conservant les caractéristiques importantes, augmentant ainsi l'efficacité du calcul et réduisant le surapprentissage.

**Types** :
1. **Max Pooling** :
   - Prend la valeur maximale dans une fenêtre de taille définie (ex. 2x2).
   
   **Exemple de code** :
   ```python
   model.add(MaxPooling2D(pool_size=(2, 2)))
   ```

2. **Average Pooling** :
   - Calcule la moyenne des valeurs dans la fenêtre de sous-échantillonnage.

   **Exemple de code** :
   ```python
   from keras.layers import AveragePooling2D

   model.add(AveragePooling2D(pool_size=(2, 2)))
   ```

**Exercice 3 : Expérimenter avec le Pooling**
1. Créez un modèle CNN en utilisant le Max Pooling et l'Average Pooling.
2. Comparez les performances des deux modèles.
3. Expérimentez avec différentes tailles de fenêtres de pooling (2x2, 3x3).

```python
# Modèle avec Max Pooling
model_max = Sequential()
model_max.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model_max.add(MaxPooling2D(pool_size=(2, 2)))
model_max.add(Flatten())
model_max.add(Dense(128, activation='relu'))
model_max.add(Dense(10, activation='softmax'))

model_max.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_max.summary()

# Modèle avec Average Pooling
model_avg = Sequential()
model_avg.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model_avg.add(AveragePooling2D(pool_size=(2, 2)))
model_avg.add(Flatten())
model_avg.add(Dense(128, activation='relu'))
model_avg.add(Dense(10, activation='softmax'))

model_avg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_avg.summary()
```

---

### Chapitre 16 : Exemple Pratique avec MNIST et Fashion MNIST
[Retour en haut](#plan)

#### 16.1 MNIST
**Description** : 
- Le jeu de données MNIST contient 60 000 images d'entraînement et 10 000 images de test de chiffres manuscrits (28x28 pixels, en niveaux de gris).
- Il est couramment utilisé pour tester les algorithmes de reconnaissance d'image.

**Performance** : 
- Un CNN basique peut atteindre une précision supérieure à 99 %.

**Exercice 4 : Construire et Entraîner un CNN sur MNIST**
1. Chargez le jeu de données MNIST.
2. Construisez un CNN avec des couches de convolution, de pooling, de flattening et dense.
3. Entraînez le modèle et évaluez sa performance.

```python
from keras.datasets import mnist
from keras.utils import to_categorical

# Charger les données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Construire le modèle
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Évaluer le modèle
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

#### 16.2 Fashion

 MNIST
**Description** : 
- Le jeu de données Fashion MNIST est similaire à MNIST mais contient des images de vêtements avec 10 classes différentes.
- Il est utilisé pour des tâches de classification plus complexes que MNIST.

**Défi** : 
- Un CNN basique peut atteindre une précision de 90 %, et plus avec un ajustement.

**Exercice 5 : Construire et Entraîner un CNN sur Fashion MNIST**
1. Chargez le jeu de données Fashion MNIST.
2. Construisez un CNN avec des couches de convolution, de pooling, de flattening et dense.
3. Entraînez le modèle et évaluez sa performance.

```python
from keras.datasets import fashion_mnist

# Charger les données Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Construire le modèle
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Évaluer le modèle
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```


---
### Chapitre 17 : Apprentissage par Transfert
[Retour en haut](#plan)

#### 17.1 Concept
- **Objectif** : Utiliser un modèle pré-entraîné sur un large jeu de données pour améliorer la performance sur un problème spécifique.
- **Étapes** :
  - Utiliser un modèle existant pour la détection de caractéristiques.
  - Remplacer le classificateur final par un classificateur adapté à notre problème.
  - Entraîner ce nouveau classificateur sur notre jeu de données spécifique.

L'apprentissage par transfert consiste à tirer parti des réseaux de neurones pré-entraînés sur de vastes ensembles de données pour des tâches spécifiques. Cela permet de réduire le temps d'entraînement et d'améliorer les performances en utilisant les caractéristiques apprises par ces modèles.

**Exemple de modèle pré-entraîné : Inception V3**
- **Inception V3** est un modèle de classification d'images puissant, pré-entraîné sur le jeu de données ImageNet. Il peut être utilisé pour extraire des caractéristiques visuelles et construire des classificateurs personnalisés pour diverses applications.

#### 17.2 Exemple avec Inception V3

**Étapes de l'apprentissage par transfert :**
1. **Charger le modèle pré-entraîné :** Utiliser Inception V3 sans la couche de sortie pour extraire des caractéristiques.
2. **Ajouter un classificateur personnalisé :** Construire et ajouter des couches de classification spécifiques à notre problème.
3. **Compiler et entraîner le modèle :** Utiliser notre jeu de données pour entraîner le nouveau classificateur.

**Exercice 1 : Apprentissage par transfert avec Inception V3**
1. **Charger le modèle pré-entraîné sans la dernière couche :**

```python
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

base_model = InceptionV3(weights='imagenet', include_top=False)
```

2. **Ajouter des couches de classification personnalisées :**

```python
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
```

3. **Compiler et entraîner le modèle :**

```python
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Charger les données (exemple avec CIFAR-10)
from keras.datasets import cifar10
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
```

4. **Évaluer le modèle :**

```python
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

**Exercice 2 : Fine-tuning du modèle pré-entraîné**
1. **Débloquer certaines couches du modèle pré-entraîné :**

```python
for layer in base_model.layers[:249]:
    layer.trainable = False
for layer in base_model.layers[249:]:
    layer.trainable = True
```

2. **Compiler de nouveau le modèle avec un taux d'apprentissage plus faible :**

```python
from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
```

3. **Évaluer le modèle fine-tuné :**

```python
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

#### Structure des Dossiers de Données
[Retour en haut](#plan)

**Organisation des données :**
- **Emplacement du programme :** Le dossier `data` contient les sous-dossiers `train` et `validate`.
- **Données d'entraînement :**
  - `train/cats` : 1000 images de chats.
  - `train/dogs` : 1000 images de chiens.
- **Données de validation :**
  - `validate/cats` : 400 images de chats.
  - `validate/dogs` : 400 images de chiens.

**Téléchargement des données :**
Les données peuvent être téléchargées depuis [Kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) (train.zip).

**Exercice 3 : Organiser les données pour l'entraînement**
1. **Créer une structure de dossiers pour les données d'entraînement et de validation :**

```python
import os
import shutil

# Créer les répertoires si nécessaire
os.makedirs('data/train/cats', exist_ok=True)
os.makedirs('data/train/dogs', exist_ok=True)
os.makedirs('data/validate/cats', exist_ok=True)
os.makedirs('data/validate/dogs', exist_ok=True)

# Déplacer les images dans les répertoires correspondants
for file in cat_files_train:
    shutil.move(file, 'data/train/cats/')
for file in dog_files_train:
    shutil.move(file, 'data/train/dogs/')
for file in cat_files_val:
    shutil.move(file, 'data/validate/cats/')
for file in dog_files_val:
    shutil.move(file, 'data/validate/dogs/')
```

### Synthèse
[Retour en haut](#plan)

- **Résumé :** Les CNNs résolvent les problèmes d'analyse d'images en extrayant des cartes de caractéristiques et en utilisant des couches de convolution, de non-linéarité et de pooling.
- **Exemples :** Démonstration avec Fashion MNIST, et utilisation de l'apprentissage par transfert avec Inception V3.

**Exercice 4 : Projet de Fin de Chapitre**
1. **Définir un nouveau problème de classification d'images.**
2. **Utiliser un modèle pré-entraîné pour extraire des caractéristiques.**
3. **Construire un classificateur personnalisé et entraîner le modèle.**
4. **Évaluer et analyser les performances.**

**Exemple de projet : Classification de Fleurs**
- Utiliser le jeu de données [Flowers Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition) disponible sur Kaggle.
- Utiliser un modèle pré-entraîné comme ResNet50.
- Construire et entraîner un classificateur personnalisé pour classer les images de fleurs en cinq catégories différentes.

---

**Projet de Fin de Chapitre : Classification de Fleurs**

**1. Charger le modèle pré-entraîné :**

```python
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

base_model = ResNet50(weights='imagenet', include_top=False)
```

**2. Ajouter des couches de classification personnalisées :**

```python
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
```

**3. Compiler et entraîner le modèle :**

```python
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Charger les données de fleurs
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('flowers/train', target_size=(224, 224), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('flowers/validate', target_size=(224, 224), batch_size=32, class_mode='categorical')

model.fit(training_set, steps_per_epoch=2000//32, epochs=10, validation_data=test_set, validation_steps=800//32)
```

**4. Évaluer le modèle :**

```python
score = model.evaluate(test_set)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

Ce chapitre final fournit une compréhension approfondie de l'apprentissage par transfert et de son application pratique, vous offrant  des opportunités pour expérimenter avec des projets réels et approfondir votre maîtrise des réseaux de neurones convolutionnels ;).

---

### Annexe : 
[Retour en haut](#plan)

### Livre sur les Réseaux de Neurones Convolutifs (CNN) avec Keras




### Conclusion
[Retour en haut](#plan)
- Maîtriser les réseaux de neurones convolutifs (CNN) est crucial pour l'apprentissage non supervisé, en particulier dans le domaine de l'imagerie. Les CNN sont la composante principale pour extraire des caractéristiques détaillées des images, permettant d'utiliser des techniques non supervisées telles que les autoencodeurs pour découvrir des structures et motifs cachés dans les données visuelles. Une partie dédiée à l'imagerie dans ce cours explorera comment appliquer des techniques d'apprentissage non supervisé aux images. En combinant ces connaissances avec des méthodes avancées d'apprentissage non supervisé, vous serez mieux équipé pour traiter et analyser des données visuelles complexes. Les exemples et exercices de ce cours vous aideront à développer ces compétences et à les appliquer à des problèmes réels.
