#  Comparaison des Autoencodeurs

- Ce fichier présente une comparaison entre deux implémentations d'autoencodeurs appliquées au jeu de données MNIST, qui contient des images de chiffres manuscrits de 28x28 pixels. 
- Le premier code utilise un autoencodeur composé uniquement de couches denses (fully connected layers), tandis que le second code utilise un autoencodeur convolutionnel.
- Ce document comprend les codes complets, une table comparative détaillée, et des conclusions sur l'utilisation appropriée de chaque type d'autoencodeur.

---

# **Code 1 : Autoencodeur avec Couches Denses sur le Jeu de Données MNIST**

---

**Objectif :**

Cet extrait de code implémente un autoencodeur utilisant uniquement des couches denses (fully connected layers). Contrairement au modèle convolutionnel, cet autoencodeur ne capture pas les motifs spatiaux locaux, mais se concentre sur les relations globales entre les pixels. Ce modèle est généralement plus simple, mais peut être moins efficace pour des tâches complexes comme le traitement d'images.

**Code :**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])

X_train = X_train / 255
X_test = X_test / 255

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers import SGD

encoder = Sequential()
encoder.add(Flatten(input_shape=[28, 28]))
encoder.add(Dense(400, activation="relu"))
encoder.add(Dense(200, activation="relu"))
encoder.add(Dense(100, activation="relu"))
encoder.add(Dense(50, activation="relu"))
encoder.add(Dense(25, activation="relu"))

decoder = Sequential()
decoder.add(Dense(50, input_shape=[25], activation='relu'))
decoder.add(Dense(100, activation='relu'))
decoder.add(Dense(200, activation='relu'))
decoder.add(Dense(400, activation='relu'))
decoder.add(Dense(28 * 28, activation="sigmoid"))
decoder.add(Reshape([28, 28]))

autoencoder = Sequential([encoder, decoder])

autoencoder.compile(loss='binary_crossentropy', optimizer=SGD(lr=1.5))

history = autoencoder.fit(X_train, X_train, epochs=20, validation_data=(X_test, X_test))

decoded_imgs = autoencoder.predict(X_test)
```

---

# **Code 2 : Autoencodeur Convolutionnel sur le Jeu de Données MNIST**

---

**Objectif :**

Cet extrait de code implémente un autoencodeur convolutionnel, qui utilise des couches de convolution pour encoder les images en une représentation compressée, puis des couches de convolution transposée pour reconstruire les images originales. Ce type d'autoencodeur est particulièrement efficace pour traiter des données structurées en grille, comme des images, car il capte les motifs spatiaux locaux grâce aux convolutions.

**Code :**

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Charger le jeu de données MNIST
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

# Normaliser les données
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# Redimensionner les données pour les adapter au modèle
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

# Modèle de l'encodeur
input_img = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

# Modèle du décodeur
x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Autoencodeur complet
autoencoder = tf.keras.models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Entraîner l'autoencodeur
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

# Encoder et décoder quelques images
encoded_imgs = autoencoder.predict(X_test)
decoded_imgs = autoencoder.predict(X_test)

# Afficher les résultats
n = 10  # Nombre d'images à afficher
plt.figure(figsize=(20, 4))
for i in range(n):
    # Afficher les images originales
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Afficher les images reconstruites
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
```

---

# **Table Comparative des Deux Modèles**

| **Aspect**                  | **Code 1 : Autoencodeur avec Couches Denses** | **Code 2 : Autoencodeur Convolutionnel** |
|-----------------------------|-----------------------------------------------|------------------------------------------|
| **Type d'Autoencodeur**      | Couches Denses (Fully Connected)              | Convolutionnel                           |
| **Bibliothèques Importées**  | `pandas`, `numpy`, `tensorflow`, `matplotlib`| `numpy`, `tensorflow`, `matplotlib`      |
| **Jeu de Données**           | MNIST                                        | MNIST                                    |
| **Prétraitement des Données**| Normalisation, Redimensionnement en 2D       | Normalisation, Redimensionnement en 4D   |
| **Encodage**                 | `Dense`, `Flatten`                           | `Conv2D`, `MaxPooling2D`                 |
| **Dimensions d'Entrée**      | `(28, 28)`                                   | `(28, 28, 1)`                            |
| **Couches de l'Encodeur**    | - Dense(400, relu)<br>- Dense(200, relu)<br>- Dense(100, relu)<br>- Dense(50, relu)<br>- Dense(25, relu) | - Conv2D(16, 3x3, relu, padding='same')<br>- MaxPooling2D(2x2, padding='same')<br>- Conv2D(8, 3x3, relu, padding='same')<br>- MaxPooling2D(2x2, padding='same') |
| **Décodage**                 | `Dense`, `Reshape`                           | `Conv2D`, `UpSampling2D`                 |
| **Couches du Décodeur**      | - Dense(50, relu)<br>- Dense(100, relu)<br>- Dense(200, relu)<br>- Dense(400, relu)<br>- Dense(28*28, sigmoid)<br>- Reshape([28, 28]) | - Conv2D(8, 3x3, relu, padding='same')<br>- UpSampling2D(2x2)<br>- Conv2D(16, 3x3, relu, padding='same')<br>- UpSampling2D(2x2)<br>- Conv2D(1, 3x3, sigmoid, padding='same') |
| **Optimiseur**               | `SGD`                                         | `adam`                                    |
| **Fonction de Perte**        | `binary_crossentropy`                         | `binary_crossentropy`                    |
| **Nombre d'Époques**         | 20                                            | 50                                       |
| **Capacité de Compression**  | Apprend les relations globales entre les pixels via des couches entièrement connectées | Capte les motifs spatiaux locaux grâce aux convolutions |
| **Usage Recommandé**         | Données moins structurées ou simples         | Données structurées en grille (images, vidéos) |
| **Visualisation des Résultats** | Affichage des images originales et reconstruites | Affichage des images originales et reconstruites |

---

# **Conclusions Générales**

Ces deux exemples d'autoencodeurs démontrent l'application des réseaux de neurones pour la réduction de dimensionnalité et la reconstruction des images dans le contexte du jeu de données MNIST :

1. **Autoencodeur avec Couches Denses (Code 1)** : Ce modèle est plus simple et convient mieux pour des données où les relations globales entre les pixels sont plus importantes. Bien qu'il puisse être utilisé pour des images simples comme MNIST, il est moins performant pour des tâches complexes impliquant des images plus riches en détails.

2. **Autoencodeur Convolutionnel (Code 2)** : Ce modèle est particulièrement adapté aux données structurées en grille, comme les images, car il capture efficacement les motifs locaux grâce aux couches de convolution. Il est recommandé pour des tâches où les détails spatiaux locaux sont essentiels, comme la reconnaissance d'images.

En résumé, le choix entre un autoencodeur avec couches denses et un autoencodeur convolutionnel dépend de la nature des données et des exigences spécifiques de la tâche à accomplir. Le modèle convolutionnel est plus puissant pour traiter les images et autres données structurées, tandis que le modèle dense peut être suffisant pour des tâches plus simples ou des données non structurées.
