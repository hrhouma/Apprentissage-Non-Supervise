# Pratique 01 pour Autoencodeurs

Ce fichier présente trois extraits de code qui implémentent des autoencodeurs, une forme de réseau de neurones utilisée pour l'apprentissage non supervisé, principalement pour la réduction de dimensionnalité et l'apprentissage de caractéristiques. Chaque extrait de code a un objectif différent et démontre l'application d'autoencodeurs sur divers types de données.

# Pour plus de détails ==> Drive du cours 
* 09-Introduction_aux_auto-encodeurs ==> 03-Exercices
---

# **Code 1 : Autoencodeur Simple sur Données Synthétiques**

---


**Objectif :**

- Cet extrait de code montre comment implémenter un autoencodeur simple sur un jeu de données synthétiques généré à l'aide de la fonction `make_blobs` de `sklearn`. 
- L'objectif principal est de réduire la dimensionnalité du jeu de données de trois à deux dimensions, puis de visualiser les résultats pour comprendre l'efficacité de la réduction. Le code ajoute également du bruit aléatoire aux données pour simuler des conditions réelles, puis entraîne l'autoencodeur pour reconstruire les données à partir de cette représentation réduite.

**Schéma du Processus :**

```
       +-----------------------------------+
       |   Données Originales (X1, X2)     |
       +-----------------------------------+
                     |
                     v
       +-----------------------------------+
       |  Ajout de Bruit (Génération X3)   |
       +-----------------------------------+
                     |
                     v
       +-----------------------------------+
       |  Données avec Bruit (X1, X2, X3)  |
       +-------------+---------------------+
                     |
                     v
       +-------------+---------------------+
       |       Autoencodeur                |
       |  Encoder : 3 -> 2                 |
       +-------------+---------------------+
                     |
                     v
       +-------------+---------------------+
       |  Données Réduites (X1, X2)         |
       +-----------------------------------+

```

**Code :**

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

data = make_blobs(n_samples=300, n_features=2, centers=2, cluster_std=1.0, random_state=101)

X, y = data

np.random.seed(seed=101)
z_noise = np.random.normal(size=len(X))
z_noise = pd.Series(z_noise)

feat = pd.DataFrame(X)
feat = pd.concat([feat, z_noise], axis=1)
feat.columns = ['X1', 'X2', 'X3']

feat.head()

plt.scatter(feat['X1'], feat['X2'], c=y)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(feat['X1'], feat['X2'], feat['X3'], c=y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

encoder = Sequential()
encoder.add(Dense(units=2, activation='relu', input_shape=[3]))

decoder = Sequential()
decoder.add(Dense(units=3, activation='relu', input_shape=[2]))

autoencoder = Sequential([encoder, decoder])

autoencoder.compile(loss='mse', optimizer=SGD(lr=1.5))

history = autoencoder.fit(feat, feat, epochs=20)

encoded_2dim = encoder.predict(feat)

encoded_2dim = pd.DataFrame(data=encoded_2dim, columns=['X1', 'X2'])

plt.scatter(encoded_2dim['X1'], encoded_2dim['X2'], c=y)
```

---

# **Code 2 : Autoencodeur Convolutionnel sur le Jeu de Données MNIST**

---

**Objectif :**

Cet extrait de code applique un autoencodeur convolutionnel au jeu de données MNIST (**voir l'annexe 01**), qui contient des images de chiffres manuscrits (28x28 pixels). L'autoencodeur réduit la dimensionnalité des images, puis tente de reconstruire les images d'origine à partir de cette représentation compressée. Le but est de démontrer la capacité de l'autoencodeur à capturer l'essence des données en une représentation de plus faible dimension tout en préservant les caractéristiques importantes pour la reconstruction.

**Schéma du Processus :**

```
       +----------------------------+
       |    Image Originale (28x28)   |
       +-------------+---------------+
                     |
                     v
       +-------------+---------------+
       |    Autoencodeur CNN   |
       |  Encoder -> Décoder   |
       +-------------+---------------+
                     |
                     v
       +-------------+---------------+
       |   Image Reconstituée (28x28) |
       +----------------------------+
```

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

# **Conclusions Générales**

Ces deux exemples montrent comment les autoencodeurs peuvent être appliqués à différents types de données, qu'il s'agisse de données synthétiques, d'images de chiffres manuscrits (MNIST), ou d'autres types de données. L'objectif général est de montrer comment ces modèles peuvent réduire la dimensionnalité des données tout en préservant suffisamment d'informations pour permettre une reconstruction fidèle des données d'origine. Les visualisations créées après l'entraînement des modèles aident à évaluer l'efficacité de la compression et de la reconstruction.

# Anenxe 01 - Est-ce vraiment convolutionnel ?


Le deuxième extrait de code prétend (;) être un autoencodeur convolutionnel appliqué au jeu de données MNIST, qui contient des images de chiffres manuscrits (28x28 pixels). Mais ne vous laissez pas berner—c'est une blague ! En réalité, cet extrait de code met en œuvre un autoencodeur simple, utilisant uniquement *des couches denses (fully connected layers), et non des couches convolutionnelles.*

L'objectif de cet autoencodeur est de réduire la dimensionnalité des images tout en conservant l'essence des données. Ensuite, il tente de reconstruire les images originales à partir de cette représentation compressée. Ce processus démontre la capacité de l'autoencodeur à capturer les caractéristiques importantes des données, même après la réduction de leur dimensionnalité, tout en permettant une reconstruction fidèle des images.


