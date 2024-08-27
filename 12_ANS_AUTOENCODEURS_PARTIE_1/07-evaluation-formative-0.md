# testez avec Dropout


# Exemple : version modifiée avec une première couche dense à 800 unités + couches de Dropout pour régulariser le modèle :

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
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout
from tensorflow.keras.optimizers import SGD

encoder = Sequential()
encoder.add(Flatten(input_shape=[28, 28]))
encoder.add(Dense(800, activation="relu"))
encoder.add(Dropout(0.3))
encoder.add(Dense(400, activation="relu"))
encoder.add(Dropout(0.3))
encoder.add(Dense(200, activation="relu"))
encoder.add(Dense(100, activation="relu"))
encoder.add(Dense(50, activation="relu"))
encoder.add(Dense(25, activation="relu"))

decoder = Sequential()
decoder.add(Dense(50, input_shape=[25], activation='relu'))
decoder.add(Dense(100, activation='relu'))
decoder.add(Dense(200, activation='relu'))
decoder.add(Dense(400, activation='relu'))
decoder.add(Dense(800, activation='relu'))
decoder.add(Dense(28 * 28, activation="sigmoid"))
decoder.add(Reshape([28, 28]))

autoencoder = Sequential([encoder, decoder])

autoencoder.compile(loss='binary_crossentropy', optimizer=SGD(lr=1.5))

history = autoencoder.fit(X_train, X_train, epochs=20, validation_data=(X_test, X_test))

decoded_imgs = autoencoder.predict(X_test)
```
------
### Instructions :
-----

1. **Augmentation de la première couche dense à 800 unités** : Cela permet de maintenir une représentation initiale proche de la taille d'entrée (784 pixels par image).
2. **Ajout de Dropout** : Les couches de `Dropout` avec un taux de 0.3 sont ajoutées après les deux premières couches denses pour réduire le surapprentissage en éliminant aléatoirement 30% des neurones pendant l'entraînement. Cela aide à régulariser le modèle et à améliorer sa capacité de généralisation.

Nous pouvons ajuster les taux de dropout selon les besoins pour trouver le meilleur équilibre entre performance et régularisation.

