# Testez avec Dropout


***Le Dropout désactive aléatoirement des neurones à chaque itération pour éviter que le modèle ne dépende trop de certains neurones spécifiques. Cela force le réseau à apprendre de manière plus générale, améliorant ainsi sa capacité à généraliser sur de nouvelles données. Même si les meilleurs neurones sont temporairement désactivés, d'autres neurones apprennent des caractéristiques similaires. Lors de la phase de test, tous les neurones sont actifs, ce qui permet d'utiliser pleinement les meilleures caractéristiques apprises.***

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
### Instructions #1 :
-----

1. **Augmentation de la première couche dense à 800 unités** : Cela permet de maintenir une représentation initiale proche de la taille d'entrée (784 pixels par image).
2. **Ajout de Dropout** : Les couches de `Dropout` avec un taux de 0.3 sont ajoutées après les deux premières couches denses pour réduire le surapprentissage en éliminant aléatoirement 30% des neurones pendant l'entraînement. Cela aide à régulariser le modèle et à améliorer sa capacité de généralisation.

Nous pouvons ajuster les taux de dropout selon les besoins pour trouver le meilleur équilibre entre performance et régularisation.



# Annexe01 - Pourquoi utilise-t-on un Dropout de 30% dans le modèle ?

L'utilisation d'un taux de Dropout de 30% dans un modèle de réseau de neurones, y compris dans l'autoencodeur que vous avez conçu, se justifie par les raisons suivantes :

1. **Prévention du surapprentissage (overfitting)** : Le Dropout est une technique de régularisation qui empêche le modèle de s'adapter trop étroitement aux données d'entraînement. En désactivant aléatoirement 30% des neurones à chaque itération d'entraînement, le modèle est obligé de ne pas trop dépendre d'un ensemble spécifique de neurones, ce qui renforce sa capacité à généraliser sur de nouvelles données.

2. **Favoriser la robustesse** : Le Dropout oblige le réseau à apprendre des caractéristiques plus robustes et moins spécifiques à certaines combinaisons de neurones. Avec 30% de neurones désactivés, le réseau devient plus résilient et capable de maintenir de bonnes performances même si certains neurones sont absents, ce qui améliore sa tolérance au bruit et aux variations des données.

3. **Équilibre entre régularisation et conservation de l'information** : Un taux de 30% est souvent considéré comme un bon compromis entre une régularisation efficace et la conservation de la capacité d'apprentissage du réseau. Des taux plus élevés, comme 50% ou plus, pourraient trop réduire l'information transmise à travers le réseau, tandis que des taux plus bas pourraient ne pas offrir une régularisation suffisante.

4. **Expérimentation courante** : Dans la pratique, 30% est un taux souvent utilisé car il a montré de bons résultats dans de nombreux contextes, en particulier pour des modèles avec une architecture relativement profonde. Il est suffisamment agressif pour réduire le surapprentissage tout en permettant au réseau de conserver une grande partie de sa capacité à apprendre et à modéliser les données.

En résumé, un taux de Dropout de 30% est utilisé pour trouver un juste équilibre entre la régularisation et la performance du modèle, aidant à prévenir le surapprentissage tout en permettant au réseau de continuer à apprendre efficacement.

# Annexe02 - Comment fonctionne le Dropout ?

### Comment fonctionne le Dropout ?

Le Dropout est une technique de régularisation utilisée dans les réseaux de neurones pour réduire le surapprentissage (overfitting). Voici comment il fonctionne et pourquoi il est efficace :

1. **Principe de base** :
   - Lors de chaque itération d'entraînement, le Dropout désactive (met à zéro) aléatoirement un pourcentage spécifié de neurones dans le réseau de neurones.
   - Par exemple, si un Dropout de 30% est appliqué à une couche, cela signifie que, pour chaque itération d'entraînement, environ 30% des neurones de cette couche seront désactivés de manière aléatoire.

2. **Fonctionnement** :
   - **Étape 1 :** Avant chaque mise à jour des poids lors de la phase d'entraînement, un sous-ensemble aléatoire de neurones est sélectionné pour être désactivé.
   - **Étape 2 :** Ces neurones désactivés ne participent pas à la transmission de l'information lors de cette itération, c'est-à-dire qu'ils n'envoient aucune valeur à la couche suivante et n'ont pas de contribution dans la mise à jour des poids.
   - **Étape 3 :** Lors de la phase d'entraînement suivante, un nouveau sous-ensemble de neurones est désactivé, et ainsi de suite.

3. **Impact sur l'entraînement** :
   - En désactivant aléatoirement des neurones, le réseau est obligé d'apprendre des caractéristiques plus générales et robustes, car il ne peut pas compter sur l'activation d'un ensemble fixe de neurones pour chaque prédiction.
   - Le réseau devient plus résilient aux variations des données, car chaque neurone doit apprendre à contribuer de manière significative à la prédiction même lorsqu'il est combiné avec différents ensembles d'autres neurones.

4. **Phase d'inférence (test/prédiction)** :
   - Une fois l'entraînement terminé, le Dropout n'est pas utilisé lors de la phase de test ou de prédiction.
   - Au lieu de cela, tous les neurones sont actifs, et leurs poids sont généralement ajustés en les multipliant par le taux de Dropout utilisé pendant l'entraînement (par exemple, multiplier par 0.7 si le Dropout était de 30%). Cela permet de compenser le fait que tous les neurones sont maintenant actifs.

5. **Avantages** :
   - **Régularisation** : Réduit le surapprentissage en empêchant les neurones de trop se spécialiser ou de devenir dépendants les uns des autres.
   - **Amélioration de la généralisation** : Le modèle devient plus généraliste, c'est-à-dire qu'il est plus susceptible de bien fonctionner sur de nouvelles données non vues pendant l'entraînement.

### Illustration

Imaginez un réseau de neurones avec 10 neurones dans une couche donnée. Lors d'une itération, 3 neurones (30%) sont désactivés aléatoirement. Lors de la prochaine itération, un autre ensemble de neurones (peut-être les mêmes ou différents) sera désactivé. Cela se répète jusqu'à la fin de l'entraînement. Lors de la phase de test, tous les neurones sont activés, mais leurs contributions sont réduites pour correspondre à l'effet du Dropout pendant l'entraînement.

Le Dropout est donc une méthode efficace pour améliorer la robustesse du modèle en s'assurant que chaque neurone apprend à contribuer indépendamment des autres.

# Annexe 03 - Choisir aléatoirement les neurones avec Dropout : Perd-on les meilleurs neurones ?

### Choisir aléatoirement les neurones avec Dropout : Perd-on les meilleurs neurones ?

Le Dropout sélectionne effectivement les neurones à désactiver de manière aléatoire à chaque itération d'entraînement. Cette approche peut soulever des préoccupations quant à la possibilité de "perdre" temporairement les meilleurs neurones, ceux qui sont les plus efficaces ou les plus importants pour la tâche en cours. Cependant, voici pourquoi cette méthode est bénéfique et pourquoi le risque de perdre les meilleurs neurones n'est pas une réelle préoccupation :

1. **Régularisation efficace** :
   - L'idée derrière le Dropout est de forcer le réseau de neurones à ne pas dépendre uniquement d'un petit ensemble de neurones, mais plutôt à distribuer l'apprentissage à travers tout le réseau. Cela permet au modèle de devenir plus robuste, car même si certains neurones particulièrement efficaces sont désactivés temporairement, d'autres neurones apprendront également à prendre en charge ces fonctions.

2. **Apprentissage redondant** :
   - Dans un réseau de neurones, de nombreux neurones finissent par apprendre des caractéristiques similaires. Ainsi, même si certains neurones particulièrement efficaces sont désactivés lors d'une itération, d'autres neurones capables d'apprendre des informations similaires restent actifs et continuent de contribuer à l'apprentissage.

3. **Amélioration de la généralisation** :
   - En désactivant des neurones de manière aléatoire, le modèle devient plus généraliste. Si le réseau apprend à effectuer des prédictions avec un sous-ensemble aléatoire de neurones, il sera moins probable qu'il surapprenne des particularités spécifiques aux données d'entraînement. Cela augmente la capacité du modèle à généraliser à de nouvelles données non vues.

4. **Compensation lors de la phase de test** :
   - Lors de la phase de test ou de prédiction, le Dropout n'est pas utilisé, et tous les neurones sont actifs. Le modèle bénéficie donc de toutes les connaissances apprises par l'ensemble des neurones pendant l'entraînement. Les poids des neurones sont ajustés pour compenser le fait que le réseau avait été entraîné avec des neurones désactivés, garantissant ainsi que les meilleurs neurones, ainsi que tous les autres, contribuent pleinement aux prédictions finales.

5. **Minimisation des risques grâce à l'aléatoire** :
   - Le fait que la désactivation soit aléatoire et change à chaque itération permet d'assurer que, même si certains neurones particulièrement performants sont désactivés à un moment donné, ils ne seront pas systématiquement désactivés. Cela permet à ces neurones de continuer à contribuer à l'apprentissage global du modèle.

En résumé, bien que le Dropout désactive aléatoirement des neurones, ce mécanisme est conçu pour éviter la dépendance excessive à quelques neurones et encourager l'apprentissage à travers tout le réseau. Cela aide à créer un modèle plus robuste et capable de généraliser efficacement, sans perdre l'information critique que les meilleurs neurones peuvent fournir.

