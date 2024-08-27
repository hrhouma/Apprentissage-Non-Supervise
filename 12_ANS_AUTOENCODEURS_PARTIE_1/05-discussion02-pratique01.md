# Dans notre cas 400 n'est pas supérieur à 784 ?


- Effectivement, 400 n'est pas supérieur à 784, et ce choix de nombre de neurones n'est pas basé sur une relation directe avec 784.POurtant, nous avons choisi 400 neurones pour commencer ?

---
# Discussion:
---

### Dimensions des Données d'Entrée pour MNIST

Pour les images du jeu de données MNIST :
- **Dimensions** : 28x28 pixels par image
- **Canal** : 1 (puisque ce sont des images en niveaux de gris)

Ainsi, lorsqu'on aplatit chaque image pour l'adapter à une couche dense, cela donne une taille de **784 valeurs** (28 x 28).

### Choix du Nombre de Neurones

#### **Pourquoi 400, puis 200, etc. ?**

- Le choix des nombres de neurones comme 400, 200, etc., ne correspond pas à une règle stricte de moitié ou de relation directe avec 784. 
- Je vous présente quelques raisons pour lesquelles ces nombres pourraient être choisis :

1. **Compression Progressive** : L'idée est de compresser progressivement les informations, mais pas nécessairement de diviser par deux à chaque étape. La réduction peut être décidée en fonction de l'objectif du modèle et de la complexité des données :
   - **400** : Un nombre légèrement inférieur à 784 permet de commencer la compression tout en capturant une grande partie des informations d'origine.
   - **200** : Réduction supplémentaire pour forcer le modèle à apprendre une représentation plus compacte.
   - **100** : Continue de réduire la dimension pour capturer l'essence des données.
   - **50** et **25** : Compression encore plus forte, pour capturer les caractéristiques essentielles avec moins de neurones.

2. **Conception Heuristique** : Ces nombres sont souvent choisis de manière empirique, c'est-à-dire en testant différentes architectures et en observant les performances. Il n'y a pas de règle stricte indiquant que les neurones doivent être divisés par deux, mais c'est une pratique courante car elle simplifie la conception du réseau tout en maintenant une réduction de dimensionnalité progressive.

3. **Éviter le Surapprentissage** : En réduisant progressivement le nombre de neurones, on essaie de prévenir le surapprentissage (overfitting) en forçant le modèle à généraliser les caractéristiques importantes plutôt qu'à mémoriser les données d'entrée.

### Résumé sur le Choix du Nombre de Neurones

- **Pas de Règle Absolue** : Il n'y a pas de règle universelle pour choisir les nombres de neurones. Les valeurs comme 400, 200, 100, etc., sont souvent choisies par heuristique, expérimentation, ou sur la base d'architectures existantes.
- **Objectif de Compression** : Le but est de compresser progressivement les informations, tout en permettant au modèle de capturer les caractéristiques essentielles nécessaires pour la reconstruction.
- **Expérimentation** : Le choix final dépendra des performances du modèle sur les données spécifiques après expérimentation et ajustements.

Ces choix sont plus une question d'équilibre entre la complexité du modèle, la capacité de calcul disponible, et l'objectif de l'application, plutôt que de suivre une formule mathématique stricte.



# Question 1 : Pourquoi 400 et non pas 800?

Dans ce code, nous utilisons un autoencodeur pour encoder et décoder les images du dataset MNIST. La question sur pourquoi utiliser 400 unités dans la couche dense du `encoder` au lieu de 800 se rapporte au choix des dimensions dans le modèle d'autoencodeur.

### Choix de 400 unités dans la première couche dense de l'encodeur

1. **Simplicité et efficacité**: Un autoencodeur est conçu pour compresser les données en une représentation de plus petite dimension avant de les reconstruire. Utiliser 400 unités permet de réduire la dimension tout en gardant suffisamment d'informations pour que la reconstruction soit efficace. Une couche avec 800 unités serait plus complexe et pourrait entraîner un modèle plus difficile à entraîner, en augmentant le risque d'overfitting.

2. **Profondeur du réseau**: L'autoencodeur que vous avez construit comporte plusieurs couches denses, avec chaque couche successive ayant un nombre décroissant d'unités. Cette structure favorise une extraction progressive des caractéristiques les plus importantes. En utilisant 400 unités plutôt que 800, vous permettez une compression plus rapide des données tout en maintenant une profondeur raisonnable pour l'extraction des caractéristiques.

3. **Diminution progressive de la dimension**: Vous avez choisi de réduire progressivement la dimension à travers les couches denses (400 → 200 → 100 → 50 → 25). Cette réduction progressive permet d’apprendre une représentation plus compacte des données tout en minimisant la perte d’information. Si vous aviez utilisé 800 unités dans la première couche dense, la diminution aurait dû être plus rapide, ce qui pourrait compliquer l'entraînement et la qualité de la reconstruction.

4. **Contrainte de calcul**: Avec 400 unités, le modèle est plus léger, ce qui réduit le temps de calcul et la mémoire nécessaire pour l'entraînement, par rapport à un modèle avec 800 unités. Cela permet un entraînement plus rapide tout en atteignant une bonne performance.

En résumé, 400 unités ont été choisies comme un bon compromis entre la capacité à capturer les caractéristiques importantes des images et la complexité du modèle. Cela permet d’entraîner efficacement l’autoencodeur tout en évitant un modèle trop complexe qui pourrait surapprendre les données d’entraînement.


# Question 2

Pourquoi ne pas avoir choisi 800 unités dans la première couche dense du modèle d'autoencodeur, et ensuite ajouté des couches de Dropout pour tester leur effet ?

### Réponse :
Le choix de ne pas utiliser 800 unités dans la première couche dense du modèle d'autoencodeur peut être justifié par les considérations suivantes :

1. **Complexité du modèle** : Utiliser 800 unités dans la première couche dense peut augmenter la complexité du modèle, ce qui pourrait rendre l'entraînement plus difficile et augmenter le risque de surapprentissage, surtout si les données disponibles sont limitées. Une complexité accrue nécessite souvent des mécanismes supplémentaires pour régulariser le modèle.

2. **Régularisation par Dropout** : Le Dropout est une technique puissante pour réduire le surapprentissage en désactivant aléatoirement une fraction des neurones pendant l'entraînement. Après avoir testé différents nombres d'unités dans la première couche, il peut être préférable d'ajouter des couches de Dropout pour voir si elles aident à améliorer la généralisation du modèle sur les données de test. Les tests avec Dropout permettent d'évaluer l'impact de la régularisation sur la performance du modèle.

3. **Test d'efficacité** : En testant des couches de Dropout après une configuration dense initiale avec 800 unités, on peut observer comment ces choix affectent la performance du modèle. Cela permet d’ajuster le nombre d’unités et le taux de Dropout en fonction des résultats, afin d’atteindre un équilibre optimal entre capacité de représentation et régularisation.

En résumé, il est judicieux de ne pas choisir d'emblée 800 unités sans évaluer leur impact sur la performance du modèle. L'ajout de couches de Dropout et la réalisation de tests permettent de déterminer si cette configuration permet d'améliorer les résultats ou si un autre nombre d'unités serait plus efficace.


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

### Modifications apportées :
1. **Augmentation de la première couche dense à 800 unités** : Cela permet de maintenir une représentation initiale proche de la taille d'entrée (784 pixels par image).
2. **Ajout de Dropout** : Les couches de `Dropout` avec un taux de 0.3 sont ajoutées après les deux premières couches denses pour réduire le surapprentissage en éliminant aléatoirement 30% des neurones pendant l'entraînement. Cela aide à régulariser le modèle et à améliorer sa capacité de généralisation.

Nous pouvons ajuster les taux de dropout selon les besoins pour trouver le meilleur équilibre entre performance et régularisation.
