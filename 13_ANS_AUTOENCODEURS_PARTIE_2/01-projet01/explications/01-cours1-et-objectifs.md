# **Dénoiser des images avec des autoencodeurs**

----
# **Objectif du projet**
----

L'objectif de ce projet est de montrer à vos étudiants comment un modèle de réseau neuronal, appelé autoencodeur, peut être utilisé pour résoudre un problème pratique : **le débruitage d'images**. Dans ce cas, nous avons ajouté un bruit artificiel à des images et nous allons entraîner un modèle à les restaurer dans leur version originale. 

Nous allons également combiner ce débruitage avec un autre modèle, appelé **classificateur**, pour reconnaître les chiffres présents sur les images après leur débruitage.

# **Pourquoi faire cela ?**
Dans la vie réelle, les données sont souvent corrompues ou bruitées. Par exemple, une image peut être floue ou pixelisée, et l'objectif est de restaurer ces images afin de pouvoir mieux les utiliser (pour la reconnaissance d'objets, par exemple).

# **Concepts importants :**
1. **Autoencodeur** : Un réseau neuronal qui apprend à compresser et à décompresser des données. Il prend une image bruitée en entrée, la compresse dans un espace de plus petite dimension, puis la reconstruit pour la débruiter.
2. **Classificateur** : Un réseau neuronal qui prend une image en entrée et lui attribue une étiquette (ici, un chiffre entre 0 et 9).

# **Étapes du projet :**

# 1. **Préparation des données**
Nous utilisons le jeu de données **MNIST**, un ensemble d'images de chiffres manuscrits. Ces images sont transformées pour être facilement utilisables par nos modèles, notamment en normalisant les valeurs des pixels (en les ramenant entre 0 et 1).

# 2. **Ajout de bruit**
Pour rendre l'exercice plus réaliste, nous ajoutons du bruit à ces images. Cela signifie que nous modifions les images originales en ajoutant des pixels aléatoires. L'idée est d'apprendre au modèle à corriger ces erreurs et à restaurer l'image d'origine.

# 3. **Création d'un classificateur**
Un **classificateur** est un modèle qui associe une étiquette (ici un chiffre) à une image. Avant de débruiter les images, nous utilisons un classificateur pour voir à quel point il est performant sur des images bruitées et non bruitées. 

# 4. **Création de l'autoencodeur**
L'autoencodeur est un modèle qui apprend à enlever le bruit. Il passe par deux étapes :
- **Encodage** : L'image est compressée (réduction des informations inutiles).
- **Décodage** : L'image est reconstruite à partir de cette compression, en enlevant le bruit ajouté.

# 5. **Dénombrement du bruit**
Nous entraînons l'autoencodeur à partir d'images bruitées et nous vérifions s'il réussit à restaurer ces images à leur état d'origine.

# 6. **Combinaison de l'autoencodeur et du classificateur**
Nous combinons ensuite les deux modèles : l'autoencodeur nettoie d'abord les images, puis le classificateur les analyse pour reconnaître les chiffres.

# **Conclusion**
- Ce projet montre comment on peut utiliser des réseaux de neurones pour résoudre un problème complexe : la restauration d'images corrompues. 
- Ce type de méthode est utilisé dans de nombreux domaines, comme la vision par ordinateur, où les images doivent être améliorées avant de pouvoir être analysées automatiquement.


------------
# **Tâche 1 : Importation des bibliothèques**
------------

La première étape consiste à importer toutes les bibliothèques nécessaires pour la manipulation des données, la visualisation et la construction des modèles.

```python
import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.utils import to_categorical
```

- **Numpy** : Pour manipuler les tableaux de données numériques.
- **MNIST** : Le dataset des chiffres manuscrits.
- **Keras** : Une API haut niveau qui permet de créer et de former des modèles de réseaux de neurones.
- **Matplotlib** : Pour visualiser les résultats.

------------
## **Tâche 2 : Prétraitement des données**
------------

Le prétraitement est essentiel pour préparer les données à l'entraînement des modèles.

### **Explication :**
- **Normalisation** : Les valeurs des pixels des images (qui vont de 0 à 255) sont ramenées entre 0 et 1 en divisant par 255.
- **Reshape** : Les images de 28x28 pixels sont transformées en vecteurs de taille 784 (28x28).

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float')/255.
x_test = x_test.astype('float')/255.

x_train = np.reshape(x_train, (60000, 784))
x_test = np.reshape(x_test, (10000, 784))
```

------------
# **Tâche 3 : Ajout de bruit**
------------

Pour entraîner l'autoencodeur à débruiter des images, nous devons d'abord ajouter du bruit aux images d'entraînement et de test. Le bruit est ajouté en utilisant la fonction `np.random.rand`, qui génère des valeurs aléatoires entre 0 et 1.

```python
x_train_noisy = x_train + np.random.rand(60000, 784) * 0.9
x_test_noisy = x_test + np.random.rand(10000, 784) * 0.9

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
```

### **Pourquoi ajouter du bruit ?**
Cela permet de simuler un environnement avec des données corrompues, obligeant le modèle à apprendre comment restaurer les données originales à partir de leur version bruitée.


------------
# **Tâche 4 : Construction et entraînement du classificateur**
------------

Un classificateur est un réseau de neurones qui apprend à associer des étiquettes à des données. Ici, nous créons un modèle de type `Sequential` avec des couches denses entièrement connectées. La dernière couche utilise la fonction d'activation `softmax` pour la classification.

```python
classifier = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

classifier.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit(x_train, y_train, epochs=3, batch_size=512)
```

### **Pourquoi classifier les images ?**
Cela nous permet de mesurer l’impact du bruit sur la capacité du modèle à reconnaître des chiffres et de voir comment le modèle performe après débruitage.

------------
# **Tâche 5 : Construction de l’autoencodeur**
------------

Un autoencodeur est un modèle qui apprend à comprimer les données (encodage) et à les reconstruire (décodage). Dans notre cas, l'autoencodeur prend en entrée une image bruitée et tente de la restaurer à sa version d'origine.

```python
input_image = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_image)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_image, decoded)
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
```

- **Encodage** : Réduction des dimensions à 64.
- **Décodage** : Reconstruction des données initiales à partir des données encodées.

### **Pourquoi utiliser un autoencodeur ?**
L'autoencodeur apprend une représentation compacte des données, puis il apprend à restaurer les données d'origine. Dans ce cas, il apprend à restaurer une image bruitée.

------------
# **Tâche 6 : Entraînement de l'autoencodeur**
------------

L'entraînement de l'autoencodeur se fait en utilisant les images bruitées comme entrée et les images originales comme cibles.

```python
autoencoder.fit(
    x_train_noisy, x_train,
    epochs=100, batch_size=512,
    validation_split=0.2, verbose=False,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
)
```

- **EarlyStopping** : Arrête l'entraînement si l'amélioration stagne, ce qui permet d’éviter le surapprentissage.

------------
# **Tâche 7 : Images débruitées**
------------

Après l'entraînement, nous utilisons l'autoencodeur pour générer des images débruitées à partir des images bruitées.

```python
preds = autoencoder.predict(x_test_noisy)
plot(x_test_noisy, None)
plot(preds, None)
```

### **Visualisation :**
- **Images bruitées** : Affiche les images après ajout de bruit.
- **Images débruitées** : Affiche les images après que l'autoencodeur a tenté de les restaurer.

------------
# **Tâche 8 : Modèle composite (autoencodeur + classificateur)**
------------

Enfin, nous combinons l'autoencodeur et le classificateur en un seul modèle. Ce modèle prend une image bruitée, la nettoie avec l'autoencodeur, puis la classe avec le classificateur.

```python
input_image = Input(shape=(784,))
x = autoencoder(input_image)
y = classifier(x)

denoise_and_classify = Model(input_image, y)

predictions = denoise_and_classify.predict(x_test_noisy)
```

### **Pourquoi combiner ?**
Cela permet de comparer les performances de classification avant et après débruitage, et de montrer comment un modèle de classification peut tirer parti d'une étape de prétraitement automatisée pour améliorer ses résultats.




----
# **Annexe : Explication supplémentaire**
---

L'objectif de ce travail est de montrer comment utiliser des autoencodeurs pour débruiter des images, et également combiner ce processus avec un classificateur. Le problème que nous allons résoudre consiste à débruiter des images du dataset MNIST, qui contient des chiffres manuscrits de 0 à 9, et à les classifier après débruitage.

# **Les concepts clés abordés :**
1. **Autoencodeur** : Un réseau neuronal utilisé pour apprendre une représentation comprimée (encodée) des données.
2. **Classifier** : Un modèle de machine learning qui attribue une étiquette à une donnée.
3. **Dénombrement de bruit** : Ajouter du bruit à une image, puis apprendre à restaurer la version originale.
4. **Optimisation du modèle** : Utilisation de fonctions de perte et d'optimiseurs pour améliorer les performances.
