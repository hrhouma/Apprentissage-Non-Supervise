### Cours : Zero Padding dans les Réseaux de Neurones Convolutionnels (CNN)
# RÉFÉRENCE :
- https://deeplizard.com/learn/video/qSTv_m-KFk0
  
#### Introduction

Ce cours explore en détail le concept de **Zero Padding** dans les réseaux de neurones convolutionnels (CNN). Nous examinerons pourquoi le Zero Padding est important, comment il fonctionne, ses types, et comment l'implémenter en utilisant Keras. Nous inclurons également des exercices pratiques pour consolider votre compréhension.

#### Motivation pour le Zero Padding

Les convolutions dans les CNN réduisent les dimensions des canaux d'image. Par exemple, une image d'entrée de 28x28 convoluée avec un filtre 3x3 produit une sortie de 26x26. Cette réduction continue à chaque couche, ce qui peut poser problème en diminuant trop la taille des images, surtout si des données importantes se trouvent sur les bords.

**Problèmes liés à la réduction des dimensions :**
- **Perte d'information** : Les informations situées sur les bords de l'image peuvent être perdues.
- **Sorties trop petites** : Après plusieurs convolutions, les dimensions des images peuvent devenir si petites qu'elles perdent leur signification.

#### Qu'est-ce que le Zero Padding ?

Le Zero Padding consiste à ajouter une bordure de pixels avec la valeur zéro autour des images d'entrée. Cela permet de maintenir la taille originale de l'image après convolution. Par exemple, une image 4x4 convoluée avec un filtre 3x3 sans padding réduit la taille de la sortie à 2x2. Avec Zero Padding, la taille de la sortie reste 4x4.

**Formule pour la taille de sortie sans padding :**
\[ \text{Taille de sortie} = (\text{Taille d'entrée} - \text{Taille du filtre} + 1) \]

**Formule pour la taille de sortie avec padding :**
\[ \text{Taille de sortie} = \text{Taille d'entrée} \]

#### Types de Padding

1. **Valid Padding** : Pas de padding. Les dimensions de la sortie sont réduites.
   - **Exemple** : Une image 5x5 convoluée avec un filtre 3x3 produit une sortie 3x3.
   
2. **Same Padding** : Ajoute des zéros autour des bords de l'image pour que la taille de la sortie soit la même que celle de l'entrée.
   - **Exemple** : Une image 5x5 convoluée avec un filtre 3x3 produit une sortie 5x5.

#### Implémentation en Keras

Nous allons voir comment spécifier le padding dans Keras.

```python
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten

# Modèle sans padding (Valid Padding)
model_valid = Sequential([
    Dense(16, input_shape=(20, 20, 3), activation='relu'),
    Conv2D(32, kernel_size=(3,3), activation='relu', padding='valid'),
    Conv2D(64, kernel_size=(5,5), activation='relu', padding='valid'),
    Conv2D(128, kernel_size=(7,7), activation='relu', padding='valid'),
    Flatten(),
    Dense(2, activation='softmax')
])
model_valid.summary()

# Modèle avec Same Padding
model_same = Sequential([
    Dense(16, input_shape=(20,20,3), activation='relu'),
    Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
    Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'),
    Conv2D(128, kernel_size=(7,7), activation='relu', padding='same'),
    Flatten(),
    Dense(2, activation='softmax')
])
model_same.summary()
```

#### Exercices Pratiques

1. **Calcul de la taille de sortie sans padding**
   - **Question** : Quelle sera la taille de sortie d'une image 32x32 convoluée avec un filtre 5x5 sans padding ?
   - **Réponse** : 
   \[
   \text{Taille de sortie} = (32 - 5 + 1) \times (32 - 5 + 1) = 28 \times 28
   \]

2. **Calcul de la taille de sortie avec same padding**
   - **Question** : Quelle sera la taille de sortie d'une image 32x32 convoluée avec un filtre 5x5 avec same padding ?
   - **Réponse** : 
   \[
   \text{Taille de sortie} = 32 \times 32
   \]

3. **Implémentation en Keras**
   - **Exercice** : Créez un modèle CNN avec les spécifications suivantes :
     - Image d'entrée de taille 28x28 avec 1 canal (grayscale).
     - Une couche convolutionnelle avec 32 filtres de taille 3x3 et same padding.
     - Une couche convolutionnelle avec 64 filtres de taille 3x3 et valid padding.
     - Une couche de pooling max de taille 2x2.
     - Une couche entièrement connectée avec 128 neurones.
     - Une couche de sortie avec 10 neurones (pour une classification de 10 classes).

   ```python
   import keras
   from keras.models import Sequential
   from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

   model_ex = Sequential([
       Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28, 28, 1)),
       Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid'),
       MaxPooling2D(pool_size=(2,2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(10, activation='softmax')
   ])
   model_ex.summary()
   ```

#### Conclusion

Le Zero Padding est essentiel pour éviter la réduction excessive des dimensions des images dans les CNN. Il permet de conserver des informations importantes et facilite l'entraînement du modèle. En utilisant Keras, nous pouvons facilement spécifier le type de padding pour chaque couche convolutionnelle.

### Visionner la vidéo

Pour une explication visuelle et des exemples supplémentaires, regardez la vidéo "Zero Padding in Convolutional Neural Networks" de Deeplizard disponible [ici](https://youtu.be/YRhxdVk_sIs).
