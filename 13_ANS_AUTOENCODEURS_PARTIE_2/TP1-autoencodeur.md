# **Tutoriel : Création et entraînement d'un autoencodeur convolutif pour la classification d'images**

---

# **Objectif du TP :**
Le but de ce travail est de développer un autoencodeur convolutif pour encoder et reconstruire des images de deux espèces marines : les dauphins et les requins. Après avoir entraîné l'autoencodeur, nous utiliserons un **SVM (Support Vector Machine)** pour classifier les images en fonction des embeddings (représentations compactes des images) générés par l'encodeur.

---

# **Avant de commencer : C'est quoi un autoencodeur ?**

Un **autoencodeur** est un type de réseau de neurones qui apprend à compresser les données d'entrée en une représentation plus petite et à reconstruire les données originales à partir de cette représentation. Imaginez que vous avez une grande image de dauphin et que vous voulez la représenter avec moins d'informations, mais sans trop perdre de détails. L'autoencodeur "apprend" à le faire.

- **Encoder** : La première partie du réseau compresse les images (ou les données) en un vecteur plus petit. Ce vecteur contient les caractéristiques principales.
- **Decoder** : La deuxième partie du réseau essaie de reconstruire l'image originale à partir de cette représentation compacte.

---
# **Étape 1 : Configuration de l'environnement de travail**
---

Pour ce TP, nous utiliserons **Python** avec les bibliothèques suivantes :
- **Keras** et **TensorFlow** pour construire le modèle d'autoencodeur.
- **scikit-learn** pour appliquer le SVM et les métriques d'évaluation.
- **Google Colab** est recommandé pour l'exécution du modèle, car il offre la possibilité d'utiliser un GPU gratuitement, ce qui accélérera le processus d'entraînement.

**Installation des librairies** :
```bash
!pip install keras tensorflow scikit-learn matplotlib
```

**Utilisation du GPU dans Colab** :
Allez dans **Modifier > Paramètres du notebook > Accélérateur matériel** et sélectionnez **GPU**. Cela vous permettra de former des réseaux de neurones plus rapidement.

---
# **Étape 2 : Explication des notions clés**
---

#### **C’est quoi une couche Conv2D ?**

La couche **Conv2D** (Convolution 2D) est une couche clé dans les réseaux de neurones convolutifs (CNNs), qui est utilisée pour extraire les caractéristiques d'une image.

- **Conv** signifie "convolution", une opération qui applique un filtre (ou un noyau) sur une image pour en extraire des caractéristiques importantes. Par exemple, si vous passez un filtre sur une image, cela peut détecter les contours des objets dans cette image.
- **2D** signifie que nous travaillons avec des images en deux dimensions (longueur et largeur), donc nous appliquons la convolution sur des images 2D.

Chaque filtre utilisé dans la couche Conv2D capte différents motifs (contours, textures, etc.). En général, les couches convolutives sont les premières couches des réseaux de neurones utilisés pour la reconnaissance d'image.

```python
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
```
Ici, nous utilisons 32 filtres de taille 3x3 pour détecter des caractéristiques dans l'image d'entrée.

---

#### **C'est quoi un SVM (Support Vector Machine) ?**

Un **SVM (Support Vector Machine)** est un algorithme d'apprentissage supervisé utilisé principalement pour la classification. L'idée principale du SVM est de trouver une "frontière" qui sépare au mieux les différentes classes dans vos données. 

Dans notre cas, après avoir extrait les embeddings (représentations compactes) des images de dauphins et de requins, nous utiliserons un SVM pour classifier ces images.

Par exemple, si vous avez des points de données représentant des dauphins et des requins sur un graphique, le SVM tentera de tracer une ligne qui sépare ces points le mieux possible.

---
# **Étape 3 : Préparation des données**
---

Nous allons travailler avec un ensemble de données comprenant **4200 images** de deux espèces marines :
- **Ensemble d'entraînement** : 1 800 images de dauphins et 1 800 images de requins.
- **Ensemble de test** : 300 images de dauphins et 300 images de requins.

Pour entraîner l'autoencodeur, nous diviserons les données d'entraînement en **données d'entraînement** et **données de validation**.

#### **Prétraitement des images**
Il est recommandé de normaliser les images en divisant chaque pixel par 255, car les pixels ont des valeurs comprises entre 0 et 255.

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'path_to_data', target_size=(128, 128), batch_size=32, class_mode='input', subset='training')
validation_generator = train_datagen.flow_from_directory(
    'path_to_data', target_size=(128, 128), batch_size=32, class_mode='input', subset='validation')
```

---
### **Étape 4 : Conception de l'autoencodeur**
---

Nous allons maintenant construire notre **autoencodeur convolutif**. Il est composé de deux parties : **l'encodeur** et **le décodeur**.

#### **Encoder : Extraction des caractéristiques de l'image**

L'encodeur comprime l'image originale en un vecteur de caractéristiques plus petit.

```python
def encoder(input_layer):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)
    return encoded
```
- **MaxPooling2D** réduit la taille de l'image tout en conservant les caractéristiques importantes. Cela permet de diminuer la complexité du modèle.

#### **Decoder : Reconstruction de l'image à partir des caractéristiques extraites**

Le décodeur prend la sortie de l'encodeur et tente de reconstruire l'image d'origine.

```python
def decoder(encoded):
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded
```
- **UpSampling2D** est l'opposé de MaxPooling2D, il augmente la taille de l'image.
- **Sigmoid** est utilisée comme fonction d'activation pour garantir que les pixels sont entre 0 et 1.

---
### **Étape 5 : Compilation et entraînement du modèle**
---

Nous compilons maintenant l'autoencodeur en utilisant l'optimiseur **adam** et la fonction de perte **mse** (Mean Squared Error). L'optimiseur ajuste les poids du modèle pour minimiser la différence entre l'image originale et l'image reconstruite.

```python
input_layer = Input(shape=(128, 128, 3))  # Taille de l'image (128x128x3 canaux)
autoencoder = Model(input_layer, decoder(encoder(input_layer)))

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
```

**Entraînement du modèle** :
```python
history = autoencoder.fit(train_generator, epochs=50, validation_data=validation_generator)
```

---
# **Étape 6 : Évaluation du modèle**
---

#### **Reconstruction des images**

Après avoir entraîné le modèle, nous évaluons sa capacité à reconstruire les images :

```python
reconstructed_images = autoencoder.predict(test_images)

# Affichage des images originales et reconstruites
import matplotlib.pyplot as plt
n = 5  # Afficher 5 images
for i in range(n):
    plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i])
    plt.subplot(2, n, i + n + 1)
    plt.imshow(reconstructed_images[i])
plt.show()
```

#### **Application du SVM pour la classification**

Nous utilisons la partie **encodeur** pour extraire les embeddings des images et appliquons un **SVM linéaire** pour la classification :

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

embeddings = encoder_model.predict(test_images)
embeddings_flattened = embeddings.reshape(len(embeddings), -1)  # Aplatir les embeddings

svm = SVC(kernel='linear')
svm.fit(embeddings_flattened, test_labels)

# Évaluation
accuracy = svm.score(embeddings_flattened, test_labels)
print("Accuracy:", accuracy)
```

---

# **Étape 7 : Visualisation avec t-SNE**

Pour visualiser les embeddings en 2D, nous utilisons t-SNE :

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings_flattened)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=test_labels)
plt.colorbar()
plt.show()
```

---

### **Conclusion**

Dans ce tutoriel, nous avons vu comment :
1. Créer un autoencodeur convolutif pour compresser et reconstruire des images.
2. Extraire les embeddings des images et les utiliser pour la classification avec un SVM.
3. Visualiser les embeddings en 2 dimensions avec t-SNE.

