
# Instructions pour les étapes : 


---


# PARTIE 1

```python
import sys
sys.path.append("..")
```

---

# PARTIE 2

```python
import numpy as np # algèbre linéaire
import pandas as pd # traitement des données, entrée/sortie de fichiers CSV (par exemple pd.read_csv)

# Les fichiers de données d'entrée sont disponibles dans le répertoire "../input/".
# Par exemple, exécuter ceci (en cliquant sur exécuter ou en appuyant sur Shift+Enter) listera tous les fichiers dans le répertoire d'entrée.

import os
for dirname, _, filenames in os.walk('/../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Tous les résultats que vous écrivez dans le répertoire actuel sont sauvegardés comme sortie.
```

---

# PARTIE 3

```python
import cv2
import tqdm
import tarfile
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
```

---

# PARTIE 4

```python
# Chemins des fichiers d'attributs et des images LFW
ATTRS_NAME = "../input/lfw_attributes.txt"
IMAGES_NAME = "../input/lfw-deepfunneled.tgz"
RAW_IMAGES_NAME = "../input/lfw.tgz"
```

---

# PARTIE 5

```python
def decode_image_from_raw_bytes(raw_bytes):
    # Décodage des bytes en image
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
```

---

# PARTIE 6

```python
def load_lfw_dataset(
        use_raw=False,
        dx=80, dy=80,
        dimx=45, dimy=45):

    # Lire les attributs
    df_attrs = pd.read_csv(ATTRS_NAME, sep='\t', skiprows=1)
    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])
    imgs_with_attrs = set(map(tuple, df_attrs[["person", "imagenum"]].values))

    # Lire les photos
    toutes_les_photos = []
    ids_photos = []

    # Utilisation de tqdm pour afficher une barre de progression
    from tqdm.notebook import tqdm
    with tarfile.open(RAW_IMAGES_NAME if use_raw else IMAGES_NAME) as f:
        for m in tqdm(f.getmembers()):
            if m.isfile() and m.name.endswith(".jpg"):
                # Préparer l'image
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                img = img[dy:-dy, dx:-dx]
                img = cv2.resize(img, (dimx, dimy))

                # Extraire l'identifiant de la personne et l'ajouter aux données collectées
                nom_fichier = os.path.split(m.name)[-1]
                nom_splitté = nom_fichier[:-4].replace('_', ' ').split()
                id_personne = ' '.join(nom_splitté[:-1])
                numéro_photo = int(nom_splitté[-1])
                if (id_personne, numéro_photo) in imgs_with_attrs:
                    toutes_les_photos.append(img)
                    ids_photos.append({'person': id_personne, 'imagenum': numéro_photo})

    ids_photos = pd.DataFrame(ids_photos)
    toutes_les_photos = np.stack(toutes_les_photos).astype('uint8')
    tous_les_attrs = ids_photos.merge(df_attrs, on=('person', 'imagenum')).drop(["person", "imagenum"], axis=1)

    return toutes_les_photos, tous_les_attrs
```

---

# PARTIE 7

```python
X, attr = load_lfw_dataset(use_raw=True, dimx=38, dimy=38)
X = X.astype('float32') / 255.0

img_shape = X.shape[1:]
IMG_SHAPE = X.shape[1:]
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
```

---

# PARTIE 8

```python
import matplotlib.pyplot as plt
plt.title('sample image')
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(X[i])

print("X shape:",X.shape)
print("attr shape:",attr.shape)
```

---

# PARTIE 9

```python
import tensorflow as tf
import keras, keras.layers as L
from tensorflow.python.keras.backend import get_session
s = get_session()
```

---

# PARTIE 10

```python
def build_pca_autoencoder(img_shape, code_size=32):
    # Définir l'encodeur
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size))

    # Définir le décodeur
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(np.prod(img_shape)))
    decoder.add(L.Reshape(img_shape))

    return encoder, decoder
```

---

# PARTIE 11

```python
encoder, decoder = build_pca_autoencoder(img_shape, code_size=32)

# Définir le modèle autoencodeur
inp = L.Input(img_shape)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inp, reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')
```

---

# PARTIE 12 (Cela prendra environ **15 minutes**)

```python
autoencoder.fit(x=X_train, y=X_train, epochs=32, validation_data=[X_test, X_test])
```

---

# PARTIE 13

```python
def visualize(img, encoder, decoder):
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2, -1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    plt.imshow(reco.clip(0, 1))
    plt.show()
```

---

# PARTIE 14

```python
score = autoencoder.evaluate(X_test, X_test, verbose=0)
print("Final MSE:", score)

for i in range(5):
    img = X_test[i]
    visualize(img, encoder, decoder)
```



# PARTIE 15

```python
def build_deep_autoencoder(img_shape, code_size=32):
    """Une version plus complexe de l'autoencodeur avec des couches de convolution"""
    H, W, C = img_shape

    # Construction de l'encodeur
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Conv2D(16, [2, 2], activation='relu', padding='same'))
    encoder.add(L.Conv2D(32, [2, 2], activation='relu', padding='same'))
    encoder.add(L.MaxPooling2D())  # Réduit la taille des images
    encoder.add(L.Flatten())
    encoder.add(L.Dense(1024, activation='relu'))
    encoder.add(L.Dense(code_size))  # La sortie de l'encodeur

    # Construction du décodeur
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(1024, activation='relu'))
    decoder.add(L.Dense(2048, activation='relu'))
    decoder.add(L.Dense(38 * 38 * 3, activation='relu'))
    decoder.add(L.Reshape([38, 38, 3]))

    return encoder, decoder
```

---

# PARTIE 16

```python
get_dim = lambda layer: np.prod(layer.output.shape[1:]) if hasattr(layer.output, 'shape') else np.prod(layer.output.get_shape().as_list()[1:])

# Tester l'autoencodeur avec différentes tailles de code
for code_size in [1, 8, 32, 128, 512, 1024]:
    encoder, decoder = build_deep_autoencoder(img_shape, code_size=code_size)

    print((code_size,), img_shape)
    assert encoder.output_shape[1:] == (code_size,), "L'encodeur doit produire un code de la taille spécifiée"
    assert decoder.output_shape[1:] == img_shape, "Le décodeur doit produire une image de forme valide"
    assert len(encoder.trainable_weights) >= 6, "L'encodeur doit contenir au moins 3 couches denses"
    assert len(decoder.trainable_weights) >= 6, "Le décodeur doit contenir au moins 3 couches denses"

    for layer in encoder.layers + decoder.layers:
        dim = get_dim(layer)
        assert dim >= code_size, f"La couche {layer.name} est plus petite que le goulot d'étranglement ({dim} unités)"
```

---

# PARTIE 17

```python
# Redéfinir la forme de l'image
img_shape = X.shape[1:]
IMG_SHAPE = X.shape[1:]

# Vérifier la forme des autoencodeurs avec différentes tailles de code
get_dim = lambda layer: np.prod(layer.output.shape[1:]) if hasattr(layer.output, 'shape') else np.prod(layer.output.get_shape().as_list()[1:])
```

---

# PARTIE 18

```python
# Vérification de différentes tailles de code
for code_size in [1, 8, 32, 128, 512]:
    encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=code_size)
    print(f"Test de la taille de code {code_size}")

    # Vérification de la taille de sortie de l'encodeur
    assert encoder.output_shape[1:] == (code_size,), "L'encodeur doit produire un code de taille correcte"

    # Vérification de la taille de sortie du décodeur
    assert decoder.output_shape[1:] == IMG_SHAPE, "Le décodeur doit produire une image de taille correcte"

    # Vérification du nombre de couches dans l'encodeur et le décodeur
    assert len(encoder.trainable_weights) >= 6, "L'encodeur doit contenir au moins 3 couches"
    assert len(decoder.trainable_weights) >= 6, "Le décodeur doit contenir au moins 3 couches"

    # Vérification de la capacité de chaque couche
    for layer in encoder.layers + decoder.layers:
        dim = get_dim(layer)
        assert dim >= code_size, f"La couche {layer.name} est plus petite que le goulot d'étranglement ({dim} unités)"
```

---

# PARTIE 19

```python
# Redéfinir l'autoencodeur avec une taille de code de 32
encoder, decoder = build_deep_autoencoder(img_shape, code_size=32)

# Créer le modèle complet
inp = L.Input(img_shape)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inp, reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')
```

---

# PARTIE 20 (Cela prendra environ **10 minutes**)

```python
# Entraîner l'autoencodeur
autoencoder.fit(x=X_train, y=X_train, epochs=32, validation_data=[X_test, X_test])

# Évaluer les performances de l'autoencodeur
reconstruction_mse = autoencoder.evaluate(X_test, X_test, verbose=0)
assert len(encoder.output_shape) == 2 and encoder.output_shape[1] == 32, "Assurez-vous que l'encodeur contient la bonne taille de code"
print("Final MSE:", reconstruction_mse)

# Visualiser les résultats pour 5 images de test
for i in range(5):
    img = X_test[i]
    visualize(img, encoder, decoder)
```

---

# PARTIE 21

```python
def apply_gaussian_noise(X, sigma=0.1):
    """
    Ajoute du bruit provenant d'une distribution normale avec un écart-type sigma
    :param X: un tenseur d'image de forme [batch, height, width, 3]
    """
    # Générer du bruit gaussien
    noise = np.random.normal(0, sigma, X.shape[1:])
    
    # Ajouter le bruit à l'image
    noisy_image = X + noise
    
    # Limiter les valeurs des pixels à la plage [0, 1]
    noisy_image = np.clip(noisy_image, 0, 1)
    
    return noisy_image
```

---

# PARTIE 22

```python
# Sauvegarder l'état original des images
X_original = X[:100].copy()

# Tester l'application du bruit gaussien
try:
    theoretical_std = (X_original.std()**2 + 0.5**2)**.5
    our_std = apply_gaussian_noise(X_original, sigma=0.5).std()

    # Vérification avec une tolérance
    tolerance = 0.18
    assert abs(theoretical_std - our_std) < tolerance, "L'écart-type ne correspond pas. Utilisez sigma comme écart-type."

    # Vérifier la moyenne
    assert abs(apply_gaussian_noise(X_original, sigma=0.5).mean() - X_original.mean()) < 0.01, "La moyenne a changé. Ajoutez un bruit de moyenne nulle."

except AssertionError as e:
    print(e)
    X = X_original.copy()
```

---

# PARTIE 23

```python
# Visualiser les effets du bruit gaussien
plt.subplot(1, 4, 1)
plt.imshow(X[0])

plt.subplot(1, 4, 2)
plt.imshow(np.clip(apply_gaussian_noise(X[:1], sigma=0.01)[0], 0, 1))

plt.subplot(1, 4, 3)
plt.imshow(np.clip(apply_gaussian_noise(X[:1], sigma=0.1)[0], 0, 1))

plt.subplot(1, 4, 4)
plt.imshow(np.clip(apply_gaussian_noise(X[:1], sigma=0.5)[0], 0, 1))
```

---

# PARTIE 24

```python
# Construire un encodeur/décodeur avec une taille de code de 512
encoder, decoder = build_deep_autoencoder(img_shape, code_size=512)
assert encoder.output_shape[1:] == (512,), "L'encodeur doit produire un code de taille correcte"

# Créer le modèle
inp = L.Input(img_shape)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inp, reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')
```

---

# PARTIE 25 (Cela prendra **1 heure** au moins)

```python
# Boucle d'entraînement avec bruit gaussien (corruption des échantillons)
for i in range(50):
    print(f"Epoch {i+1}/50, Génération d'échantillons corrompus...")
    X_train_noise = apply_gaussian_noise(X_train)
    X_test_noise = apply_gaussian_noise(X_test)

    autoencoder.fit(x=X_train_noise, y=X_train, epochs=1, validation_data=[X_test_noise, X_test])
```

---

## PARTIE 26

```python
# Évaluer les performances du débruitage avec l'autoencodeur
denoising_mse = autoencoder.evaluate(apply_gaussian_noise(X_test), X_test, verbose=0)
print("Final MSE:", denoising_mse)

# Visualiser les images originales et reconstruites après débruitage
for i in range(5):
    img = X_test[i]
    visualize(img, encoder, decoder)
```

---

## PARTIE 27

```python
# Sauvegarder les modèles encodeur et décodeur entraînés
encoder.save("./encoder.keras")
decoder.save("./decoder.keras")
```

---

## PARTIE 28

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Images et codes extraits de l'encodeur
images = X_train  # Dataset d'images
codes = encoder.predict(X_train)  # Encode toutes les images avec l'autoencodeur
assert len(codes) == len(images)

# Utiliser NearestNeighbors pour trouver les plus proches voisins dans l'espace latent
nn = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(codes)

# Fonction pour obtenir les images similaires dans l'espace latent
def get_similar(image, n_neighbors=5):
    assert image.ndim == 3, "image must be [height, width, channels]"
    
    # Encoder l'image (obtenir son vecteur latent)
    code = encoder.predict(image[None])  # [None] ajoute la dimension batch
    
    # Obtenir les plus proches voisins dans l'espace latent
    distances, indices = nn.kneighbors(code, n_neighbors=n_neighbors)
    
    return distances[0], images[indices[0]]

# Fonction pour afficher les images similaires
def show_similar(image):
    distances, neighbors = get_similar(image, n_neighbors=11)
    
    plt.figure(figsize=[8,6])
    plt.subplot(3, 4, 1)
    plt.imshow(image)
    plt.title("Original image")
    
    for i in range(11):
        plt.subplot(3, 4, i+2)
        plt.imshow(neighbors[i])
        plt.title(f"Dist=%.3f" % distances[i])
    plt.show()

# Exemple d'utilisation
# Remplacez X_test par vos données de test (devrait être un tableau 4D avec la forme [n_samples, height, width, channels])

# Tester avec différentes images
show_similar(X_test[2])    # Exemple 1 : afficher les images similaires pour l'image à l'index 2
show_similar(X_test[500])  # Exemple 2 : afficher les images similaires pour l'image à l'index 500
show_similar(X_test[66])   # Exemple 3 : afficher les images similaires pour l'image à l'index 66
```

---

## PARTIE 29

```python
# Générer une interpolation entre deux images dans l'espace latent
for _ in range(5):
    image1, image2 = X_test[np.random.randint(0, len(X_test), size=2)]

    code1, code2 = encoder.predict(np.stack([image1, image2]))

    plt.figure(figsize=[10, 4])
    for i, a in enumerate(np.linspace(0, 1, num=7)):
        # Générer un code interpolé entre les deux images
        output_code = code1 * (1 - a) + code2 * a
        # Décoder le code interpolé en image
        output_image = decoder.predict(output_code[None])[0]

        plt.subplot(1, 7, i + 1)
        plt.imshow(output_image)
        plt.title("a=%.2f" % a)
        
    plt.show()
```
