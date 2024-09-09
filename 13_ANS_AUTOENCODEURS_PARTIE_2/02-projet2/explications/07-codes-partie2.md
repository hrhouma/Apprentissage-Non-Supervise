
# PARTIE 1

```ssh
import sys
sys.path.append("..")
```

# PARTIE 2
```ssh
import numpy as np # algèbre linéaire
import pandas as pd # traitement des données, entrée/sortie de fichiers CSV (par exemple pd.read_csv)

import os
for dirname, _, filenames in os.walk('/../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

```

# PARTIE 3

```ssh
import cv2
import tqdm
import tarfile
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model
```

# PARTIE 4

```ssh
ATTRS_NAME = "../input/lfw_attributes.txt"
IMAGES_NAME = "../input/lfw-deepfunneled.tgz"
RAW_IMAGES_NAME = "../input/lfw.tgz"
```

# PARTIE 5

```ssh
def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
```

# PARTIE 6

```ssh
def load_lfw_dataset(
        use_raw=False,
        dx=80, dy=80,
        dimx=45, dimy=45):
    df_attrs = pd.read_csv(ATTRS_NAME, sep='\t', skiprows=1)
    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])
    imgs_with_attrs = set(map(tuple, df_attrs[["person", "imagenum"]].values))

    toutes_les_photos = []
    ids_photos = []

    from tqdm.notebook import tqdm
    with tarfile.open(RAW_IMAGES_NAME if use_raw else IMAGES_NAME) as f:
        for m in tqdm(f.getmembers()):
            if m.isfile() and m.name.endswith(".jpg"):
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                img = img[dy:-dy, dx:-dx]
                img = cv2.resize(img, (dimx, dimy))

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

# PARTIE 7

```ssh
X, attr = load_lfw_dataset(use_raw=True,dimx=38,dimy=38)
X = X.astype('float32') / 255.0

img_shape = X.shape[1:]
IMG_SHAPE = X.shape[1:]
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
```

# PARTIE 8

```ssh
import matplotlib.pyplot as plt
plt.title('sample image')
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(X[i])

print("X shape:",X.shape)
print("attr shape:",attr.shape)
```

# PARTIE 9

```ssh
import tensorflow as tf
import keras, keras.layers as L
from tensorflow.python.keras.backend import get_session
s = get_session()
```

# PARTIE 10

```ssh
def build_pca_autoencoder(img_shape, code_size=32):
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size))

    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(np.prod(img_shape)))
    decoder.add(L.Reshape(img_shape))

    return encoder, decoder
```

# PARTIE 11

```ssh
encoder, decoder = build_pca_autoencoder(img_shape, code_size=32)

inp = L.Input(img_shape)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inp, reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')
```

# PARTIE 12

```ssh
autoencoder.fit(x=X_train,y=X_train,epochs=32,
                validation_data=[X_test,X_test])
```

# PARTIE 13

```ssh
def visualize(img,encoder,decoder):
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    plt.imshow(reco.clip(0,1))
    plt.show()
```

# PARTIE 14

```ssh
score = autoencoder.evaluate(X_test,X_test,verbose=0)
print("Final MSE:",score)

for i in range(5):
    img = X_test[i]
    visualize(img,encoder,decoder)
```

# PARTIE 15

```ssh
def build_deep_autoencoder(img_shape,code_size=32):
    H,W,C = img_shape

    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Conv2D(16,[2,2],activation='relu',padding='same'))
    encoder.add(L.Conv2D(32,[2,2],activation='relu',padding='same'))
    encoder.add(L.MaxPooling2D())
    encoder.add(L.Flatten())
    encoder.add(L.Dense(1024,activation='relu'))
    encoder.add(L.Dense(code_size))

    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(1024,activation='relu'))
    decoder.add(L.Dense(2048,activation='relu'))
    decoder.add(L.Dense(38*38*3,activation='relu'))
    decoder.add(L.Reshape([38,38,3]))

    return encoder,decoder
```

# PARTIE 16

```ssh
get_dim = lambda layer: np.prod(layer.output.shape[1:]) if hasattr(layer.output, 'shape') else np.prod(layer.output.get_shape().as_list()[1:])

for code_size in [1, 8, 32, 128, 512, 1024]:
    encoder, decoder = build_deep_autoencoder(img_shape, code_size=code_size)

    print((code_size,), img_shape)
    assert encoder.output_shape[1:] == (code_size,)
    assert decoder.output_shape[1:] == img_shape
    assert len(encoder.trainable_weights) >= 6
    assert len(decoder.trainable_weights) >= 6

    for layer in encoder.layers + decoder.layers:
        dim = get_dim(layer)
        assert dim >= code_size

print("All tests passed!")
```

# PARTIE 17

```ssh
img_shape = X.shape[1:]
IMG_SHAPE = X.shape[1:]

get_dim = lambda layer: np.prod(layer.output.shape[1:]) if hasattr(layer.output, 'shape') else np.prod(layer.output.get_shape().as_list()[1:])
```

# PARTIE 18

```ssh
for code_size in [1, 8, 32, 128, 512]:
    encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=code_size)
    print("Testing code size %i" % code_size)

    assert encoder.output_shape[1:] == (code_size,)
    assert decoder.output_shape[1:] == IMG_SHAPE
    assert len(encoder.trainable_weights) >= 6
    assert len(decoder.trainable_weights) >= 6

    for layer in encoder.layers + decoder.layers:
        dim = get_dim(layer)
        assert dim >= code_size

print("All tests passed!")

```

# PARTIE 19
```ssh
encoder,decoder = build_deep_autoencoder(img_shape,code_size=32)

inp = L.Input(img_shape)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inp,reconstruction)
autoencoder.compile('adamax','mse')
```
# PARTIE 20

```ssh
autoencoder.fit(x=X_train,y=X_train,epochs=32,
                validation_data=[X_test,X_test])

reconstruction_mse = autoencoder.evaluate(X_test,X_test,verbose=0)
assert len(encoder.output_shape)==2 and encoder.output_shape[1]==32
print("Final MSE:", reconstruction_mse)
for i in range(5):
    img = X_test[i]
    visualize(img,encoder,decoder)
```

# PARTIE 21

```ssh
def apply_gaussian_noise(X, sigma=0.1):
    noise = np.random.normal(0, sigma, X.shape[1:])
    noisy_image = X +

 noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image
```

# PARTIE 22

```ssh
X_original = X[:100].copy()

try:
    theoretical_std = (X_original.std()**2 + 0.5**2)**.5
    our_std = apply_gaussian_noise(X_original, sigma=0.5).std()
    tolerance = 0.18
    assert abs(theoretical_std - our_std) < tolerance
    assert abs(apply_gaussian_noise(X_original, sigma=0.5).mean() - X_original.mean()) < 0.01
except AssertionError as e:
    print(e)
    X = X_original.copy()

```

# PARTIE 23
```ssh
plt.subplot(1, 4, 1)
plt.imshow(X[0])

plt.subplot(1, 4, 2)
plt.imshow(np.clip(apply_gaussian_noise(X[:1], sigma=0.01)[0], 0, 1))

plt.subplot(1, 4, 3)
plt.imshow(np.clip(apply_gaussian_noise(X[:1], sigma=0.1)[0], 0, 1))

plt.subplot(1, 4, 4)
plt.imshow(np.clip(apply_gaussian_noise(X[:1], sigma=0.5)[0], 0, 1))
```

# PARTIE 24

```ssh
encoder,decoder = build_deep_autoencoder(img_shape,code_size=512)
assert encoder.output_shape[1:]==(512,)

inp = L.Input(img_shape)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inp,reconstruction)
autoencoder.compile('adamax','mse')
```

# PARTIE 25

```ssh
for i in range(50):
    print("Epoch %i/50, Generating corrupted samples..."%i)
    X_train_noise = apply_gaussian_noise(X_train)
    X_test_noise = apply_gaussian_noise(X_test)

    autoencoder.fit(x=X_train_noise,y=X_train,epochs=1,
                    validation_data=[X_test_noise,X_test])
```

# PARTIE 26

```ssh
denoising_mse = autoencoder.evaluate(apply_gaussian_noise(X_test), X_test, verbose=0)
print("Final MSE:", denoising_mse)
for i in range(5):
    img = X_test[i]
    visualize(img, encoder, decoder)
```

# PARTIE 27

```ssh
encoder.save("./encoder.keras")
decoder.save("./decoder.keras")
```


# PARTIE 28

```ssh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

images = X_train
codes = encoder.predict(X_train)
assert len(codes) == len(images)

nn = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(codes)

def get_similar(image, n_neighbors=5):
    assert image.ndim == 3
    
    code = encoder.predict(image[None])
    distances, indices = nn.kneighbors(code, n_neighbors=n_neighbors)
    
    return distances[0], images[indices[0]]

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

show_similar(X_test[2])
show_similar(X_test[500])
show_similar(X_test[66])

```

# PARTIE 29

```ssh
for _ in range(5):
    image1,image2 = X_test[np.random.randint(0,len(X_test),size=2)]

    code1, code2 = encoder.predict(np.stack([image1,image2]))

    plt.figure(figsize=[10,4])
    for i,a in enumerate(np.linspace(0,1,num=7)):

        output_code = code1*(1-a) + code2*(a)
        output_image = decoder.predict(output_code[None])[0]

        plt.subplot(1,7,i+1)
        plt.imshow(output_image)
        plt.title("a=%.2f"%a)
        
    plt.show()
```

