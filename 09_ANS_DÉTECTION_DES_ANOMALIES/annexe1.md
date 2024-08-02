# Détection des Anomalies avec l'Apprentissage Non Supervisé

- Les étudiants apprendront à utiliser divers algorithmes pour détecter des anomalies : DBSCAN pour identifier des outliers dans des jeux de données en utilisant `pandas` et `matplotlib`; Isolation Forest pour détecter des anomalies dans les données d'assurance avec encodage des variables catégorielles et visualisation en 3D; et un autoencodeur pour détecter des anomalies dans des données standardisées. De plus, les étudiants exploreront par eux-mêmes l'utilisation de réseaux adversariaux génératifs (GANs) pour générer des données synthétiques et identifier les anomalies, ainsi que l'application des Deep Belief Networks (DBN) à travers des travaux de recherche.

# Détection des Anomalies avec l'Apprentissage Non Supervisé

**Description 1 : Utilisation de DBSCAN**
Les étudiants vont apprendre à utiliser l'algorithme DBSCAN pour détecter les anomalies dans un jeu de données. Ils comprendront comment prétraiter les données avec `pandas` et `StandardScaler`, et visualiser les résultats avec `matplotlib`.

**Description 2 : Utilisation de l'Isolation Forest**
Les étudiants vont découvrir comment appliquer l'algorithme Isolation Forest pour identifier les anomalies dans des données d'assurance. Ils apprendront à encoder les variables catégorielles, à utiliser `Pandas` pour la manipulation des données et à visualiser les anomalies détectées avec `matplotlib` en 3D.

**Description 3 : Détection d'Anomalies dans les Images**
Les étudiants vont explorer l'utilisation de l'Isolation Forest pour détecter les anomalies dans les images. Ils apprendront à quantifier les images, à entraîner un modèle de détection d'anomalies, et à utiliser OpenCV pour afficher les résultats, indiquant si une image est une anomalie ou non.

**Description 4 : Détection des Anomalies avec des Autoencodeurs et GANs**
Les étudiants vont apprendre à utiliser des autoencodeurs et des réseaux adversariaux génératifs (GANs) pour détecter des anomalies dans un ensemble de données. Ils comprendront comment prétraiter les données, encoder les variables catégorielles, entraîner un modèle d'autoencodeur, et utiliser les GANs pour générer des données synthétiques et identifier les anomalies.

```python
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers
import os
os.chdir('drive/My Drive/Anomaly')

# Charger les données
data = pd.read_csv("Anomaly dataset for deep learning.csv")
data.head()

# Encodage des étiquettes
for column in data.columns:
    if data[column].dtype == np.object:
        encoded = LabelEncoder()
        encoded.fit(data[column])
        data[column] = encoded.transform(data[column])
data.head()

# Vérification des valeurs nulles
list1 = [f"{i} - {sum(pd.isnull(data[i]))}" for i in data.columns]
print(list1)

# Prétraitement des données
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data = (data - mean) / std
data.head()

# Construction de l'autoencodeur
input_layer = tf.keras.layers.Input(shape=(data.shape[1],))
encoded = tf.keras.layers.Dense(128, activation="relu")(input_layer)
encoded = tf.keras.layers.Dense(64, activation="relu")(encoded)
decoded = tf.keras.layers.Dense(128, activation="relu")(encoded)
decoded = tf.keras.layers.Dense(data.shape[1], activation="sigmoid")(decoded)
autoencoder = tf.keras.models.Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

# Entraînement de l'autoencodeur
autoencoder.fit(data, data, epochs=100, batch_size=64)

# Calcul de l'erreur de reconstruction
predictions = autoencoder.predict(data)
reconstruction_error = np.mean((predictions - data)**2, axis=1)

# Seuil pour identifier les anomalies
threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)
anomalies = np.where(reconstruction_error > threshold)
anomaly_data = data.iloc[[i for i in anomalies[0]]]
print(anomaly_data)
```

Les étudiants vont aussi construire et entraîner un modèle GAN pour générer des données synthétiques et utiliser un modèle DBN pour détecter les anomalies.

```python
# Construction du modèle GAN
generator = tf.keras.Sequential()
generator.add(tf.keras.layers.Dense(64, input_dim=100, activation="relu"))
generator.add(tf.keras.layers.Dense(128, activation="relu"))
generator.add(tf.keras.layers.Dense(data.shape[1], activation="sigmoid"))

discriminator = tf.keras.Sequential()
discriminator.add(tf.keras.layers.Dense(128, input_dim=data.shape[1], activation="relu"))
discriminator.add(tf.keras.layers.Dense(64, activation="relu"))
discriminator.add(tf.keras.layers.Dense(1, activation="sigmoid"))
discriminator.compile(optimizer="adam", loss="binary_crossentropy")
discriminator.trainable = False

inputs = tf.keras.layers.Input(shape=(100,))
generated_data = generator(inputs)
validity = discriminator(generated_data)
gan = tf.keras.models.Model(inputs, validity)
gan.compile(optimizer="adam", loss="binary_crossentropy")

# Entraînement du GAN
for epoch in range(100):
    noise = np.random.normal(0, 1, (data.shape[0], 100))
    synthetic_data = generator.predict(noise)
    real_labels = np.ones((data.shape[0], 1))
    synthetic_labels = np.zeros((data.shape[0], 1))
    d_loss_real = discriminator.train_on_batch(data, real_labels)
    d_loss_synthetic = discriminator.train_on_batch(synthetic_data, synthetic_labels)
    g_loss = gan.train_on_batch(noise, real_labels)
    print(f"Epoch: {epoch+1}, D Loss (real): {d_loss_real:.4f}, D Loss (synthetic): {d_loss_synthetic:.4f}, G Loss: {g_loss:.4f}")

# Évaluation des données synthétiques avec le discriminateur
scores = discriminator.predict(data)
threshold = np.mean(scores) + 3 * np.std(scores)
anomalies = np.where(scores < threshold)
anomaly_data = data.iloc[[i for i in anomalies[0]]]
print(anomaly_data)
```

Les étudiants apprendront à définir et entraîner un modèle DBN pour la détection des anomalies.

```python
# Définition du modèle DBN
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.BatchNormalization(input_shape=(37,)))
model.add(tf.keras.layers.Dense(32, activation='sigmoid', name="layer1"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(16, activation='sigmoid', name="layer2"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(37, activation='sigmoid', name="output"))
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(data, data, epochs=50)
reconstructed_data = model.predict(data)

# Calcul de l'erreur de reconstruction et identification des anomalies
error = np.mean(np.abs(data - reconstructed_data))
threshold = 0.1
anomalies = np.abs(data - reconstructed_data) > threshold
print(data[anomalies])
```
