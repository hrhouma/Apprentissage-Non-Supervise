Voici la suite avec la **Partie 15 : Construction d'un autoencodeur profond**.

---

# Partie 15 : Construction d'un autoencodeur profond

## Description
Dans cette partie, nous construisons un autoencodeur plus complexe avec des couches de convolution. Cette architecture profonde permet au modèle de mieux capturer les caractéristiques visuelles des images, ce qui améliore la qualité des reconstructions par rapport à un modèle simple.

## Code

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

## Justification

### 1. Encodeur avec des couches de convolution
- **`L.Conv2D(...)`** : les couches de convolution sont utilisées pour extraire les caractéristiques spatiales de l'image. Elles permettent au modèle de capturer des informations locales importantes telles que les bords, les textures, et les motifs complexes dans l'image.
  
- **`L.MaxPooling2D()`** : cette couche réduit la taille des images (réduction de dimensionnalité) tout en conservant les informations les plus importantes. Cela permet de réduire le nombre de paramètres à apprendre, rendant l'entraînement plus efficace.

- **`L.Flatten()` et **`L.Dense(1024)`** : après les couches de convolution, nous aplatissons l'image en un vecteur, puis appliquons une couche dense pour réduire encore la dimension avant d'arriver à la couche de code latent.

### 2. Décodeur pour reconstruire l'image
- **`decoder.add(L.Dense(1024, activation='relu'))`** : le décodeur commence par transformer le vecteur de code en un format plus large.
  
- **`L.Reshape(...)`** : après avoir redimensionné le vecteur dense en une image aplatie, nous le réorganisons en une image avec les dimensions d'origine.

---

# Annexe : code 
---

L'instruction suivante :

```python
encoder.add(L.Conv2D(16, [2, 2], activation='relu', padding='same'))
```

utilise une couche de convolution pour extraire des caractéristiques locales de l'image. Voici une explication détaillée des différentes étapes de cette fonction :

### 1. Couches de convolution
- **`L.Conv2D(16, [2, 2], activation='relu', padding='same')`** : cette couche de convolution applique 16 filtres de taille 2x2 sur l'image d'entrée pour extraire des caractéristiques. L'activation `relu` est utilisée pour introduire la non-linéarité, tandis que le `padding='same'` garantit que la sortie conserve la même dimension que l'entrée.

- **`L.MaxPooling2D()`** : réduit la dimension de l'image en conservant les caractéristiques les plus importantes, ce qui rend l'encodeur plus efficace tout en réduisant la complexité.

### 2. Code latent
- **`L.Dense(code_size)`** : cette couche finale de l'encodeur réduit toutes les caractéristiques extraites à un vecteur de code latent de taille `code_size`.

### 3. Décodeur
- Le décodeur fait essentiellement l'inverse de l'encodeur, transformant le code latent en une reconstruction d'image. Les couches denses du décodeur reconstituent progressivement l'image originale, et `L.Reshape()` réorganise le vecteur en une image de la taille d'origine.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Passons maintenant à la **Partie 16 : Entraînement de l'autoencodeur avec différentes tailles de code**.

---

# Partie 16 : Entraînement de l'autoencodeur avec différentes tailles de code

## Description
Dans cette partie, nous testons différentes tailles de code latent pour l'autoencodeur profond afin de voir comment cela affecte les performances du modèle. Nous allons entraîner l'autoencodeur avec plusieurs tailles de code et vérifier les dimensions des couches pour chaque configuration.

## Code

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

## Justification

### 1. Utilisation de différentes tailles de code
- **`for code_size in [1, 8, 32, 128, 512, 1024]`** : tester différentes tailles de code permet d'explorer la capacité du modèle à capturer les informations pertinentes à différents niveaux de compression. Un code plus petit force le modèle à compresser davantage l'image, ce qui peut entraîner une perte d'informations. Un code plus grand conserve plus de détails.

### 2. Vérification des dimensions de l'encodeur et du décodeur
- **`assert encoder.output_shape[1:] == (code_size,)`** : cette ligne vérifie que l'encodeur produit bien un vecteur de la taille spécifiée par `code_size`.

- **`assert decoder.output_shape[1:] == img_shape`** : cette ligne vérifie que le décodeur reconstruit l'image avec les dimensions correctes après être passé par le code latent.

### 3. Validation des couches
- **`assert len(encoder.trainable_weights) >= 6`** : cette ligne vérifie que l'encodeur contient au moins 3 couches de neurones denses. De même pour le décodeur. Cela garantit que le modèle est suffisamment complexe pour apprendre à encoder et reconstruire efficacement les images.

---

# Annexe : code 
---

L'instruction suivante :

```python
for code_size in [1, 8, 32, 128, 512, 1024]:
    encoder, decoder = build_deep_autoencoder(img_shape, code_size=code_size)
```

sert à tester l'autoencodeur avec différentes tailles de code. Voici une explication détaillée de cette boucle :

### 1. Boucle pour tester différentes tailles de code
- **`for code_size in [1, 8, 32, 128, 512, 1024]`** : cette boucle teste l'autoencodeur en variant la taille du code latent. Chaque taille de code représente un niveau différent de compression de l'image. Tester avec plusieurs tailles permet de trouver la meilleure balance entre compression et qualité de reconstruction.

### 2. Validation des sorties
- **`assert encoder.output_shape[1:] == (code_size,)`** : après chaque test, cette ligne vérifie que l'encodeur produit effectivement un vecteur latent de la taille spécifiée.
  
- **`assert decoder.output_shape[1:] == img_shape`** : de la même manière, cette vérification garantit que le décodeur produit une image de la taille attendue après la reconstruction.

### 3. Validation des dimensions internes
- **`get_dim(layer)`** : cette fonction calcule le nombre d'unités de sortie de chaque couche, ce qui permet de vérifier que chaque couche du modèle respecte bien les contraintes de taille fixées par la taille du code latent.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Je vais maintenant continuer avec la **Partie 17 : Redéfinition de l'image et vérifications des tailles de code**.

---

# Partie 17 : Redéfinition de l'image et vérifications des tailles de code

## Description
Dans cette partie, nous redéfinissons la forme de l

'image pour les tests futurs et vérifions à nouveau la cohérence des dimensions des couches d'encodeur et de décodeur pour les différentes tailles de code.

## Code

```python
# Redéfinir la forme de l'image
img_shape = X.shape[1:]
IMG_SHAPE = X.shape[1:]

# Vérifier la forme des autoencodeurs avec différentes tailles de code
get_dim = lambda layer: np.prod(layer.output.shape[1:]) if hasattr(layer.output, 'shape') else np.prod(layer.output.get_shape().as_list()[1:])
```

## Justification

### 1. Redéfinition de la forme de l'image
- **`img_shape = X.shape[1:]`** : nous redéfinissons la forme de l'image pour nous assurer que les tests futurs utiliseront les bonnes dimensions des images. Cela est important car la taille de l'image affecte directement la manière dont l'autoencodeur apprend et reconstruit l'image.

### 2. Calcul des dimensions de sortie des couches
- **`get_dim(layer)`** : cette fonction lambda calcule les dimensions de sortie de chaque couche, ce qui permet de valider que les tailles de code et les dimensions des images reconstruites sont correctes.

---

# Annexe : code 
---

L'instruction suivante :

```python
img_shape = X.shape[1:]
```

sert à redéfinir la forme de l'image pour être utilisée dans les tests futurs. Voici une explication détaillée :

### 1. Redéfinir la forme de l'image
- **`img_shape = X.shape[1:]`** : cette ligne extrait les dimensions de l'image à partir des données d'entraînement (X). Cela garantit que nous utilisons la bonne taille d'image dans les prochaines étapes du processus.

### 2. Calcul des dimensions des couches
- **`get_dim(layer)`** : cette fonction lambda calcule les dimensions de sortie de chaque couche du modèle. Cela permet de valider que les dimensions des couches sont cohérentes, que ce soit pour l'encodeur ou pour le décodeur.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Je vais continuer avec la **Partie 18 : Vérification des tailles de code et tests**.
