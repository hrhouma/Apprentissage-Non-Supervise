Voici la suite avec la **Partie 23 : Visualisation des effets du bruit**.

---

# Partie 23 : Visualisation des effets du bruit

## Description
Dans cette partie, nous visualisons les effets du bruit gaussien appliqué aux images. Nous comparons l'image originale avec des versions de cette image contenant différents niveaux de bruit. Cette étape permet de mieux comprendre comment le bruit affecte les images et à quel point l'autoencodeur devra être robuste pour débruiter ces images.

## Code

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

## Justification

### 1. Visualisation de l'image originale
- **`plt.subplot(1, 4, 1)`** : cette ligne prépare une grille de 1 ligne et 4 colonnes pour afficher l'image originale dans la première colonne.

- **`plt.imshow(X[0])`** : affiche la première image du jeu de données d'entraînement sans aucun bruit ajouté.

### 2. Visualisation des images bruitées
- **`apply_gaussian_noise(X[:1], sigma=0.01)`** : cette ligne applique un bruit gaussien très léger (sigma=0.01) à l'image, et la fonction `imshow` affiche l'image bruitée dans la deuxième colonne.

- **`apply_gaussian_noise(X[:1], sigma=0.1)`** : cette ligne applique un bruit gaussien modéré (sigma=0.1) à l'image et l'affiche dans la troisième colonne.

- **`apply_gaussian_noise(X[:1], sigma=0.5)`** : cette ligne applique un bruit beaucoup plus fort (sigma=0.5) à l'image, simulant une perturbation importante, et l'image est affichée dans la quatrième colonne.

Cette comparaison permet de voir comment différents niveaux de bruit affectent les images et à quel point l'image est dégradée avec un bruit plus important.

---

# Annexe : code 
---

L'instruction suivante :

```python
plt.subplot(1, 4, 1)
plt.imshow(X[0])
```

sert à visualiser l'image originale et ses versions bruitées avec différents niveaux de bruit gaussien. Voici une explication détaillée de ce processus :

### 1. Grille de visualisation
- **`plt.subplot(1, 4, 1)`** : prépare un espace avec 4 colonnes pour afficher les différentes versions de l'image.

- **`plt.imshow(X[0])`** : affiche la première image du jeu de données, sans bruit ajouté.

### 2. Visualisation des images bruitées
- **`apply_gaussian_noise(X[:1], sigma=0.01)`** : cette fonction applique un bruit gaussien avec une faible intensité (`sigma=0.01`). Nous affichons cette image bruitée pour voir une version légèrement dégradée.

- **`apply_gaussian_noise(X[:1], sigma=0.1)`** : cette version applique un bruit plus fort, avec un `sigma` de 0.1, ce qui entraîne une dégradation plus visible de l'image.

- **`apply_gaussian_noise(X[:1], sigma=0.5)`** : cette version utilise un bruit encore plus intense (`sigma=0.5`), ce qui rend l'image beaucoup plus difficile à reconnaître.

Ces visualisations permettent de comprendre comment le bruit gaussien affecte les images et à quel point il devient difficile de débruiter les images lorsque le bruit est plus intense.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Passons maintenant à la **Partie 24 : Autoencodeur avec taille de code 512**.

---

# Partie 24 : Autoencodeur avec taille de code 512

## Description
Dans cette partie, nous construisons et entraînons un autoencodeur avec une taille de code plus grande (512). L'augmentation de la taille du code permet au modèle de capturer plus d'informations sur les images, ce qui peut améliorer la qualité des reconstructions, en particulier pour des tâches complexes comme le débruitage.

## Code

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

## Justification

### 1. Taille de code de 512
- **`build_deep_autoencoder(img_shape, code_size=512)`** : l'utilisation d'une taille de code de 512 permet au modèle de capturer plus d'informations sur les images, car le code latent contient plus d'unités. Cela est particulièrement utile lorsque le modèle doit traiter des images complexes ou lorsque nous voulons minimiser la perte d'informations pendant la compression.

### 2. Vérification de la taille du code
- **`assert encoder.output_shape[1:] == (512,)`** : cette assertion vérifie que l'encodeur produit bien un vecteur de taille 512, correspondant à la taille du code que nous avons spécifiée.

### 3. Création du modèle
- **`autoencoder = keras.models.Model(inp, reconstruction)`** : comme dans les parties précédentes, nous créons un modèle d'autoencodeur qui prend une image en entrée, la compresse en un code latent de taille 512, puis la reconstruit.

### 4. Compilation du modèle
- **`autoencoder.compile(optimizer='adamax', loss='mse')`** : le modèle est compilé avec l'optimiseur Adamax et la fonction de perte MSE. Ces choix sont adaptés à la tâche de reconstruction d'images.

---

# Annexe : code 
---

L'instruction suivante :

```python
encoder, decoder = build_deep_autoencoder(img_shape, code_size=512)
```

construit un autoencodeur avec une taille de code de 512. Voici une explication détaillée :

### 1. Taille de code 512
- **`code_size=512`** : cette taille de code plus grande permet au modèle de capturer davantage de détails de l'image. Un code plus grand contient plus d'unités, ce qui permet de représenter des informations plus complexes sur l'image.

### 2. Vérification de la sortie de l'encodeur
- **`assert encoder.output_shape[1:] == (512,)`** : cette assertion garantit que l'encodeur produit bien un vecteur latent de la taille correcte (512), ce qui permet de compresser les images en conservant un maximum de détails.

### 3. Création du modèle
- **`autoencoder = keras.models.Model(inp, reconstruction)`** : cette ligne définit le modèle complet en reliant l'entrée à la sortie, passant par l'encodeur et le décodeur.

### 4. Compilation
- **`autoencoder.compile(optimizer='adamax', loss='mse')`** : nous compilons le modèle en utilisant les mêmes paramètres d'optimisation que dans les parties précédentes, adaptés à cette tâche de reconstruction.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Je vais continuer avec la **Partie 25 : Entraînement avec bruit gaussien**.
