Voici la suite avec la **Partie 11 : Entraînement de l'autoencodeur**.

---

# Partie 11 : Entraînement de l'autoencodeur

## Description
Dans cette partie, nous compilons et entraînons l'autoencodeur. L'objectif est d'apprendre à l'autoencodeur à encoder et à reconstruire les images à partir de leur représentation compacte (code latent). Nous utilisons la fonction de perte MSE (Mean Squared Error) pour mesurer l'erreur entre les images reconstruites et les images d'origine.

## Code

```python
encoder, decoder = build_pca_autoencoder(img_shape, code_size=32)

# Définir le modèle autoencodeur
inp = L.Input(img_shape)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inp, reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')
```

## Justification

### 1. Compilation du modèle
- **`autoencoder.compile(optimizer='adamax', loss='mse')`** : nous compilons l'autoencodeur en utilisant l'optimiseur Adamax, qui est une variante de l'algorithme d'optimisation Adam. La fonction de perte utilisée ici est l'erreur quadratique moyenne (MSE). L'objectif est de minimiser cette erreur pendant l'entraînement, de manière à ce que les images reconstruites ressemblent le plus possible aux images originales.

### 2. Définition du modèle d'autoencodeur
- **`keras.models.Model(inp, reconstruction)`** : cette ligne crée un modèle Keras qui relie l'entrée de l'image (`inp`) à la reconstruction finale générée par le décodeur. Cela permet d'entraîner l'encodeur et le décodeur ensemble.

### 3. Création du modèle
- **`encoder(inp)`** : l'entrée est d'abord encodée en une représentation compacte par l'encodeur.
  
- **`decoder(code)`** : ensuite, le décodeur reconstruit l'image à partir de cette représentation compacte.

---

# Annexe : code 
---

L'instruction suivante :

```python
autoencoder.compile(optimizer='adamax', loss='mse')
```

permet de compiler le modèle d'autoencodeur en utilisant un optimiseur et une fonction de perte. Voici une explication détaillée de ce processus :

### 1. Optimiseur Adamax
- **`optimizer='adamax'`** : Adamax est une variante de l'optimiseur Adam. C'est un optimiseur basé sur la descente de gradient stochastique qui est particulièrement efficace pour des réseaux profonds comme les autoencodeurs.

### 2. Fonction de perte MSE
- **`loss='mse'`** : MSE, ou Mean Squared Error, est une fonction de perte couramment utilisée dans les tâches de reconstruction d'images. Elle calcule la moyenne des carrés des différences entre les pixels des images originales et celles reconstruites. L'objectif de l'autoencodeur est de minimiser cette perte pour produire des reconstructions de haute qualité.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Voici maintenant la **Partie 12 : Entraînement de l'autoencodeur**.

---

# Partie 12 : Entraînement de l'autoencodeur

## Description
Dans cette partie, nous entraînons l'autoencodeur sur les données d'entraînement. L'entraînement consiste à ajuster les poids du modèle pour minimiser la perte, ce qui améliore la qualité des reconstructions d'images au fil des époques. Nous utilisons ici 32 époques d'entraînement.

## Code

```python
autoencoder.fit(x=X_train, y=X_train, epochs=32, validation_data=[X_test, X_test])
```

## Justification

### 1. Entraînement de l'autoencodeur
- **`autoencoder.fit(x=X_train, y=X_train, epochs=32)`** : cette ligne entraîne l'autoencodeur sur l'ensemble de données d'entraînement (`X_train`). Le modèle essaie de minimiser la différence entre les images d'entrée (`X_train`) et leurs reconstructions (les prédictions du modèle). L'entraînement se fait pendant 32 époques.

### 2. Validation
- **`validation_data=[X_test, X_test]`** : cette ligne indique que nous voulons valider le modèle sur un ensemble de données de test à chaque époque. Le modèle est évalué sur ces données, mais sans ajuster les poids. Cela permet de surveiller les performances du modèle et de s'assurer qu'il ne surapprend pas (surapprentissage).

---

# Annexe : code 
---

L'instruction suivante :

```python
autoencoder.fit(x=X_train, y=X_train, epochs=32, validation_data=[X_test, X_test])
```

sert à entraîner l'autoencodeur. Voici une explication détaillée de cette commande :

### 1. `fit`
- **`fit`** est la méthode utilisée pour entraîner le modèle. Elle ajuste les poids du réseau de neurones pour minimiser la perte définie (ici, MSE) en ajustant les paramètres internes du modèle.

### 2. Données d'entraînement
- **`x=X_train, y=X_train`** : le modèle prend les images d'entrée `X_train` et essaie de reconstruire ces mêmes images (`y=X_train`). L'objectif est de produire une reconstruction aussi proche que possible de l'image d'origine.

### 3. Validation
- **`validation_data=[X_test, X_test]`** : les données de test `X_test` sont utilisées pour évaluer les performances du modèle après chaque époque d'entraînement. Cela permet de surveiller si le modèle généralise bien sur des données qu'il n'a jamais vues.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Voici maintenant la **Partie 13 : Visualisation de la reconstruction**.

---

# Partie 13 : Visualisation de la reconstruction

## Description
Dans cette partie, nous visualisons les résultats de l'autoencodeur en comparant les images originales avec leurs reconstructions. Cette étape permet d'observer visuellement comment le modèle reconstruit les images à partir du code latent.

## Code

```python
def visualize(img, encoder, decoder):
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2, -1]))

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    plt.imshow(reco.clip(0, 1))
    plt.show()
```

## Justification

### 1. Visualisation de l'image originale
- **`plt.subplot(1, 3, 1)`** : cette ligne crée un subplot pour afficher l'image originale. Nous utilisons ici la fonction `plt.imshow(img)` pour afficher l'image dans la première position de la grille.

### 2. Visualisation du code latent
- **`code = encoder.predict(img[None])[0]`** : l'image est encodée en un vecteur de code latent à l'aide de l'encodeur.
  
- **`plt.subplot(1, 3, 2)`** : nous affichons le code latent sous forme de matrice pour avoir une idée de la manière dont le modèle compresse l'image.

### 3. Visualisation de la reconstruction
- **`reco = decoder.predict(code[None])[0]`** : le décodeur reconstruit l'image à partir du code latent.

- **`plt.subplot(1, 3, 3)`** : nous affichons l'image reconstruite dans la troisième position de la grille.

---

# Annexe : code 
---

L'instruction suivante :

```python
def visualize(img, encoder, decoder):
```

sert à visualiser l'image d'origine, son code latent et sa reconstruction. Voici une explication détaillée :

### 1. Encodage de l'image
- **`encoder.predict(img[None])[0]`** : cette ligne prend l'image et la passe à travers l'encodeur pour obtenir le vecteur de code latent. Le `None` ici ajoute une dimension pour que l'entrée corresponde au format attendu par le modèle.

### 2. Reconstruction de l'image
- **`decoder.predict(code[None])[0]`** : à partir du code latent, le décodeur reconstruit l'image.

### 3. Affichage
- **`plt.imshow(...)`** est utilisé trois fois pour afficher respectivement l'image originale, le code latent et l'image reconstruite.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Je continue avec la **Partie 14 : Évaluation de la performance**.

---

# Partie 14 : Évaluation de la performance

## Description
Dans cette partie, nous évaluons la qualité des reconstructions produites par l'autoencodeur en utilisant l'ensemble de test. Nous mesurons l'erreur quadratique moyenne (MSE) entre les images reconstruites et les images originales.

## Code

```python
score = autoencoder.evaluate(X_test, X_test, verbose=0)
print("Final MSE:", score)

for i in range(5):
    img = X

_test[i]
    visualize(img, encoder, decoder)
```

## Justification

### 1. Évaluation du modèle
- **`autoencoder.evaluate(X_test, X_test, verbose=0)`** : cette commande évalue le modèle sur l'ensemble de test. L'autoencodeur essaie de reconstruire les images de test, et la fonction retourne l'erreur quadratique moyenne (MSE) entre les reconstructions et les images originales.

### 2. Affichage des résultats
- **`for i in range(5)`** : cette boucle permet de visualiser cinq images de test ainsi que leurs reconstructions, afin d'observer visuellement les performances du modèle.

---

# Annexe : code 
---

L'instruction suivante :

```python
score = autoencoder.evaluate(X_test, X_test, verbose=0)
print("Final MSE:", score)
```

permet d'évaluer les performances de l'autoencodeur. Voici une explication détaillée :

### 1. Évaluation du modèle
- **`autoencoder.evaluate(X_test, X_test)`** : cette commande calcule la perte (MSE) entre les images d'origine (`X_test`) et leurs reconstructions. Cela permet de quantifier l'erreur moyenne par pixel entre les images originales et les images reconstruites.

### 2. Visualisation des résultats
- **`visualize(img, encoder, decoder)`** : cette fonction affiche les images originales, les codes latents et les reconstructions pour vérifier visuellement la qualité des reconstructions.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Je vais continuer avec les autres parties dans ce même format.
