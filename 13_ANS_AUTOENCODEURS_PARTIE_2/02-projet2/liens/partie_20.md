Voici la suite avec la **Partie 20 : Entraînement final avec taille de code 32**.

---

# Partie 20 : Entraînement final avec taille de code 32

## Description
Dans cette partie, nous entraînons l'autoencodeur avec une taille de code fixe de 32 sur les données d'entraînement. Cet entraînement vise à minimiser l'erreur entre les images originales et leurs reconstructions, afin d'obtenir des reconstructions de haute qualité.

## Code

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

## Justification

### 1. Entraînement du modèle
- **`autoencoder.fit(...)`** : cette commande entraîne l'autoencodeur sur l'ensemble d'entraînement (`X_train`) pendant 32 époques. Pendant l'entraînement, le modèle essaie de minimiser la différence entre les images d'origine et leurs reconstructions.

### 2. Évaluation des performances
- **`autoencoder.evaluate(X_test, X_test)`** : après l'entraînement, nous évaluons les performances du modèle sur l'ensemble de test (`X_test`). L'erreur quadratique moyenne (MSE) est calculée pour mesurer la qualité des reconstructions produites par l'autoencodeur.

### 3. Visualisation des résultats
- **`visualize(img, encoder, decoder)`** : cette fonction affiche l'image originale, son code latent, et sa reconstruction pour cinq exemples tirés de l'ensemble de test. Cela permet d'évaluer visuellement la capacité du modèle à reconstruire les images à partir du code latent.

---

# Annexe : code 
---

L'instruction suivante :

```python
autoencoder.fit(x=X_train, y=X_train, epochs=32, validation_data=[X_test, X_test])
```

permet d'entraîner l'autoencodeur avec une taille de code de 32. Voici une explication détaillée :

### 1. Entraînement
- **`fit(x=X_train, y=X_train)`** : le modèle est entraîné à partir de l'ensemble d'entraînement. Les images d'origine (`X_train`) sont utilisées à la fois comme entrées et comme cibles, car l'objectif de l'autoencodeur est de reconstruire ces mêmes images.

### 2. Validation
- **`validation_data=[X_test, X_test]`** : pendant l'entraînement, le modèle est également évalué sur l'ensemble de test (`X_test`) pour surveiller les performances en dehors de l'ensemble d'entraînement et prévenir le surapprentissage.

### 3. Évaluation des performances
- **`evaluate(X_test, X_test)`** : après l'entraînement, cette ligne évalue les performances finales du modèle en calculant la perte (MSE) sur l'ensemble de test.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Passons maintenant à la **Partie 21 : Application de bruit gaussien**.

---

# Partie 21 : Application de bruit gaussien

## Description
Dans cette partie, nous ajoutons du bruit gaussien aux images d'entraînement pour entraîner un autoencodeur capable de débruiter les images. Le bruit gaussien est un bruit aléatoire ajouté à chaque pixel, ce qui simule des perturbations que le modèle devra apprendre à corriger.

## Code

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

## Justification

### 1. Bruit gaussien
- **`np.random.normal(0, sigma, X.shape[1:])`** : cette ligne génère du bruit gaussien à ajouter à chaque pixel de l'image. Le bruit gaussien suit une distribution normale centrée sur 0 avec un écart-type `sigma`. Ce type de bruit est couramment utilisé pour simuler des perturbations aléatoires dans les images.

### 2. Ajout du bruit
- **`noisy_image = X + noise`** : chaque pixel de l'image est perturbé par une petite quantité de bruit aléatoire, ce qui produit une version bruitée de l'image d'origine.

### 3. Limitation des valeurs des pixels
- **`np.clip(noisy_image, 0, 1)`** : après avoir ajouté le bruit, cette ligne limite les valeurs des pixels à l'intervalle [0, 1]. Cela garantit que les images restent dans la plage valide des valeurs de pixels (entre 0 et 1) après l'ajout du bruit.

---

# Annexe : code 
---

L'instruction suivante :

```python
def apply_gaussian_noise(X, sigma=0.1):
```

permet d'ajouter du bruit gaussien à une image. Voici une explication détaillée :

### 1. Génération de bruit gaussien
- **`np.random.normal(0, sigma, X.shape[1:])`** : cette ligne génère du bruit suivant une distribution normale avec un écart-type `sigma`. Le bruit est ajouté indépendamment à chaque pixel de l'image.

### 2. Ajout du bruit à l'image
- **`noisy_image = X + noise`** : chaque pixel de l'image est perturbé par une petite quantité de bruit. Cela simule des perturbations naturelles comme du bruit de capteur ou des interférences lors de la prise de vue d'une image.

### 3. Clipping des valeurs des pixels
- **`np.clip(noisy_image, 0, 1)`** : cette opération est utilisée pour limiter les valeurs des pixels afin qu'elles restent dans la plage valide [0, 1], car les valeurs des pixels doivent représenter des intensités lumineuses comprises entre 0 (noir) et 1 (blanc).

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Passons maintenant à la **Partie 22 : Sauvegarde des modèles**.

---

# Partie 22 : Sauvegarde des modèles

## Description
Dans cette partie, nous sauvegardons les modèles d'encodeur et de décodeur après l'entraînement. Cela permet de réutiliser les modèles plus tard sans avoir à les réentraîner. Nous sauvegardons les modèles au format `.keras` pour faciliter le chargement ultérieur.

## Code

```python
# Sauvegarder les modèles encodeur et décodeur entraînés
encoder.save("./encoder.keras")
decoder.save("./decoder.keras")
```

## Justification

### 1. Sauvegarde des modèles
- **`encoder.save("./encoder.keras")`** : cette commande sauvegarde le modèle d'encodeur dans un fichier avec l'extension `.keras`. Cela permet de conserver le modèle après l'entraînement pour une utilisation future sans avoir à le réentraîner.
  
- **`decoder.save("./decoder.keras")`** : de la même manière, cette ligne sauvegarde le modèle de décodeur.

Sauvegarder les modèles est une étape importante pour rendre l'entraînement réutilisable, surtout si le processus d'entraînement est long.

---

# Annexe : code 
---

L'instruction suivante :

```python
encoder.save("./encoder.keras")
```

permet de sauvegarder les modèles après leur entraînement. Voici une explication détaillée :

### 1. Sauvegarde des modèles
- **`encoder.save("./encoder.keras")`** : cette ligne sauvegarde le modèle d'encodeur sous forme de fichier `.keras`. Cela permet de charger le modèle plus tard pour faire des prédictions ou des évaluations supplémentaires sans avoir besoin de réentraîner le modèle à partir de zéro.

- **`decoder.save("./decoder.keras")`** : cette ligne fait de même pour le modèle de décodeur.

Sauvegarder les modèles permet de réutiliser facilement le modèle sans avoir à le réentraîner, ce qui est particulièrement utile pour les modèles complexes ou les entraînements longs.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Je vais continuer avec la **Partie 23 : Visualisation des effets du bruit**.
