Voici la suite avec la **Partie 27 : Sauvegarde des modèles entraînés**.

---

# Partie 27 : Sauvegarde des modèles entraînés

## Description
Dans cette partie, nous sauvegardons les modèles d'encodeur et de décodeur après l'entraînement au débruitage. Cela permet de réutiliser les modèles plus tard sans avoir à les réentraîner. Sauvegarder les modèles est une étape importante pour pouvoir les recharger et les utiliser à tout moment.

## Code

```python
# Sauvegarder les modèles encodeur et décodeur entraînés
encoder.save("./encoder_denoising.keras")
decoder.save("./decoder_denoising.keras")
```

## Justification

### 1. Sauvegarde des modèles
- **`encoder.save("./encoder_denoising.keras")`** : cette ligne sauvegarde le modèle d'encodeur qui a été entraîné au débruitage. Le modèle est sauvegardé dans un fichier `.keras` qui peut être rechargé ultérieurement sans avoir à réentraîner le modèle.

- **`decoder.save("./decoder_denoising.keras")`** : de la même manière, cette ligne sauvegarde le modèle de décodeur, permettant de reconstruire les images à partir des codes latents générés par l'encodeur.

La sauvegarde des modèles est une étape importante pour préserver les résultats de l'entraînement et les réutiliser dans d'autres contextes, comme la production ou l'évaluation.

---

# Annexe : code 
---

L'instruction suivante :

```python
encoder.save("./encoder_denoising.keras")
```

permet de sauvegarder les modèles d'encodeur et de décodeur après leur entraînement au débruitage. Voici une explication détaillée :

### 1. Sauvegarde de l'encodeur
- **`encoder.save("./encoder_denoising.keras")`** : cette commande sauvegarde l'encodeur dans un fichier `.keras`, ce qui permet de le réutiliser sans avoir à le réentraîner à chaque fois. Sauvegarder le modèle permet de gagner du temps et des ressources si l'entraînement prend beaucoup de temps.

### 2. Sauvegarde du décodeur
- **`decoder.save("./decoder_denoising.keras")`** : cette ligne fait de même pour le décodeur, permettant de sauvegarder le processus de reconstruction des images à partir des codes latents.

Ces sauvegardes sont essentielles pour déployer les modèles ou les réutiliser dans des contextes futurs.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Passons maintenant à la **Partie 28 : Utilisation de NearestNeighbors**.

---

# Partie 28 : Utilisation de NearestNeighbors

## Description
Dans cette partie, nous utilisons l'algorithme `NearestNeighbors` pour trouver les images similaires dans l'espace latent du code généré par l'encodeur. Cette technique permet de voir à quel point les images qui se ressemblent dans l'espace visuel sont proches dans l'espace latent.

## Code

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
show_similar(X_test[2])
show_similar(X_test[500])
show_similar(X_test[66])
```

## Justification

### 1. Extraction des codes latents
- **`encoder.predict(X_train)`** : nous utilisons l'encodeur pour convertir chaque image d'entraînement en un vecteur de code latent. Cela nous permet de travailler dans un espace de caractéristiques réduit.

### 2. Recherche des plus proches voisins
- **`NearestNeighbors(n_neighbors=50)`** : cette commande utilise l'algorithme `NearestNeighbors` pour trouver les images dont les vecteurs latents sont les plus proches dans l'espace latent. Plus deux images sont proches dans cet espace, plus elles sont similaires.

### 3. Comparaison des images
- **`get_similar()`** : cette fonction permet de trouver les images les plus proches de l'image donnée dans l'espace latent. Elle renvoie également les distances entre l'image donnée et ses voisines.

### 4. Visualisation des images similaires
- **`show_similar()`** : cette fonction affiche l'image originale et ses voisines les plus proches. Cela permet de voir visuellement quelles images se ressemblent dans l'espace latent.

---

# Annexe : code 
---

L'instruction suivante :

```python
nn = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(codes)
```

permet de trouver les images similaires dans l'espace latent. Voici une explication détaillée de ce processus :

### 1. Extraction des codes latents
- **`encoder.predict(X_train)`** : nous utilisons l'encodeur pour transformer les images d'entraînement en vecteurs latents, ce qui réduit leur dimension et permet de comparer les images dans un espace de caractéristiques réduit.

### 2. Recherche des plus proches voisins
- **`NearestNeighbors(n_neighbors=50, algorithm='ball_tree')`** : cette commande utilise l'algorithme `NearestNeighbors` pour trouver les 50 images les plus proches dans l'espace latent. L'algorithme `ball_tree` est utilisé ici car il est bien adapté aux grands ensembles de données.

### 3. Visualisation des voisins
- **`show_similar()`** : cette fonction affiche l'image originale ainsi que les images qui lui ressemblent le plus dans l'espace latent, en affichant les distances entre elles.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Je vais continuer avec la **Partie 29 : Interpolation entre deux images**.
