### Cours d'Introduction à l'Imagerie Numérique 

#### Table des Matières

1. [Introduction à l’Imagerie Numérique](#introduction-a-l-imagerie-numerique)
   - [Qu’est-ce qu’une image ?](#quest-ce-quune-image)
   - [Application de la vision par ordinateur](#application-de-la-vision-par-ordinateur)
2. [Numérisation d’une Image](#numerisation-dune-image)
   - [Représentation matricielle des images](#representation-matricielle-des-images)
   - [Manipulation de matrices en Python](#manipulation-de-matrices-en-python)
3. [Types d’Images et Espaces de Couleurs](#types-dimages-et-espaces-de-couleurs)
   - [Images binaires, niveaux de gris, et couleur](#images-binaires-niveaux-de-gris-et-couleur)
   - [Espaces de couleurs : RGB et HSV](#espaces-de-couleurs-rgb-et-hsv)
4. [Problèmes de Bruit et Dimensionalité des Images](#problemes-de-bruit-et-dimensionalite-des-images)
   - [Nature et origine du bruit dans les images](#nature-et-origine-du-bruit-dans-les-images)
   - [Problème de la dimensionnalité](#probleme-de-la-dimensionalite)
5. [Traitement d’Images : Bas, Moyen et Haut Niveau](#traitement-dimages-bas-moyen-et-haut-niveau)
   - [Extraction des caractéristiques (Bas niveau)](#extraction-des-caracteristiques-bas-niveau)
   - [Transformation en descripteurs (Moyen niveau)](#transformation-en-descripteurs-moyen-niveau)
   - [Interprétation et compréhension des scènes (Haut niveau)](#interpretation-et-comprehension-des-scenes-haut-niveau)
6. [Manipulation des Images en Python](#manipulation-des-images-en-python)
   - [Lire, écrire et afficher des images](#lire-ecrire-et-afficher-des-images)
   - [Conversion d'espaces de couleurs](#conversion-despaces-de-couleurs)
   - [Gestion des formats d'images et compression](#gestion-des-formats-dimages-et-compression)
7. [Conclusion et Perspectives](#conclusion-et-perspectives)

---

### 1. Introduction à l’Imagerie Numérique

#### Qu’est-ce qu’une image ?
Une image est une représentation visuelle des objets réels, stockée sous forme numérique pour être traitée par des systèmes informatiques. Les images numériques jouent un rôle crucial dans la vision par ordinateur, qui vise à créer des systèmes capables de comprendre et d’interpréter ces images. Des applications courantes incluent la reconnaissance faciale, la détection d'empreintes digitales, et la surveillance vidéo.

[Retour en haut](#cours-dintroduction-a-limagerie-numerique-pour-debutants)

---

### 2. Numérisation d’une Image

#### Représentation matricielle des images
Les images numériques sont composées de pixels organisés en une matrice de lignes et de colonnes. Chaque pixel possède une valeur qui représente son intensité lumineuse, généralement comprise entre 0 (noir) et 255 (blanc).

En Python, avec la bibliothèque NumPy, nous pouvons manipuler des images sous forme de matrices.

```python
import numpy as np

# Créer une matrice 3x3 représentant une image
I = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Accéder à la deuxième ligne
ligne2 = I[1, :]
print(ligne2)

# Modifier une colonne entière
I[:, 1] = 0
print(I)
```

[Retour en haut](#cours-dintroduction-a-limagerie-numerique-pour-debutants)

---

### 3. Types d’Images et Espaces de Couleurs

#### Images binaires, niveaux de gris, et couleur
Les images peuvent être classifiées selon le type d'information qu'elles contiennent :
- **Images binaires** : Pixels à deux niveaux (0 ou 1), utilisées principalement pour la segmentation ou la détection de formes.
- **Images en niveaux de gris** : Pixels avec des valeurs entre 0 et 255, représentant différentes intensités de lumière.
- **Images couleur** : Composées de plusieurs canaux de couleurs (comme le RGB).

[Retour en haut](#cours-dintroduction-a-limagerie-numerique-pour-debutants)

---

### 4. Problèmes de Bruit et Dimensionalité des Images

#### Nature et origine du bruit dans les images
Le bruit dans les images est une perturbation qui peut altérer la qualité visuelle et rendre l’analyse difficile. Il peut être introduit par des facteurs tels que des capteurs défectueux, des interférences électriques, ou des méthodes de compression.

Voici comment générer du bruit sur une image en niveaux de gris :

```python
import numpy as np
import matplotlib.pyplot as plt

# Création d'une image en niveaux de gris
image = np.full((100, 100), 128, dtype=np.uint8)

# Ajout de bruit
bruit = np.random.normal(0, 20, (100, 100))
image_bruitee = np.clip(image + bruit, 0, 255).astype(np.uint8)

plt.subplot(1, 2, 1)
plt.title('Image originale')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Image avec bruit')
plt.imshow(image_bruitee, cmap='gray')
plt.show()
```

[Retour en haut](#cours-dintroduction-a-limagerie-numerique-pour-debutants)

---

### 5. Traitement d’Images : Bas, Moyen et Haut Niveau

#### Extraction des caractéristiques (Bas niveau)
Le traitement d'images commence par l'extraction de caractéristiques à partir de l'image brute. Ces caractéristiques peuvent être des contours, des textures, ou des gradients de luminosité.

[Retour en haut](#cours-dintroduction-a-limagerie-numerique-pour-debutants)

---

### 6. Manipulation des Images en Python

#### Lire, écrire et afficher des images
Python permet de manipuler facilement des images avec des bibliothèques comme PIL (Pillow) et OpenCV.

```python
from PIL import Image
import matplotlib.pyplot as plt

# Lire une image depuis un fichier
I = Image.open('cameraman.tif')

# Afficher l'image
plt.imshow(I, cmap='gray')
plt.axis('off')
plt.show()

# Sauvegarder l'image dans un autre format
I.save('camera1.png')
```

[Retour en haut](#cours-dintroduction-a-limagerie-numerique-pour-debutants)

---

### 7. Conclusion et Perspectives

Ce cours vous a introduit aux concepts de base de l'imagerie numérique et au traitement d'images en Python. En comprenant les types d'images, les espaces de couleurs, les défis du bruit, et les différents niveaux de traitement, vous êtes maintenant mieux équipés pour explorer des applications plus avancées en vision par ordinateur.

[Retour en haut](#cours-dintroduction-a-limagerie-numerique-pour-debutants)
