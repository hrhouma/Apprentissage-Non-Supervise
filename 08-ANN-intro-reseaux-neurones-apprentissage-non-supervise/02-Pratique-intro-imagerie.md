# Analyse et Visualisation d'Images avec Python

# PARTIE 1 -  Introduction

Ce README fournit une explication détaillée de chaque section du code, avec un accent sur l'analyse et la visualisation d'images en utilisant des bibliothèques Python telles que Matplotlib, NumPy, et Scikit-Image. Le code illustre des techniques de base pour le chargement, la manipulation, et la visualisation d'images en utilisant plusieurs exemples célèbres comme 'Cameraman', 'Lena', et 'Mona Lisa'.

### Prérequis

Avant d'exécuter le code, assurez-vous d'avoir installé les bibliothèques Python suivantes :

- Matplotlib
- NumPy
- Pillow
- Scikit-Image

Vous pouvez les installer en utilisant pip :

```bash
pip install matplotlib numpy pillow scikit-image
```

<hr/>
<hr/>
<hr/>

### 1. Chargement et Affichage de l'Image Cameraman

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import urllib.request
from PIL import Image
import numpy as np
from io import BytesIO

# Charger l'image à partir de l'URL
url = 'https://github.com/scikit-image/scikit-image/raw/main/skimage/data/camera.png'
with urllib.request.urlopen(url) as response:
    img_data = response.read()

# Convertir les données en image
img = Image.open(BytesIO(img_data))
cameraman = np.array(img)

# Afficher l'image
plt.imshow(cameraman, cmap='gray')
plt.title("Image Cameraman")
plt.axis('off')
plt.show()
```

#### Explication
- **Objectif** : Ce bloc de code charge et affiche l'image 'Cameraman' à partir d'une URL.
- **Étapes** :
  1. L'image est téléchargée à partir de l'URL spécifiée.
  2. Elle est convertie en un tableau NumPy pour une manipulation plus facile.
  3. Enfin, l'image est affichée en niveaux de gris avec `plt.imshow`.

<hr/>
<hr/>
<hr/>

# PARTIE 2 - Affichage de l'Image Lena et Extraction des Valeurs de Pixels

```python
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
import matplotlib.patches as patches

# Charger l'image Lena
lena = data.astronaut()

# Afficher l'image
fig, ax = plt.subplots()
ax.imshow(lena)
plt.title("Image Lena")
plt.axis('off')

# Définir les coordonnées et la taille du rectangle
rect = patches.Rectangle((0, 0), 10, 10, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()

# Extraire les valeurs des pixels de la partie supérieure gauche de l'image
R = lena[0:10, 0:10, 0]
G = lena[0:10, 0:10, 1]
B = lena[0:10, 0:10, 2]

# Afficher les valeurs des pixels
print("Pixels de la partie supérieure gauche de l'image :")
print("R (Rouge) :\n", R)
print("G (Vert) :\n", G)
print("B (Bleu) :\n", B)
```

#### Explication
- **Objectif** : Afficher l'image 'Lena', délimiter une région spécifique, et extraire les valeurs de pixels RGB de cette région.
- **Étapes** :
  1. L'image 'Lena' est chargée en utilisant `data.astronaut()` de Scikit-Image.
  2. Un rectangle est dessiné sur l'image pour indiquer la région d'intérêt.
  3. Les valeurs des canaux Rouge, Vert, et Bleu des pixels de la région supérieure gauche sont extraites et affichées.


<hr/>
<hr/>
<hr/>

# PARTIE 3 -  Visualisation des Canaux RGB de l'Image Mona Lisa

```python
import matplotlib.pyplot as plt
from skimage import io

# Charger l'image Monalisa
monalisa = io.imread('https://upload.wikimedia.org/wikipedia/commons/6/6a/Mona_Lisa.jpg')

# Afficher les canaux RGB
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['Reds', 'Greens', 'Blues']
for i, ax in enumerate(axes):
    ax.imshow(monalisa[:, :, i], cmap=colors[i])
    ax.set_title(f'Canal {colors[i]}')
    ax.axis('off')
plt.show()
```

#### Explication
- **Objectif** : Charger et afficher les canaux Rouge, Vert, et Bleu de l'image 'Mona Lisa'.
- **Étapes** :
  1. L'image est chargée à partir de l'URL donnée.
  2. Les trois canaux sont séparément affichés en utilisant des cartes de couleurs correspondant à chaque canal.


<hr/>
<hr/>
<hr/>

# PARTIE 4 - Conversion de l'Image Lena en Niveaux de Gris

```python
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray

# Charger l'image Lena en couleur
lena = data.astronaut()

# Convertir l'image Lena en échelle de gris
lena_gray = rgb2gray(lena)

# Afficher l'image en échelle de gris
plt.imshow(lena_gray, cmap='gray')
plt.title("Image Lena en échelle de gris")
plt.axis('off')
plt.show()

# Afficher les valeurs des pixels de la partie supérieure gauche de l'image en échelle de gris
print("Valeurs des pixels de la partie supérieure gauche en échelle de gris :\n", lena_gray[0:10, 0:10])
```

#### Explication
- **Objectif** : Convertir et afficher l'image 'Lena' en niveaux de gris.
- **Étapes** :
  1. L'image en couleur est convertie en niveaux de gris.
  2. L'image convertie est affichée, et les valeurs de luminance des pixels sont extraites.

<hr/>
<hr/>
<hr/>

# PARTIE 5 - Quantification des Niveaux de Gris

```python
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, color, img_as_ubyte

# Charger l'image Lena
image = data.astronaut()

# Convertir l'image en échelle de gris
gray_image = color.rgb2gray(image)
gray_image = img_as_ubyte(gray_image)

# Fonction pour quantifier les niveaux de gris
def quantize_gray_levels(image, levels):
    return np.round(image * (levels - 1) / 255) * (255 / (levels - 1))

# Liste des niveaux de gris à tester
gray_levels = [2, 5, 10, 50, 256]

# Créer une figure pour afficher les résultats
fig, axes = plt.subplots(1, len(gray_levels), figsize=(20, 5))
for i, level in enumerate(gray_levels):
    quantized_image = quantize_gray_levels(gray_image, level)
    axes[i].imshow(quantized_image, cmap='gray')
    axes[i].set_title(f'{level} niveaux')
    axes[i].axis('off')
plt.suptitle('Niveaux de gris')
plt.show()
```

#### Explication
- **Objectif** : Montrer l'effet de la quantification des niveaux de gris sur une image en niveaux de gris.
- **Étapes** :
  1. L'image en niveaux de gris est quantifiée en différents niveaux (de 2 à 256).
  2. Chaque image quantifiée est affichée côte à côte pour comparer les effets de la réduction des niveaux de gris.

<hr/>
<hr/>
<hr/>

### Conclusion

Ce code vous permet de comprendre les concepts de base de la manipulation et de la visualisation d'images avec Python. Il couvre des aspects comme le chargement des images, la conversion en niveaux de gris, l'extraction des valeurs de pixels, et la quantification des niveaux de gris. Les exemples fournis utilisent des images classiques pour illustrer ces concepts de manière concrète et pratique.

<hr/>
<hr/>
<hr/>

# Annexe : bibliothèques `numpy`, `Pillow`, et `scikit-image` :

| **Bibliothèque** | **Description** | **Fonctionnalités Principales** | **Utilisation Typique** | **Exemple de Code** |
|------------------|-----------------|-------------------------------|------------------------|---------------------|
| **NumPy**        | `NumPy` est une bibliothèque de calcul scientifique en Python. Elle fournit un puissant objet `ndarray` pour manipuler des tableaux multidimensionnels ainsi qu'une collection de fonctions mathématiques pour effectuer des opérations sur ces tableaux. | - Manipulation de tableaux multidimensionnels (`ndarray`).<br>- Opérations mathématiques avancées (algèbre linéaire, statistiques, etc.).<br>- Manipulation et transformation de données. | - Calculs numériques.<br>- Traitement de données sous forme de matrices.<br>- Préparation de données pour l'apprentissage automatique et d'autres analyses. | **Exemple de Code**:<br> import numpy as np<br> Création d'un tableau 1D:<br> array = np.array([1, 2, 3, 4, 5])<br> Création d'un tableau 2D:<br> matrix = np.array([[1, 2], [3, 4]])<br> Calcul de la somme des éléments:<br> sum_array = np.sum(array) |
| **Pillow**       | `Pillow` est une bibliothèque de traitement d'images en Python. C'est une version améliorée de l'ancienne bibliothèque `PIL` (Python Imaging Library). Elle permet de créer, modifier et enregistrer des images dans divers formats. | - Chargement et sauvegarde d'images dans divers formats (JPEG, PNG, GIF, etc.).<br>- Redimensionnement, recadrage, rotation et transformation des images.<br>- Application de filtres et d'effets aux images.<br>- Manipulation des pixels et ajustements de couleur. | - Traitement d'images pour les applications web ou desktop.<br>- Préparation d'images pour l'analyse et la reconnaissance d'images.<br>- Création d'images à la volée dans des scripts automatisés. | **Exemple de Code**:<br> from PIL import Image<br> Charger une image:<br> image = Image.open('path_to_your_image.jpg')<br> Redimensionner l'image:<br> image_resized = image.resize((100, 100))<br> Sauvegarder l'image redimensionnée:<br> image_resized.save('resized_image.jpg') |
| **scikit-image** | `scikit-image` est une bibliothèque dédiée au traitement d'images. Elle offre une collection d'algorithmes pour l'analyse, la transformation, et la manipulation des images, construite sur `numpy`. | - Chargement et traitement d'images.<br>- Algorithmes pour le filtrage, la transformation, la segmentation, et la détection d'objets.<br>- Calcul de mesures d'image (histogrammes, gradients, etc.).<br>- Prise en charge d'images en niveaux de gris, RVB, et images multi-canal. | - Analyse d'images pour la reconnaissance de formes, la segmentation, et la détection d'objets.<br>- Prétraitement d'images pour les pipelines d'apprentissage automatique.<br>- Création de pipelines de traitement d'images personnalisés. | **Exemple de Code**:<br> from skimage import io, filters<br> Charger une image:<br> image = io.imread('path_to_your_image.jpg')<br> Appliquer un filtre gaussien:<br> image_filtered = filters.gaussian(image, sigma=1.0)<br> Sauvegarder l'image filtrée:<br> io.imsave('filtered_image.jpg', image_filtered) |


<hr/>
<hr/>
<hr/>

### Explications supplémentaires
- **NumPy** est fondamental pour toute forme de calcul scientifique en Python. Il est souvent utilisé en conjonction avec d'autres bibliothèques comme `Pandas` pour l'analyse de données et `Matplotlib` pour la visualisation.
- **Pillow** simplifie le traitement d'images, rendant des tâches comme le redimensionnement, le filtrage et l'ajustement des couleurs accessibles et faciles à réaliser.
- **scikit-image** offre des outils spécialisés pour le traitement d'images plus avancé, souvent utilisé dans les domaines de la vision par ordinateur et de la reconnaissance d'images.

Ces bibliothèques sont souvent utilisées ensemble dans des projets où le traitement d'images et l'analyse de données sont nécessaires.


<hr/>
<hr/>
<hr/>


- Pour les bibliothèques `numpy`, `Pillow`, et `scikit-image` dans un projet Python, voici comment vous pouvez les installer et les importer dans votre code.

### Étape 1: Installation des bibliothèques
- Vous pouvez les installer en utilisant `pip`. Ouvrez un terminal et exécutez les commandes suivantes :

```bash
pip install numpy pillow scikit-image
```

### Étape 2: Importation des bibliothèques
Une fois les bibliothèques installées, vous pouvez les importer dans votre script Python comme suit :

```python
import numpy as np
from PIL import Image
from skimage import io
```

### Explication

- **Numpy** (`numpy`): Une bibliothèque essentielle pour les calculs numériques en Python, particulièrement utile pour manipuler des tableaux multidimensionnels (ndarrays).
  
- **Pillow** (`PIL` ou `Pillow`): Une bibliothèque pour le traitement d'images. `Pillow` est une version améliorée de l'ancienne bibliothèque `PIL` (Python Imaging Library).

- **Scikit-image** (`skimage`): Une collection d'algorithmes pour le traitement des images, basée sur `numpy`.

### Exemple d'utilisation

Voici un exemple simple qui montre comment charger et afficher une image en utilisant ces bibliothèques :

```python
import numpy as np
from PIL import Image
from skimage import io

# Charger une image en utilisant Pillow
image_pillow = Image.open('path_to_your_image.jpg')
image_pillow.show()

# Charger une image en utilisant scikit-image
image_skimage = io.imread('path_to_your_image.jpg')

# Convertir l'image en tableau numpy
image_np = np.array(image_pillow)

# Afficher les dimensions de l'image
print(f'Dimensions de l\'image: {image_np.shape}')
```

### Explications supplémentaires
- `Image.open()` de Pillow charge une image depuis le chemin spécifié et la retourne sous forme d'objet image.
- `io.imread()` de scikit-image lit une image et la retourne sous forme de tableau `numpy`.
- `np.array()` convertit un objet image en un tableau `numpy` pour un traitement numérique plus avancé.

- N'oubliez pas de remplacer `'path_to_your_image.jpg'` par le chemin réel de votre image.
- Cela devrait vous permettre de commencer à travailler avec ces bibliothèques dans vos projets de traitement d'images.
