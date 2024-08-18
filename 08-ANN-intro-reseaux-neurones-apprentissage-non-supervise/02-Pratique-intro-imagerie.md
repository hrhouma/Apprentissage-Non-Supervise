# Analyse et Visualisation d'Images avec Python

# 1 - Introduction

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

# 2. Affichage de l'Image Lena et Extraction des Valeurs de Pixels

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

# 3. Visualisation des Canaux RGB de l'Image Mona Lisa

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

# 4. Conversion de l'Image Lena en Niveaux de Gris

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

# 5. Quantification des Niveaux de Gris

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
