<a id="cours-imagerie"></a>

### Introduction à l'Imagerie Numérique 

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

# Remarque Générale :

- *Pour les images et certains éléments, veuillez consulter les références situées en bas du document :* [Introduction à l'Imagerie Numérique](#references)

<a id="introduction-a-l-imagerie-numerique"></a>

### 1. Introduction à l’Imagerie Numérique
Une image est une représentation visuelle des objets réels, stockée sous forme numérique pour être traitée par des systèmes informatiques. Les images numériques jouent un rôle crucial dans la vision par ordinateur, qui vise à créer des systèmes capables de comprendre et d’interpréter ces images. Des applications courantes incluent la reconnaissance faciale, la détection d'empreintes digitales, et la surveillance vidéo.

[Retour en haut](#cours-imagerie)

---

<a id="numerisation-dune-image"></a>

### 2. Numérisation d’une Image
Les images numériques sont composées de pixels organisés en une matrice de lignes et de colonnes. Chaque pixel possède une valeur qui représente son intensité lumineuse, généralement comprise entre 0 (noir) et 255 (blanc).




En Python, avec la bibliothèque NumPy, nous pouvons manipuler des images sous forme de matrices.

```python
import numpy as np
import matplotlib.pyplot as plt

# Créer une matrice 3x3 représentant une image
I = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Afficher la matrice originale comme une image en niveaux de gris
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)  # Première sous-figure (1 ligne, 2 colonnes, 1ère position)
plt.imshow(I, cmap='gray')
plt.title("Avant la modification")
plt.colorbar()
plt.axis('off')

# Accéder à la deuxième ligne
ligne2 = I[1, :]
print("Deuxième ligne :")
print(ligne2)

# Modifier une colonne entière
I[:, 1] = 0
print("Matrice modifiée :")
print(I)

# Afficher la matrice modifiée comme une image en niveaux de gris
plt.subplot(1, 2, 2)  # Deuxième sous-figure (1 ligne, 2 colonnes, 2ème position)
plt.imshow(I, cmap='gray')
plt.title("Après la modification")
plt.colorbar()
plt.axis('off')

# Afficher les deux images côte à côte
plt.show()

```

[Retour en haut](#cours-imagerie)

---

<a id="types-dimages-et-espaces-de-couleurs"></a>

### 3. Types d’Images et Espaces de Couleurs

#### Images binaires, niveaux de gris, et couleur
Les images peuvent être classifiées selon le type d'information qu'elles contiennent :
- **Images binaires** : Pixels à deux niveaux (0 ou 1), utilisées principalement pour la segmentation ou la détection de formes.
- **Images en niveaux de gris** : Pixels avec des valeurs entre 0 et 255, représentant différentes intensités de lumière.
- **Images couleur** : Composées de plusieurs canaux de couleurs (comme le RGB).

## Représentation d’une image digitale (niveaux de gris)

![image](https://github.com/user-attachments/assets/6309f9a7-4b7d-4840-9474-a355fb8d5970)

Cette image *ci-haut* illustre la structure d'une image numérique en niveaux de gris, décomposée en pixels. Elle montre :

- **Colonnes et lignes de pixels** : L'image est constituée de lignes (horizontalement) et de colonnes (verticalement) de pixels.
- **Zoom sur une région** : Une petite région de l'image est agrandie pour montrer un groupe de pixels.
- **Zoom sur un pixel spécifique** : Un seul pixel est isolé et agrandi, avec ses coordonnées (x = 120), (y = 162) et sa valeur d'intensité en niveaux de gris (135).

Pour résumer, l'image ci-dessus illustre le concept de représentation des images en niveaux de gris, où chaque pixel de l'image correspond à une valeur d'intensité spécifique. Ces valeurs sont organisées en une grille de lignes et de colonnes, avec chaque pixel ayant une position précise (coordonnées \(x, y\)) et une valeur numérique indiquant sa luminosité. Plus la valeur est élevée, plus le pixel sera clair; à l'inverse, une valeur faible correspond à un pixel sombre. Cette méthode de représentation permet de capturer les détails et les nuances d'une image en utilisant différentes intensités de gris.


## Représentation d'une image digitale (couleur)

![image](https://github.com/user-attachments/assets/c8663778-fd99-4557-9a19-7d546d406a08)

La figure *ci-haut* montre comment une image couleur est représentée selon le modèle RGB (Rouge, Vert, Bleu). Il existe deux méthodes principales pour représenter cette image en couleur :

### 1. **Superposition des canaux (Figure à gauche)**
   - **Principe de la superposition** : Cette partie illustre comment une image couleur est construite en combinant trois canaux distincts, chacun correspondant à l'une des trois couleurs primaires (Rouge, Vert, Bleu).
   - **Canaux RGB** : Chaque canal est une image en niveaux de gris qui indique l'intensité de la couleur associée (Rouge, Vert ou Bleu) pour chaque pixel de l'image. En superposant ces trois canaux, on obtient l'image finale en couleur.
   - **Taille de l'image** : Pour une image de taille \(N \times M\) pixels, la taille totale de l'image en couleur sera de \(N \times M \times 3\) octets. Le facteur 3 correspond aux trois canaux (Rouge, Vert, Bleu), chacun étant codé sur 1 octet.

### 2. **Représentation par carte des couleurs (Figure à droite)**
   - **Carte des couleurs** : Cette méthode alternative consiste à associer chaque pixel de l'image à une couleur spécifique déterminée par les valeurs des canaux Rouge, Vert et Bleu. Plutôt que de stocker séparément les canaux, cette méthode utilise une carte où chaque entrée correspond à une combinaison de ces trois couleurs.
   - **Encodage des pixels** : Chaque pixel de l'image est encodé sur 3 octets, soit 24 bits au total, répartis en 8 bits pour chaque couleur primaire (Rouge, Vert, Bleu). Cette méthode permet de représenter une grande variété de couleurs avec précision.

### 3. **Résumé de l'encodage**
   - **3 octets par pixel** : Chaque pixel de l'image est représenté par 3 octets, ce qui signifie qu'il peut avoir 256 niveaux d'intensité pour chacune des couleurs primaires (Rouge, Vert, Bleu). Cela permet de représenter un total de 16 777 216 (256 x 256 x 256) couleurs différentes.



En résumé, cette figure montre les deux approches pour représenter une image couleur : soit par superposition des trois canaux de couleurs, soit par l'utilisation d'une carte des couleurs, où chaque pixel est directement associé à une valeur colorimétrique unique. Ces méthodes illustrent la manière dont chaque pixel de l'image capture les informations de couleur pour créer une image riche en détails et en nuances.



[Retour en haut](#cours-imagerie)



Pour 1000 images en haute définition (HD) de taille 1920 x 1080 pixels, voici comment la représentation par canal et par carte des couleurs se compare en termes de taille de stockage :

### Représentation par canal (RGB)

![image](https://github.com/user-attachments/assets/436310df-c773-4fd2-9dbc-a3f03277b785)

- Chaque pixel est encodé sur 3 octets (1 octet par canal pour Rouge, Vert, et Bleu). La taille totale pour 1000 images est calculée comme suit :

$$ 
1000 \times 1920 \times 1080 \times 3 = 6,220,800,000 \text{ octets} \approx 5.8 \text{ Go} 
$$

### Représentation par carte des couleurs
En utilisant une carte de 256 couleurs (chaque pixel est encodé sur 1 octet) :

$$ 
(1000 \times 1920 \times 1080 \times 1) + (256 \times 3) = 2,073,600,768 \text{ octets} \approx 1.93 \text{ Go}
$$

### Conclusion
Pour 1000 images en HD, la méthode RGB occupe environ 5.8 Go, tandis que la méthode par carte des couleurs nécessite environ 1.93 Go, démontrant que la représentation par carte des couleurs est bien plus efficace en termes de stockage.


---

<a id="problemes-de-bruit-et-dimensionalite-des-images"></a>

### 4. Problèmes de Bruit et Dimensionalité des Images

#### Nature et origine du bruit dans les images
Le bruit dans les images est comparable aux interférences que vous pourriez entendre à la radio : des grésillements qui perturbent la clarté du signal. En imagerie, ce bruit se manifeste par des pixels anormaux ou des imperfections qui ne devraient pas être là. Par exemple, lorsque vous prenez une photo dans une faible lumière, vous remarquerez peut-être des petits points ou des grains qui ne devraient pas exister, semblables à la neige sur un écran de télévision. Ces "parasites" peuvent être causés par plusieurs facteurs, comme une mauvaise qualité de capteur ou des interférences électroniques.

Voici un exemple concret de génération de bruit dans une image en niveaux de gris :

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


```python
import numpy as np
import matplotlib.pyplot as plt

# Créer une matrice 3x3 représentant une image
I = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Afficher la matrice originale comme une image en niveaux de gris
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)  # Première sous-figure (1 ligne, 3 colonnes, 1ère position)
plt.imshow(I, cmap='gray')
plt.title("Avant la modification")
plt.colorbar()
plt.axis('off')

# Accéder à la deuxième ligne
ligne2 = I[1, :]
print("Deuxième ligne :")
print(ligne2)

# Modifier une colonne entière
I[:, 1] = 0
print("Matrice modifiée :")
print(I)

# Afficher la matrice modifiée comme une image en niveaux de gris
plt.subplot(1, 3, 2)  # Deuxième sous-figure (1 ligne, 3 colonnes, 2ème position)
plt.imshow(I, cmap='gray')
plt.title("Après la modification")
plt.colorbar()
plt.axis('off')

# Générer du bruit gaussien
noise = np.random.normal(0, 0.5, I.shape)

# Ajouter le bruit à la matrice modifiée
I_noisy = I + noise

# Afficher l'image avec le bruit ajouté
plt.subplot(1, 3, 3)  # Troisième sous-figure (1 ligne, 3 colonnes, 3ème position)
plt.imshow(I_noisy, cmap='gray')
plt.title("Avec bruit ajouté")
plt.colorbar()
plt.axis('off')

# Afficher les trois images côte à côte
plt.show()
```


#### Problème de la dimensionnalité
Maintenant, imaginez que vous avez une photo numérique d'une personne. Cette image contient potentiellement des millions de pixels, et chaque pixel détient une petite portion d'information visuelle. Si vous deviez analyser cette image pour reconnaître cette personne, chaque pixel deviendrait une pièce du puzzle, une caractéristique individuelle que vous devez prendre en compte.

Mais plus vous avez de pièces à analyser, plus il devient difficile de reconstruire l'image complète dans un temps raisonnable. C’est ce que nous appelons la "malédiction de la dimensionnalité". En d'autres termes, lorsque les données deviennent trop volumineuses, leur traitement devient extrêmement complexe et inefficace.

Prenons un exemple plus concret : imaginez que vous essayez de reconnaître un visage dans une foule à partir d'une photo haute résolution. Si vous deviez examiner chaque pixel de l'image, vous seriez rapidement submergé par la quantité de données à traiter. Mais si vous pouviez d'abord réduire la quantité de données en ne conservant que les parties de l'image les plus pertinentes (comme les contours du visage, les yeux, la bouche), vous pourriez accomplir cette tâche beaucoup plus rapidement.

C'est ici que la réduction de la dimensionnalité entre en jeu. Elle vous permet de filtrer les informations superflues et de vous concentrer sur les éléments clés. Pour continuer avec l'analogie du visage, au lieu de traiter une image entière, vous vous concentrez sur les éléments essentiels comme la forme du visage, la distance entre les yeux, et la courbure des lèvres, qui sont des indicateurs bien plus fiables pour l'identification.

**Transition vers l'extraction des caractéristiques :** Après avoir réduit la dimensionnalité, l'étape suivante est l'extraction des caractéristiques spécifiques. Par exemple, pour reconnaître une personne dans une image, il serait plus efficace d'identifier les éléments distinctifs comme la forme du nez, le contour des yeux, et la structure des lèvres, plutôt que d'analyser chaque pixel individuellement. Ces caractéristiques, une fois extraites, deviennent les blocs de construction que nous utilisons pour comprendre ce que l'image représente réellement.

---

<a id="traitement-dimages-bas-moyen-et-haut-niveau"></a>

### 5. Traitement d’Images : Bas, Moyen et Haut Niveau

#### Extraction des caractéristiques (Bas niveau)
Le traitement d'images débute par l'extraction de caractéristiques spécifiques de l'image brute, un processus comparable à l'identification des traits distinctifs d'une personne. Par exemple, si vous cherchez à reconnaître une personne dans une image, vous ne vous attardez pas sur chaque détail insignifiant, mais vous cherchez plutôt des caractéristiques reconnaissables comme la forme des yeux, la structure du nez, et la courbure des lèvres. Ces éléments sont plus que de simples points de données ; ils sont les indicateurs visuels qui vous aident à identifier la personne.

[Retour en haut](#cours-imagerie)


---

# Annexe  1 : **carte des couleurs** 

La **carte des couleurs** est une représentation visuelle qui associe chaque valeur numérique d'un pixel à une couleur spécifique. En d'autres termes, une carte de couleurs est un tableau ou une matrice où chaque position représente un pixel, et la valeur à chaque position correspond à une couleur particulière selon une échelle prédéfinie.

### Exemple de Carte des Couleurs :

Prenons l'exemple d'une image très simple, en noir et blanc (échelle de gris) avec une taille de \( 2 \times 2 \) pixels. Voici comment pourrait se présenter une carte des couleurs pour cette image :

#### Image en Niveaux de Gris :
Supposons que nous avons une image de 2x2 pixels avec les valeurs suivantes :
- Pixel en (1,1) : 0 (noir)
- Pixel en (1,2) : 255 (blanc)
- Pixel en (2,1) : 127 (gris moyen)
- Pixel en (2,2) : 64 (gris foncé)

Voici la matrice des valeurs :
$$
\begin{bmatrix}
0 & 255 \\
127 & 64
\end{bmatrix}
$$

#### Carte des Couleurs Correspondante :
Une carte des couleurs associée pourrait être définie comme suit, en utilisant une échelle de gris où 0 représente le noir et 255 représente le blanc. Chaque valeur numérique dans la matrice sera convertie en une couleur en utilisant cette échelle.

- **(0, 0, 0)** pour le noir (0)
- **(255, 255, 255)** pour le blanc (255)
- **(127, 127, 127)** pour le gris moyen (127)
- **(64, 64, 64)** pour le gris foncé (64)

La carte des couleurs pour notre image serait alors :

$$
\begin{bmatrix}
(0, 0, 0) & (255, 255, 255) \\
(127, 127, 127) & (64, 64, 64)
\end{bmatrix}
$$

Chaque tuple représente la couleur d’un pixel en RGB.

### Carte des Couleurs en Couleur (Palette RGB) :

Pour une image en couleur, chaque pixel est représenté par un triplet (R, G, B) indiquant les intensités de rouge, vert, et bleu respectivement. Par exemple, une image de 2x2 pixels pourrait avoir une carte des couleurs comme suit :

$$
\begin{bmatrix}
(255, 0, 0) & (0, 255, 0) \\
(0, 0, 255) & (255, 255, 0)
\end{bmatrix}
$$

- **(255, 0, 0)** : Rouge pur
- **(0, 255, 0)** : Vert pur
- **(0, 0, 255)** : Bleu pur
- **(255, 255, 0)** : Jaune

### Utilisation Pratique :

La carte des couleurs est particulièrement utile pour les images scientifiques ou médicales, où les valeurs de pixel représentent des données quantitatives (comme la température, l'altitude, etc.), et où l'on souhaite visualiser ces données sous forme de couleurs pour une interprétation plus facile. Un exemple courant est l’utilisation d’une carte de couleurs dans les images thermiques, où chaque température est associée à une couleur spécifique allant du bleu (froid) au rouge (chaud).

### Conclusion :

La carte des couleurs est un outil essentiel pour visualiser les données numériques d'une image en associant des valeurs numériques à des couleurs spécifiques, facilitant ainsi l'interprétation visuelle de l'information contenue dans l'image.




---



# Annexe 2 -  Utilisation des Filtres, Padding, et Convolutions dans le Traitement des Images

Dans le cadre du traitement d'images, en particulier pour l'extraction des caractéristiques de bas niveau, les concepts de filtres, padding, et convolutions jouent un rôle fondamental. 


## Exemple :

```
+-------------+   +-------------+   +-------------+
|  Image de   |-->|   Filtre    |-->| Caractéristique: |
|   l'oiseau  |   | Convolution |   |      Bec       |
+-------------+   +-------------+   +-------------+
```

- Ce schéma montre comment une image bruité volumineuse (haute dimension) d'un oiseau passe par un filtre et une convolution pour extraire une caractéristique spécifique, comme le bec de l'oiseau (caractéstique bas-niveau).


## Comment ça fonctionne ?

- C'est la **convolution** qui applique les **filtres** à l'image pour extraire des caractéristiques spécifiques.
- On obtient les caractéstique bas niveaux grâce à la combinaison des deux : les **filtres** et la **convolution**. 


## Définitions : 

- **Filtres** : Ils sont utilisés pour détecter des caractéristiques spécifiques dans une image, comme des contours ou des motifs. Par exemple, un filtre pourrait être conçu pour mettre en évidence les contours du bec de l'oiseau.

- **Convolution** : C'est le processus par lequel le filtre est appliqué sur l'image. Pendant la convolution, le filtre est glissé sur toute l'image, et à chaque position, il effectue des calculs pour produire une nouvelle image, appelée carte de caractéristiques, qui montre où la caractéristique recherchée (comme le bec) est présente.



- **Filtres (ou noyaux)** : Un filtre est une petite matrice appliquée à une image pour détecter certaines caractéristiques, comme les bords ou les textures. Par exemple, un filtre de détection de contours pourrait accentuer les lignes et les formes dans une image, permettant ainsi de capturer les informations essentielles pour des tâches ultérieures comme la reconnaissance de formes.

- **Convolution** : La convolution est le processus par lequel un filtre est appliqué à une image. Elle consiste à faire glisser le filtre sur toute l'image et à calculer un produit scalaire entre le filtre et les sous-matrices de l'image qu'il recouvre. Ce processus génère une nouvelle image, souvent appelée carte de caractéristiques, qui met en évidence les caractéristiques détectées par le filtre.

- **Padding** : Le padding est l'ajout de pixels supplémentaires autour des bords d'une image avant d'appliquer une convolution. Cela permet de préserver les dimensions de l'image après la convolution et d'éviter la perte d'informations importantes situées aux bords. Le padding est particulièrement utile lorsque l'on souhaite conserver les dimensions de l'image d'origine après l'application des filtres.

### Exemple d'Application : Extraction des Caractéristiques de Bas Niveau

L'extraction des caractéristiques de bas niveau à partir d'une image est essentielle pour des tâches telles que la reconnaissance de formes ou la détection d'objets. Par exemple, pour reconnaître un visage dans une image, on commence par détecter les contours des yeux, du nez, et de la bouche en appliquant des filtres adaptés. La convolution de ces filtres avec l'image originale produit des cartes de caractéristiques qui mettent en évidence ces éléments clés. 

Le padding peut être utilisé pour s'assurer que les contours près des bords de l'image ne sont pas perdus lors de l'application des filtres, garantissant ainsi que toutes les parties de l'image sont traitées de manière égale.

En résumé, les filtres, la convolution, et le padding sont des outils cruciaux dans le traitement d'images pour extraire des caractéristiques de bas niveau, qui sont ensuite utilisées pour des analyses plus avancées comme l'interprétation et la compréhension des scènes dans la vision par ordinateur.



---

# Annexe 3 : Extraction d'une Caractéristique d'une Image Bruitée


- Les filtres sont comme des lunettes spéciales que l'on pose sur une image pour mieux voir certaines parties, comme les bords ou les textures. Imaginez un filtre comme une petite grille que l'on fait glisser sur l'image, pixel par pixel, pour détecter ces éléments.

- La convolution est simplement le fait de faire glisser ce filtre sur l'image et de calculer combien il "correspond" à chaque partie de l'image. Le résultat est une nouvelle image qui montre où les caractéristiques, comme les bords, sont le plus présentes.

- Le padding consiste à ajouter des bordures autour de l'image avant d'appliquer le filtre pour s'assurer que même les bords de l'image sont bien pris en compte.

- Imaginez que vous avez une image d'un chat, mais cette image est un peu bruitée, ce qui signifie qu'il y a des pixels qui perturbent la clarté. Vous voulez extraire une caractéristique spécifique, comme le nez du chat, malgré ce bruit.

```
Image de chat bruitée :

* * * * *
* # * N *
* * @ * *
* * * * *
* * * * *

```

- **N** : représente le nez du chat.
- **#** et **@** : représentent du bruit qui rend l'image moins claire.

#### But : Identifier et extraire le nez du chat.

En appliquant des filtres et des convolutions, on va "nettoyer" l'image pour mieux détecter le nez du chat, même si l'image est perturbée par le bruit.

```
Après application d'un filtre et convolution :

0 0 0 0 0
0 1 1 N 0
0 1 0 1 0
0 0 0 0 0
0 0 0 0 0

```

- Les **1** indiquent où le filtre a trouvé des parties du nez.
- Le **N** montre que l'on a bien identifié la position du nez du chat malgré le bruit initial.

---

Ce schéma simplifié montre comment on peut utiliser des filtres pour extraire des caractéristiques importantes d'une image, même lorsqu'elle est bruitée, en se concentrant sur les éléments essentiels comme le nez du chat.


## RÉSUMÉ : 

- Imaginez que vous avez une image d'un chat. Si vous voulez identifier des parties spécifiques du visage du chat, comme son nez, vous allez utiliser des filtres. Ces filtres sont comme des petites grilles spéciales qui vont se concentrer sur certains détails, par exemple, les formes rondes ou les contours.

Quand vous appliquez un filtre sur l'image du chat, cela s'appelle une convolution. La convolution va passer ce filtre sur toute l'image, pixel par pixel, et générer une nouvelle image où seuls les éléments correspondant à ce que vous cherchez (comme le nez) seront mis en avant.

Le padding (remplissage) est un petit truc qui permet de s'assurer que même les parties du nez du chat qui sont près des bords de l'image sont bien prises en compte. C'est comme ajouter un petit cadre autour de l'image pour ne rien rater.


- https://github.com/hrhouma/Apprentissage-Non-Supervise/blob/main/08-ANN-intro-reseaux-neurones-apprentissage-non-supervise/01-cours-intro-imagerie.md
- https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
- https://yosinski.com/deepvis
- https://deeplizard.com/learn/video/qSTv_m-KFk0
- https://setosa.io/ev/image-kernels/



---



----

# Annexe 4  : Bruit dans une image



```python
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, color, img_as_ubyte, exposure
from skimage.util import random_noise

# Charger l'image "astronaut" comme substitut pour "Lena"
image = data.astronaut()

# Convertir l'image en niveaux de gris
gray_image = color.rgb2gray(image)
gray_image = img_as_ubyte(gray_image)

# Ajouter du bruit gaussien
gaussian_noise_img = random_noise(gray_image, mode='gaussian', var=0.01)
gaussian_noise_img = img_as_ubyte(gaussian_noise_img)

# Ajouter du bruit de sel et poivre
salt_pepper_noise_img = random_noise(gray_image, mode='s&p', amount=0.05)
salt_pepper_noise_img = img_as_ubyte(salt_pepper_noise_img)

# Ajouter du bruit de speckle
speckle_noise_img = random_noise(gray_image, mode='speckle', var=0.01)
speckle_noise_img = img_as_ubyte(speckle_noise_img)

# Étirement du contraste pour augmenter les niveaux de gris
stretched_gaussian = exposure.rescale_intensity(gaussian_noise_img, in_range='image', out_range=(0, 255))
stretched_salt_pepper = exposure.rescale_intensity(salt_pepper_noise_img, in_range='image', out_range=(0, 255))
stretched_speckle = exposure.rescale_intensity(speckle_noise_img, in_range='image', out_range=(0, 255))

# Afficher les résultats
plt.figure(figsize=(15, 15))

plt.subplot(3, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title("Image originale en niveaux de gris")
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(gaussian_noise_img, cmap='gray')
plt.title("Bruit gaussien")
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(stretched_gaussian, cmap='gray')
plt.title("Bruit gaussien avec niveaux de gris augmentés")
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(salt_pepper_noise_img, cmap='gray')
plt.title("Bruit de sel et poivre")
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(stretched_salt_pepper, cmap='gray')
plt.title("Bruit de sel et poivre avec niveaux de gris augmentés")
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(speckle_noise_img, cmap='gray')
plt.title("Bruit de speckle")
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(stretched_speckle, cmap='gray')
plt.title("Bruit de speckle avec niveaux de gris augmentés")
plt.axis('off')

plt.show()


```




```python
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, color, img_as_ubyte, exposure
from skimage.util import random_noise

# Charger l'image "astronaut" comme substitut pour "Lena"
image = data.astronaut()

# Convertir l'image en niveaux de gris
gray_image = color.rgb2gray(image)
gray_image = img_as_ubyte(gray_image)

# Ajouter une grande quantité de bruit de sel et poivre
salt_pepper_noise_img = random_noise(gray_image, mode='s&p', amount=0.3)  # Augmenter la quantité de bruit
salt_pepper_noise_img = img_as_ubyte(salt_pepper_noise_img)

# Étirement du contraste pour augmenter les niveaux de gris
stretched_salt_pepper = exposure.rescale_intensity(salt_pepper_noise_img, in_range='image', out_range=(0, 255))

# Afficher les résultats
plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.imshow(salt_pepper_noise_img, cmap='gray')
plt.title("Bruit de sel et poivre (30%)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(stretched_salt_pepper, cmap='gray')
plt.title("Avec niveaux de gris augmentés")
plt.axis('off')

plt.show()
```














---------


<a id="references"></a>

# Références


1. **Livres**
   - *Computer Vision: Algorithms and Applications* par Richard Szeliski : Ce livre est une ressource complète qui couvre les algorithmes et les applications de la vision par ordinateur.
   - *Deep Learning for Computer Vision* par Rajalingappaa Shanmugamani : Ce livre explore l'utilisation des techniques d'apprentissage profond pour la vision par ordinateur, notamment les réseaux de neurones convolutionnels (CNN).

2. **Articles et Guides**
   - *Deep Learning for Computer Vision: The Abridged Guide* par Run:ai : Cet article explore comment utiliser l'apprentissage profond pour améliorer les projets de vision par ordinateur, en se concentrant sur les architectures de réseaux de neurones convolutionnels (CNN)[4].
   - *What is Computer Vision?* par IBM : Un aperçu de la façon dont la vision par ordinateur utilise l'intelligence artificielle pour interpréter et comprendre les données visuelles[5].

3. **Cours et Conférences**
   - *First Principles of Computer Vision* par Shree Nayar : Une série de conférences qui aborde les fondements physiques et mathématiques de la vision par ordinateur[3].

4. **Sites Web et Ressources en Ligne**
   - *Computer Vision: What it is and why it matters* par SAS : Une ressource qui explique comment la vision par ordinateur fonctionne et pourquoi elle est importante dans divers secteurs[6].

Ces références couvrent une variété d'aspects de la vision par ordinateur, allant des concepts de base aux applications avancées utilisant l'apprentissage profond.

---

# Autres citations:

[1] https://www.youtube.com/watch?v=wVE8SFMSBJ0

[2] https://www.run.ai/guides/deep-learning-for-computer-vision

[3] https://www.ibm.com/topics/computer-vision

[4] https://www.sas.com/en_th/insights/analytics/computer-vision.html

[5] https://vitrinelinguistique.oqlf.gouv.qc.ca/fiche-gdt/fiche/8374005/vision-par-ordinateur

[6] https://www.motionmetrics.com/how-artificial-intelligence-revolutionized-computer-vision-a-brief-history/

[7] https://www.youtube.com/watch?v=OnTgbN3uXvw



---

# Autres citations pour les filtres et conv :

- https://github.com/hrhouma/Apprentissage-Non-Supervise/blob/main/08-ANN-intro-reseaux-neurones-apprentissage-non-supervise/01-cours-intro-imagerie.md
- https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
- https://yosinski.com/deepvis
- https://deeplizard.com/learn/video/qSTv_m-KFk0
- https://setosa.io/ev/image-kernels/



