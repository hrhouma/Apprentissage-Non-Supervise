# Partie 2 : Chargement des données

## Description
Dans cette partie, nous chargeons les fichiers de données et utilisons les bibliothèques `numpy`, `pandas`, et `os` pour parcourir les répertoires contenant les fichiers. Cette étape est essentielle pour vérifier la présence des fichiers nécessaires et les préparer pour le prétraitement.

## Code

```python
import numpy as np
import pandas as pd

# Les fichiers de données d'entrée sont disponibles dans le répertoire "../input/".
# Par exemple, exécuter ceci listera tous les fichiers dans le répertoire d'entrée.
import os
for dirname, _, filenames in os.walk('/../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

## Justification
- **`numpy`** : utilisé pour effectuer des calculs mathématiques sur des tableaux, qui seront essentiels lorsque nous manipulerons des images (chaque image est un tableau de pixels).
  
- **`pandas`** : utilisé pour manipuler des données tabulaires, comme des fichiers CSV. Cela facilitera la lecture des métadonnées associées aux images.

- **`os.walk`** : cette commande permet de parcourir récursivement un répertoire, listant tous les fichiers qui y sont présents. Cela est utile pour vérifier la structure des répertoires et localiser les fichiers nécessaires avant de les charger.

Cette étape garantit que toutes les données d'entrée sont bien accessibles et prêtes pour le traitement dans les étapes suivantes.

---

# Annexe : code 
---

L'instruction suivante :

```python
import os
for dirname, _, filenames in os.walk('/../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

explore les répertoires et liste les fichiers disponibles. Voici une explication détaillée de chaque élément :

### 1. `import os`
Le module `os` permet d'interagir avec le système de fichiers. Vous pouvez ainsi parcourir des répertoires, gérer des fichiers et obtenir des informations sur le système d'exploitation.

### 2. `os.walk('/../input')`
Cette fonction génère les noms de fichiers dans un répertoire donné, récursivement. Ici, le chemin spécifié est `'/../input'`, ce qui signifie que nous parcourons le répertoire parent du dossier actuel, puis le dossier `input`.

### 3. `for dirname, _, filenames in os.walk(...)`
- `dirname` : contient le chemin du répertoire en cours de traitement.
- `_` : une variable qui reçoit le sous-dossier dans chaque répertoire (mais n'est pas utilisée ici).
- `filenames` : une liste des fichiers dans le répertoire en cours.

### 4. `print(os.path.join(dirname, filename))`
Cette ligne affiche le chemin complet de chaque fichier dans le répertoire en utilisant `os.path.join` pour combiner correctement les segments du chemin.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

# Partie 3 : Importation des bibliothèques

## Description
Cette partie consiste à importer toutes les bibliothèques nécessaires pour le traitement d'images et la construction des modèles de machine learning. Ces bibliothèques couvrent des fonctions essentielles comme la manipulation d'images, la visualisation de données, et la construction de modèles d'autoencodeurs.

## Code

```python
import cv2
import tqdm
import tarfile
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input
from keras.models import Sequential, Model
```

## Justification
- **`cv2`** : OpenCV, utilisé pour charger et traiter des images. C'est une bibliothèque puissante pour la manipulation d'images et de vidéos.
  
- **`tqdm`** : utilisée pour afficher des barres de progression lors de l'exécution de boucles longues, ce qui est pratique lorsque vous traitez beaucoup d'images ou d'autres données.

- **`tarfile`** : permet de travailler avec des fichiers compressés, comme ceux au format `.tgz`. Cela est souvent utilisé pour manipuler des datasets compressés.

- **`matplotlib.pyplot`** : utilisée pour visualiser les données sous forme de graphiques ou pour afficher des images.

- **`train_test_split`** (scikit-learn) : cette fonction permet de diviser les données en ensembles d'entraînement et de test, une étape cruciale pour l'entraînement et l'évaluation de modèles.

- **Keras (`layers`, `models`)** : Keras est une API de haut niveau pour construire et entraîner des réseaux de neurones. Ici, nous l'utilisons pour définir les couches du modèle d'autoencodeur.

---

# Annexe : code 
---

L'instruction suivante :

```python
import cv2
import tqdm
import tarfile
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input
from keras.models import Sequential, Model
```

sert à importer les bibliothèques utilisées dans ce projet. Voici une description de ce que fait chaque importation :

### 1. `import cv2`
OpenCV est une bibliothèque pour le traitement d'images et de vidéos. Vous l'utiliserez pour charger, modifier et manipuler les images de votre dataset.

### 2. `import tqdm`
`tqdm` est une bibliothèque qui permet d'afficher une barre de progression lors de l'exécution d'une boucle. Cela est très utile pour suivre l'avancement des tâches longues comme le traitement d'un grand nombre d'images.

### 3. `import tarfile`
Cette bibliothèque permet de lire et d'extraire des fichiers à partir d'archives compressées (comme `.tgz`). Vous l'utiliserez pour manipuler des datasets compressés.

### 4. `import matplotlib.pyplot as plt`
Matplotlib est utilisée pour visualiser des données sous forme de graphiques. Avec `pyplot`, vous pourrez afficher des images, des graphiques et suivre les performances de votre modèle.

### 5. `from sklearn.model_selection import train_test_split`
Cette fonction de scikit-learn est utilisée pour diviser vos données en ensembles d'entraînement et de test. Cela vous permettra d'entraîner votre modèle sur une partie des données et de l'évaluer sur une autre partie.

### 6. `from keras.layers import Dense, Flatten, Reshape, Input`
Ces éléments de Keras sont utilisés pour définir les couches d'un réseau de neurones. Vous les utiliserez pour construire un autoencodeur, en définissant des couches denses et des couches de régression.

### 7. `from keras.models import Sequential, Model`
Ces deux classes de Keras sont utilisées pour créer et entraîner des modèles de réseaux de neurones.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

# Partie 4 : Chemins des fichiers

## Description
Dans cette partie, nous définissons les chemins vers les fichiers d'images et d'attributs. Centraliser ces chemins permet une gestion plus simple et plus flexible, surtout si vous devez modifier l'emplacement des fichiers à un moment donné.

## Code

```python
# Chemins des fichiers d'attributs et des images LFW
ATTRS_NAME = "../input/lfw_attributes.txt"
IMAGES_NAME = "../input/lfw-deepfunneled.tgz"
RAW_IMAGES_NAME = "../input/lfw.tgz"
```

## Justification
Définir les chemins des fichiers au début du script facilite la maintenance du projet. Si les fichiers changent d'emplacement ou si vous souhaitez utiliser d'autres datasets, vous pouvez modifier ces variables en un seul endroit.

- **`ATTRS_NAME`** : contient le chemin vers le fichier d'attributs, qui contient des informations importantes sur les images (comme des labels ou des métadonnées).

- **`IMAGES_NAME`** : contient le chemin vers les images LFW compressées, qui sont prétraitées et prêtes à être utilisées.

- **`RAW_IMAGES_NAME`** : contient le chemin vers les images brutes, non prétraitées. Cela vous permet de choisir la version des images que vous souhaitez utiliser.

---

# Annexe : code 
---

L'instruction suivante :

```python
ATTRS_NAME = "../input/lfw_attributes.txt"
IMAGES_NAME = "../input/lfw-deepfunneled.tgz"
RAW_IMAGES_NAME = "../input/lfw.tgz"
```

définit les chemins des fichiers nécessaires au projet. Voici une description détaillée de ces variables :

### 1. `ATTRS_NAME`
Cette variable contient le chemin vers le fichier des attributs des images (généralement un fichier texte ou CSV). Ce fichier peut contenir des informations supplémentaires sur les images, telles que des labels ou des métadonnées.

### 2. `IMAGES_NAME`
Cette variable contient le chemin vers le fichier compressé contenant les images LFW (Labeled Faces in the Wild), qui est un dataset prétraité. Ce fichier est utilisé pour charger les images déjà prêtes pour le modèle.

### 3. `RAW_IMAGES_NAME`
Cette variable contient le chemin vers les images brutes, non prétraitées. Vous pouvez l'utiliser si vous souhaitez travailler directement avec les images brutes et les traiter vous-même.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

