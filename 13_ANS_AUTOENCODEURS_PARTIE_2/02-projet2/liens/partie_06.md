Parfait ! Voici la suite avec la **Partie 5 : Décodage d'images** dans le style demandé.

---

# Partie 5 : Décodage d'images

## Description
Dans cette partie, nous créons une fonction pour décoder les images à partir de données brutes (bytes) et les transformer en images exploitables sous forme de matrices de pixels. Cette étape est cruciale pour transformer les images stockées sous forme compressée dans un format que les modèles d'apprentissage automatique peuvent utiliser.

## Code

```python
def decode_image_from_raw_bytes(raw_bytes):
    # Décodage des bytes en image
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
```

## Justification

Cette fonction est essentielle pour convertir les images stockées sous forme de bytes en images que le modèle peut traiter. 

### 1. `cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)`
- Cette ligne décode les bytes en une image. `cv2.imdecode` prend des données brutes et les transforme en une image que l’on peut manipuler sous forme de tableau de pixels.
- `np.asarray(bytearray(raw_bytes), dtype=np.uint8)` permet de convertir les bytes en un tableau numpy compatible avec OpenCV, nécessaire pour que la fonction `imdecode` fonctionne.

### 2. `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
- OpenCV charge les images dans un format BGR par défaut. Cette ligne convertit l'image en format RGB, plus standard pour la plupart des applications de traitement d'images, en particulier avec des réseaux de neurones.

Cette étape garantit que les images compressées sont bien converties dans un format exploitable par nos modèles d'apprentissage.

---

# Annexe : code 
---

L'instruction suivante :

```python
def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
```

décodage des images à partir de bytes en image RGB, utilisable par nos modèles. Voici une explication détaillée de ce processus :

### 1. Conversion des bytes en tableau
- `bytearray(raw_bytes)` : convertit les bytes bruts en une séquence d’octets.
- `np.asarray(..., dtype=np.uint8)` : transforme cette séquence d’octets en un tableau numpy avec des éléments de type `uint8`, qui est le type standard pour représenter les valeurs des pixels.

### 2. Décodage de l'image
- `cv2.imdecode(...)` : prend le tableau numpy créé à partir des bytes et le décode en image, sous un format exploitable par OpenCV.

### 3. Conversion en RGB
- `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` : cette fonction convertit l’image de BGR (format utilisé par OpenCV) en RGB, qui est le format standard utilisé pour la plupart des applications de vision par ordinateur.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Passons maintenant à la **Partie 6 : Chargement du jeu de données LFW**.

---

# Partie 6 : Chargement du jeu de données LFW

## Description
Cette partie contient une fonction permettant de charger les images du jeu de données LFW (Labeled Faces in the Wild) ainsi que leurs attributs. Nous extrayons les images à partir d'un fichier compressé et récupérons les attributs associés. Les images sont également redimensionnées pour correspondre aux exigences du modèle.

## Code

```python
def load_lfw_dataset(
        use_raw=False,
        dx=80, dy=80,
        dimx=45, dimy=45):
    
    # Lire les attributs
    df_attrs = pd.read_csv(ATTRS_NAME, sep='\t', skiprows=1)
    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])
    imgs_with_attrs = set(map(tuple, df_attrs[["person", "imagenum"]].values))
    
    # Lire les photos
    toutes_les_photos = []
    ids_photos = []
    
    # Utilisation de tqdm pour afficher une barre de progression
    from tqdm.notebook import tqdm
    with tarfile.open(RAW_IMAGES_NAME if use_raw else IMAGES_NAME) as f:
        for m in tqdm(f.getmembers()):
            if m.isfile() and m.name.endswith(".jpg"):
                # Préparer l'image
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                img = img[dy:-dy, dx:-dx]
                img = cv2.resize(img, (dimx, dimy))
                
                # Extraire l'identifiant de la personne et l'ajouter aux données collectées
                nom_fichier = os.path.split(m.name)[-1]
                nom_splitté = nom_fichier[:-4].replace('_', ' ').split()
                id_personne = ' '.join(nom_splitté[:-1])
                numéro_photo = int(nom_splitté[-1])
                if (id_personne, numéro_photo) in imgs_with_attrs:
                    toutes_les_photos.append(img)
                    ids_photos.append({'person': id_personne, 'imagenum': numéro_photo})
    
    ids_photos = pd.DataFrame(ids_photos)
    toutes_les_photos = np.stack(toutes_les_photos).astype('uint8')
    tous_les_attrs = ids_photos.merge(df_attrs, on=('person', 'imagenum')).drop(["person", "imagenum"], axis=1)
    
    return toutes_les_photos, tous_les_attrs
```

## Justification

Cette fonction charge les images et les attributs associés depuis un fichier compressé. Voici une justification de chaque étape :

### 1. Chargement des attributs
- **`pd.read_csv(ATTRS_NAME, sep='\t', skiprows=1)`** : cette ligne charge le fichier d'attributs en sautant la première ligne (header), puis le convertit en DataFrame pour faciliter la manipulation des données.
- **`imgs_with_attrs`** : cette variable stocke les associations entre les personnes et les numéros d'image, ce qui permet de vérifier quelles images ont des attributs associés.

### 2. Chargement des images
- **`tarfile.open(...)`** : ouvre le fichier d'archive compressé contenant les images. Selon la valeur de `use_raw`, on peut choisir entre les images brutes ou prétraitées.
- **`for m in f.getmembers():`** : itère à travers tous les fichiers de l'archive.
- **`decode_image_from_raw_bytes(f.extractfile(m).read())`** : utilise la fonction de décodage d'images définie précédemment pour transformer les bytes en image.
- **`cv2.resize(img, (dimx, dimy))`** : redimensionne chaque image pour correspondre aux dimensions exigées par le modèle.

### 3. Récupération des informations d'identité
- **`nom_fichier`, `id_personne`, `numéro_photo`** : extrait l'identifiant de la personne et le numéro de la photo à partir du nom de fichier. Ces informations sont ensuite utilisées pour associer les images aux attributs dans le DataFrame final.

### 4. Retourner les photos et attributs
- **`np.stack(toutes_les_photos)`** : regroupe toutes les images en un tableau numpy.
- **`pd.DataFrame(ids_photos)`** : crée un DataFrame avec les identifiants des photos.
- **`merge(...)`** : fusionne les informations des images avec les attributs.

---

# Annexe : code 
---

L'instruction suivante :

```python
def load_lfw_dataset(
        use_raw=False,
        dx=80, dy=80,
        dimx=45, dimy=45):
    ...
```

sert à charger les images et les associer à leurs attributs. Voici une explication détaillée du processus :

### 1. Chargement des attributs
La fonction commence par charger les attributs des images à partir d'un fichier CSV :

```python
df_attrs = pd.read_csv(ATTRS_NAME, sep='\t', skiprows=1)
```

Cette ligne utilise `pandas` pour lire le fichier CSV contenant les attributs des images, et `skiprows=1` saute la première ligne qui contient les en-têtes.

### 2. Chargement des images depuis une archive
Les images sont stockées dans une archive compressée au format `.tgz`. La bibliothèque `tarfile` est utilisée pour ouvrir cette archive et lire les images qu'elle contient :

```python
with tarfile.open(RAW_IMAGES_NAME if use_raw else IMAGES_NAME) as f:
    for m in tqdm(f.getmembers()):
        if m.isfile() and m.name.endswith(".jpg"):
            img = decode_image_from_raw_bytes(f.extractfile(m).read())
```

### 3. Redimensionnement et traitement
Les images sont ensuite redimensionnées et traitées pour correspondre aux besoins du modèle :

```python
img = cv2.resize(img, (dimx, dimy))
```

Cela garantit que toutes les images ont les mêmes dimensions.

---

[Retour à la Table des matières](../Tables-des-matieres.md)

---

Je vais continuer avec les autres parties du projet en gardant ce format.
