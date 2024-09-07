# **L'objectif du TP et la justification des choix d'architecture :**

- L’objectif global de ce travail est d’apprendre à utiliser un **autoencodeur convolutif** et un **SVM** pour traiter des images, dans le but de compresser et reconstruire ces images tout en effectuant une classification basée sur leurs représentations compactes (appelées **embeddings**). 
- Pour bien comprendre pourquoi nous faisons cela, décomposons les étapes et expliquons chaque concept :

---
# **1. Pourquoi compresser les images ?**
---

Lorsque nous avons une image, elle est souvent très grande, avec des milliers de pixels. Si nous utilisons directement ces images pour des tâches comme la classification, cela peut nécessiter une grande puissance de calcul, car le modèle doit traiter chaque pixel individuellement.

L’idée de **compresser** les images (ou d’**encoder** les images) est d’extraire les **caractéristiques essentielles** de l’image sans avoir à traiter chaque pixel. C'est comme si nous réduisions l'image à son **essence** la plus importante : les caractéristiques principales qui représentent l'objet dans l'image, tout en ignorant les détails inutiles.

L’encodeur, qui fait partie de l’autoencodeur, apprend à réduire la taille de l'image en une représentation plus petite et plus compacte (appelée **embedding**). Ces embeddings sont beaucoup plus petits que les images originales, mais contiennent toutes les informations nécessaires pour différencier les dauphins des requins, par exemple.

---
# **2. Pourquoi reconstruire les images ?**
---

La partie **reconstruction** dans un autoencodeur permet de vérifier que le modèle n’a pas perdu d’informations importantes lors de la compression. 

Autrement dit, si le modèle est capable de **reconstruire l’image originale** à partir de l’embedding (la version compressée de l'image), cela signifie que l’embedding contient suffisamment d’informations pour représenter l'image d'origine.

Cela nous permet de **valider** que la compression n'est pas trop agressive et que l’encodeur a bien appris à capturer les caractéristiques importantes de l'image.

---
# **3. Pourquoi appliquer un SVM après la compression ?**
---

Une fois que nous avons les **embeddings** (représentations compactes) des images, l’étape suivante est de classifier ces images pour savoir si elles appartiennent à la classe des **dauphins** ou des **requins**.

Le **SVM (Support Vector Machine)** est un **classificateur** très efficace, spécialement conçu pour travailler avec des données déjà prétraitées ou des caractéristiques extraites. Il est utilisé ici pour prendre les **embeddings** (qui sont des vecteurs de caractéristiques) et **décider** à quelle classe chaque embedding appartient.

Pourquoi utiliser un **SVM** plutôt qu’un autre modèle ?
- Le SVM est particulièrement bien adapté pour classifier des **données dans un espace de haute dimension** (comme les embeddings).
- Il cherche à maximiser la séparation entre les classes, c’est-à-dire qu’il tente de tracer une frontière qui sépare au mieux les **dauphins** des **requins** dans l'espace des embeddings.
- Il est très performant lorsque les données sont bien représentées, ce qui est le cas ici avec les embeddings.

---
# **4. Pourquoi combiner compression et classification ?**
---

La **combinaison** d'un **autoencodeur** et d'un **SVM** permet de profiter des points forts des deux méthodes :
- L’autoencodeur simplifie et compresse les données en extrayant les caractéristiques **importantes** de chaque image.
- Le SVM utilise ces caractéristiques pour **classer** efficacement les images en fonction des informations contenues dans l’embedding.

Cette combinaison est particulièrement utile lorsque :
- Vous avez des images volumineuses et complexes (comme les images de dauphins et de requins).
- Vous voulez réduire la **complexité** de vos données avant de les classer.
- Vous voulez vérifier que votre modèle n’a pas perdu d’informations importantes en compressant les images, ce que nous vérifions en les **reconstruisant**.

---
# **Conclusion : Pourquoi tout ça ?**
---

1. **Compresser les images** (grâce à l’autoencodeur) permet de réduire la quantité d’informations tout en préservant les caractéristiques importantes.
2. **Reconstruire les images** permet de s’assurer que la compression n’a pas supprimé d’informations essentielles.
3. **Appliquer un SVM** sur les embeddings compressés permet de classifier les images de manière efficace et rapide, en utilisant des données déjà traitées.

Cette méthode est particulièrement intéressante lorsque vous travaillez avec des ensembles de données *volumineux*, comme des images, car elle vous permet de simplifier les données tout en conservant les éléments nécessaires pour faire de bonnes prédictions.
