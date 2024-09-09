# 1 - Projet d'Apprentissage - Autoencoders

- Ce projet fait partie d'une série d'exercices disponibles sur GitHub ainsi que sur plusieurs plateformes éducatives en ligne. 
- Il est utilisé pour introduire et approfondir l'apprentissage des autoencodeurs dans le cadre de l'apprentissage non-supervisé.
- Les autoencodeurs sont des réseaux de neurones utilisés pour apprendre des représentations efficaces des données, souvent pour des tâches telles que la réduction de dimension ou la détection d'anomalies.

---

# 2 - Structure du Projet
---

- Le projet est structuré en plusieurs dossiers et fichiers comme suit :

```bash
.
├── projet1
│   └── Autoencoders-task.ipynb
└── input
```

# 3 - Instructions pour créer la structure de projet

1. **Créer un nouveau dossier `projet1`** :
   - Ouvrez votre terminal ou votre explorateur de fichiers.
   - Tapez la commande suivante pour créer le dossier :
     ```bash
     mkdir projet1
     ```

2. **Revenir au dossier parent** :
   - Utilisez la commande suivante pour revenir en arrière dans le répertoire parent :
     ```bash
     cd ..
     ```

3. **Créer un dossier `input`** :
   - Une fois dans le répertoire parent, créez un nouveau dossier `input` avec la commande suivante :
     ```bash
     mkdir input
     ```

4. **Télécharger les fichiers nécessaires et définir les variables** :
   - Téléchargez les fichiers suivants et placez-les dans le dossier `input` :
     - [lfw_attributes.txt](http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt)
     - [lfw-deepfunneled.tgz](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz)
     - [lfw.tgz](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
   
   - Définissez les variables suivantes dans le script ou le notebook en conséquence :
     ```python
     ATTRS_NAME = "../input/lfw_attributes.txt"
     IMAGES_NAME = "../input/lfw-deepfunneled.tgz"
     RAW_IMAGES_NAME = "../input/lfw.tgz"
     ```

5. **Lancer Jupyter Notebook** :
   - Une fois que la structure est en place, allez dans le dossier `projet1` :
     ```bash
     cd projet1
     ```
   - Ouvrez le terminal (ou la ligne de commande, selon votre système d'exploitation) et lancez Jupyter Notebook en tapant la commande suivante :
     ```bash
     jupyter notebook
     ```
   - Cela ouvrira une interface web où vous pourrez accéder et exécuter le fichier `Autoencoders-task.ipynb`.

Vous devriez obtenir une structure similaire à celle illustrée ci-dessus.

---

## Fichier `.ipynb`

Dans le dossier `projet1`, vous trouverez le fichier **Autoencoders-task.ipynb** qui contient tout le code nécessaire pour commencer l'exercice. Ce fichier est un notebook Jupyter où vous pouvez exécuter les différentes étapes de l'entraînement d'un autoencodeur.

---

Suivez les instructions et le contenu du notebook pour compléter le projet et apprendre à utiliser les autoencodeurs dans différents cas pratiques.
