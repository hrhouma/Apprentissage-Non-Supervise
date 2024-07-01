### Table des Matières

1. [Introduction à l'Intelligence Artificielle et à l'Apprentissage Machine](#introduction-a-lintelligence-artificielle-et-a-lapprentissage-machine)
   - [Définition et historique](#definition-et-historique)
   - [Importance et applications actuelles](#importance-et-applications-actuelles)
   - [Exemples d'applications concrètes](#exemples-dapplications-concretes)

2. [Concepts de Base de l'Apprentissage Automatique](#concepts-de-base-de-lapprentissage-automatique)
   - [Apprentissage supervisé vs non supervisé](#apprentissage-supervise-vs-non-supervise)
   - [Types d'algorithmes : régression, classification, clustering](#types-dalgorithmes--regression-classification-clustering)
   - [Variables dépendantes et explicatives](#variables-dependantes-et-explicatives)

3. [Pratique avec TensorFlow et Google Colab](#pratique-avec-tensorflow-et-google-colab)
   - [Google Colaboratory](#google-colaboratory)
   - [Installation et configuration de TensorFlow](#installation-et-configuration-de-tensorflow)
   - [Exercices pratiques avec TensorFlow](#exercices-pratiques-avec-tensorflow)

---

## Introduction à l'Intelligence Artificielle et à l'Apprentissage Machine

### Définition et Historique

**Intelligence Artificielle (IA)**
- John McCarthy, 1956 : L’un des fondateurs de la discipline et inventeur du mot AI.
- Définitions non formelles : Doter un système d’un mécanisme qui permet de simuler le comportement d’un être vivant, sa capacité d’adaptation à son environnement, et sa création de stratégie de survie pour résoudre des problèmes du monde réel.
- "L'ensemble des théories et des techniques mises en œuvre en vue de réaliser des machines capables de simuler l'intelligence."

**Apprentissage Automatique (AA)**
- Arthur Samuel, 1959 : Ensemble des techniques permettant à une machine d’apprendre à réaliser une tâche sans avoir à la programmer explicitement pour cela.
- Création et utilisation de modèles qui apprennent à partir des données pour prédire des résultats selon une problématique et un contexte de données.
- Tom Mitchell : “A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T as measured by P improves with experience E.”

### Importance et Applications Actuelles
- L'IA et l'AA sont essentiels pour résoudre des problèmes complexes et automatiser des tâches.
- Applications : reconnaissance d'image, traitement du langage naturel, diagnostics médicaux, véhicules autonomes, etc.

### Exemples d'Applications Concrètes
- Reconnaissance vocale : Siri, Google Assistant.
- Traduction automatique : Google Translate.
- Recommandations de produits : Amazon, Netflix.
- Analyse prédictive : Détection de fraudes, maintenance prédictive.

---

## Concepts de Base de l'Apprentissage Automatique

### Apprentissage Supervisé vs Non Supervisé

**Apprentissage Supervisé**
- Sortie avec un label discret (Classification) ou numérique (Régression).
- Exemple : Prédire le prix d'une maison en fonction de sa taille et du nombre de chambres.

**Apprentissage Non Supervisé**
- Clustering : Classification de données sans labels prédéfinis.
- Exemple : Regrouper des clients en segments en fonction de leurs comportements d'achat.

### Types d'Algorithmes : Régression, Classification, Clustering
- **Régression** : Prédiction de valeurs continues.
- **Classification** : Prédiction de catégories discrètes.
- **Clustering** : Regroupement de données similaires.

### Variables Dépendantes et Explicatives
- **Variable Dépendante (Y)** : Variable que l'on souhaite prédire ou expliquer.
- **Variables Explicatives (X1, X2, ..., Xn)** : Variables utilisées pour prédire ou expliquer la variable dépendante.

---

## Pratique avec TensorFlow et Google Colab

### Google Colaboratory
- Création de Notebooks en ligne pour écrire et exécuter du code.
- Utilisation de CPU/GPU pour accélérer l'entraînement des modèles.
- [Lien vers Google Colab](https://colab.research.google.com/notebooks/intro.ipynb).

### Installation et Configuration de TensorFlow
- Vérifier la version installée par défaut :
  ```python
  import tensorflow as tf
  print(tf.__version__)
  ```
- Désinstaller TensorFlow existant et installer TensorFlow 2.0 :
  ```python
  !pip uninstall tensorflow
  !pip install tensorflow-gpu==2.0.0.alpha0
  ```
- Tester TensorFlow 2.0 :
  ```python
  import tensorflow as tf
  x = tf.Variable(10)
  y = tf.Variable(2)  
  z = tf.add(x, y)
  print("somme de x et y =", z.numpy())
  ```

### Exercices Pratiques avec TensorFlow

**Exercice 1 : Générer un tableau et tracer un graphe**
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(2, 100, 10)
y = (x**2) - x*3 -5
plt.plot(x, y)
plt.show()
```

**Exercice 2 : Modèle de forêt aléatoire**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Charger le dataset
iris = load_iris()
X = iris.data
y = iris.target

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

# Construire le modèle
clf = RandomForestClassifier(n_estimators=10)

# Entraîner le classificateur
clf.fit(X_train, y_train)

# Prédictions
predicted = clf.predict(X_test)

# Vérifier la précision
print(accuracy_score(predicted, y_test))
```

**Exercice 3 : Régression linéaire**
```python
import numpy as np
from sklearn.linear_model import LinearRegression 

# Ensemble d'apprentissage avec un bruit
X_train = 5 * np.random.rand(1001)
y_train = 4 + 3 * X_train + np.random.randn(1001)

# Modèle de régression linéaire
lin_reg = LinearRegression()
lin_reg.fit(X_train.reshape(-1, 1), y_train)

# Prédiction
X_predict = 3 * np.random.rand(1001)
y_predict = lin_reg.predict(X_predict.reshape(-1, 1))

# Affichage
print('---X_predict---')
print(X_predict)
print('---Y_predict---')
print(y_predict)
```

Pour continuer l'extraction complète, nous devons procéder de manière similaire pour chaque partie du cours, en extrayant toutes les informations textuelles et les codes des fichiers fournis. Si vous souhaitez que je continue avec une autre partie ou d'autres sections spécifiques, veuillez m'indiquer lesquelles.
