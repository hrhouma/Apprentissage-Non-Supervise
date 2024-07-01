## Table des Matières

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

4. [Théorie Avancée de l'Apprentissage Profond](#theorie-avancee-de-lapprentissage-profond)
   - [Réseaux de neurones supervisés et non supervisés](#reseaux-de-neurones-supervises-et-non-supervises)
   - [Structure et fonctionnement des CNN](#structure-et-fonctionnement-des-cnn)
   - [Fonction de perte et rétropropagation](#fonction-de-perte-et-retropropagation)
   - [Exemples et applications avancées](#exemples-et-applications-avancees)

5. [Concepts et Applications des Réseaux de Neurones Convolutifs](#concepts-et-applications-des-reseaux-de-neurones-convolutifs)
   - [Introduction aux CNN](#introduction-aux-cnn)
   - [Exemples d'applications pratiques](#exemples-dapplications-pratiques)

6. [Optimisation pour l'Apprentissage Profond](#optimisation-pour-lapprentissage-profond)
   - [Surapprentissage et Sous-apprentissage](#surapprentissage-et-sous-apprentissage)
   - [Initialisation des Paramètres](#initialisation-des-parametres)
   - [Optimiseurs](#optimiseurs)
   - [Hyperparamètres](#hyperparametres)
   - [Régularisation](#regularisation)
   - [Apprentissage par Transfert](#apprentissage-par-transfert)
   - [Normalisation par Lots](#normalisation-par-lots)
   - [CPU et GPU pour l’Efficacité de Calcul](#cpu-et-gpu-pour-lefficacite-de-calcul)
   - [Augmentation et Déformation des Données](#augmentation-et-deformation-des-donnees)
   - [Stratégies de Choix des Hyperparamètres](#strategies-de-choix-des-hyperparametres)

7. [Exercices Pratiques](#exercices-pratiques)
   - [Exercice 1: Prédiction de Désabonnement de Clients](#exercice-1-prediction-de-desabonnement-de-clients)
   - [Exercice 2: Analyse des Ventes de Voitures](#exercice-2-analyse-des-ventes-de-voitures)
   - [Exercice 3: Classification des Images de Mode (Fashion MNIST)](#exercice-3-classification-des-images-de-mode-fashion-mnist)
   - [Exercice 4: Prédiction des Prix des Maisons](#exercice-4-prediction-des-prix-des-maisons)
   - [Exercice 5: Analyse des Performances des Étudiants](#exercice-5-analyse-des-performances-des-etudiants)

8. [Analyse des Données de Désabonnement](#analyse-des-donnees-de-desabonnement)
   - [Étapes de l'Analyse](#etapes-de-lanalyse)
   - [Importation des Bibliothèques](#importation-des-bibliotheques)
   - [Chargement et Inspection des Données](#chargement-et-inspection-des-donnees)
   - [Préparation des Données](#preparation-des-donnees)
   - [Analyse Exploratoire des Données](#analyse-exploratoire-des-donnees)
   - [Modélisation et Prédiction](#modelisation-et-prediction)
   - [Évaluation des Modèles](#evaluation-des-modeles)

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
```

### Exercices Pratiques avec TensorFlow (suite)

```python
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

---

## Théorie Avancée de l'Apprentissage Profond

### Fonctionnement du Réseau de Neurones
1. **Propagation vers l'avant**
   - Apprentissage
   - Fonction de coût (perte)
   - Calcul de l'erreur : Erreur = fonction d'erreur (y_calculé, y_donné)

2. **Propagation vers l’arrière**
   - Rétropropagation de l'erreur
   - Optimisation ou Calcul du gradient

### Types de Réseaux Profonds et Domaines d'Application
- **Supervisé**
  - Réseau multicouches (MLP) : Classification et Prédiction
  - Réseau à Convolution (CNN) : Vision par ordinateur
  - Réseau Récurrent (RNN) : Traitement du langage naturel, Analyse des séries temporelles

- **Non Supervisé**
  - Machines de Boltzmann Profondes : Systèmes de recommandations
  - Auto-Encodeurs : Systèmes de recommandations

### Préparation des Données
- **Mise à l'échelle (Scaling)**
  - Standardisation : Zi = (Xi - moyenne) / écart type
  - Normalisation : Zi = (Xi - min) / (max - min)

### Concepts Clés
- **Époque (Epoch)**
  - Nombre d'itérations sur l'ensemble d'apprentissage

- **Fonction de Coût vs Fonction d'Optimisation**
  - Fonction de coût : Mesure de l'erreur entre Y_calculé et Y_donné
  - Fonction d'optimisation : Minimise l'erreur

- **Rétropropagation et Optimisation**
  - La rétropropagation de l'erreur ajuste les poids synaptiques pour minimiser l'erreur.

- **Fonctions d'Erreur**
  - Pour la régression : Erreur Quadratique Moyenne (MSE), Erreur Absolue Moyenne (MAE)
  - Pour la classification : Perte charnière, Perte logistique

- **Optimisation et Descente du Gradient**
  - Descente du Gradient : Minimiser la fonction de coût en ajustant les paramètres dans la direction opposée du gradient.
  - Optimiseurs courants : Gradient Stochastique, Mini-Batch Gradient, Full Batch Gradient

- **Problèmes de Déviation des Erreurs**
  - Gradient Stochastique : Plus rapide, moins de mémoire, peut éviter les minima locaux.
  - Mini-Batch Gradient : Meilleure approximation du gradient, permet la parallélisation des calculs.

- **Hyperparamètres**
  - Taux d'apprentissage (learning rate) : Détermine la proportion du gradient utilisée lors de l'étape suivante de rétropropagation.
  - Momentum : Accélère le processus de convergence et évite les oscillations.

---

## Concepts et Applications des Réseaux de Neurones Convolutifs (CNN)

### Historique et Contexte
**Origines**
- Le réseau CNN tire son origine de recherches en biologie, notamment du système visuel humain.
- Implémentation par Yann LeCun et al. en 1998 pour classifier les ensembles de données d'images MNIST.

**Contexte**
- La classification des images est une tâche complexe.
- Les modèles traditionnels de machine learning ont des limitations.
- Une image de 128 x 128 comporte 16384 caractéristiques, et les caractéristiques n'ont pas de relation linéaire ou non linéaire évidente.

### Structure et Fonctionnement des CNN

#### Architecture des CNN
Un CNN est composé de plusieurs couches, généralement disposées de manière séquentielle :
1. **Couche d'entrée (Input Layer)** : Images (Canaux)
2. **Couches de convolution (Convolutional Layers)** : Utilisent des filtres pour extraire des caractéristiques de l'image.
3. **Couche de rectification linéaire (ReLU Layer)** : Applique la fonction ReLU pour remplacer les valeurs négatives par des zéros.
4. **Couche de pooling (Pooling Layer)** : Sous-échantillonne l'image pour réduire ses dimensions tout en conservant les caractéristiques importantes.
5. **Couche de mise à plat (Flatten Layer)** : Met à plat les données en un seul vecteur.
6. **Couches entièrement connectées (Fully Connected Layers)** : Combinent les caractéristiques pour produire la sortie finale.

#### Exemple d'Architecture CNN
1. **Étape 1** : Convolution
2. **Étape 2** : Convolution
3. **Étape 3** : Classification

### Couche de Convolution
- Utilise une matrice appelée "kernel" pour numériser une image et appliquer un filtre pour obtenir les pixels les plus importants.
- La convolution préserve la relation spatiale entre les pixels.
- Les filtres sont testés avec des valeurs différentes pendant la phase d'apprentissage. Les meilleurs sont retenus.

#### Stride et Padding
- **Stride** : Nombre de pixels utilisé pour déplacer la convolution.
- **Padding** : Ajout de pixels supplémentaires autour de l'image pour contrôler la taille de la sortie.

#### Fonctionnement Mathématique
1. Opération mathématique consistant à analyser une fonction f(x) au moyen d’une seconde fonction g(x).
2. Sortie d’une convolution consistant en une fonction de taille supérieure à celle de f et g.
3. Représentée par le symbole *.

### Couche ReLU (Rectified Linear Unit)
- Remplace les valeurs négatives par des zéros pour éliminer les caractéristiques faibles et éviter la fuite du gradient.
- Accentue la non-linéarité des images.

### Couche de Pooling
- Sous-échantillonne l'image en utilisant des filtres (kernels) pour réduire la taille de l'image tout en conservant les caractéristiques importantes.
- Types de Pooling :
  1. **Max Pooling** : Retient la valeur maximale dans la fenêtre de pooling.
  2. **Average Pooling** : Moyenne de toutes les valeurs recouvertes par la matrice.
  3. **Pooling Stochastique** : Retient une seule valeur basée sur une méthode probabiliste.

### Couche de Mise à Plat (Flatten Layer)
- Transforme les données en un vecteur unique pour les passer aux couches entièrement connectées.

### Couches Entièrement Connectées (Fully Connected Layers)
- Utilisent les valeurs du vecteur d'entrée pour effectuer des combinaisons linéaires et des fonctions d'activation afin de produire le vecteur de sortie final.

### Applications des CNN
- **Reconnaissance d'images**
- **Reconnaissance de vidéos**
- **Analyse d'images**
- **Classification d'images**
- **Systèmes de recommandation**
- **Traitement du langage naturel**
- **CNN pour le son (ex. : DeepMind's WaveNet)**

---

### Exercice Pratique avec un CNN

**Exercice : Construire un CNN avec TensorFlow**
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Charger et préparer les données MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Ajouter une dimension de canal
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Créer un modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compiler et entraîner le modèle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Évaluer le modèle
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
```

---

---

## Optimisation pour l'Apprentissage Profond

### Surapprentissage et Sous-apprentissage

**Surapprentissage (Overfitting)**
- Modèle excellent avec les données d’apprentissage mais ne se généralise pas aux nouvelles données.
- L'algorithme apprend trop des données d'entraînement au point de créer des règles qui n’existent pas dans la réalité.

**Sous-apprentissage (Underfitting)**
- Modèle médiocre avec les données d’apprentissage et ne se généralise pas non plus avec les nouvelles données.
- Ne capture pas les tendances sous-jacentes des données d'entraînement.

**Compromis Biais-Variance**
- Biais : Erreur d'approximation (données d'entraînement).
- Variance : Erreur d'estimation (données de test).
- Compromis entre la complexité du modèle et sa capacité à généraliser.

### Initialisation des Paramètres

**Instabilité des Gradients**
- **Fuite du gradient (Vanishing Gradient)** : Le gradient devient de plus en plus petit en progressant vers les couches inférieures.
- **Explosion du gradient (Exploding Gradient)** : Les gradients deviennent de plus en plus grands, causant des poids extrêmement importants et une divergence de l'algorithme.

**Solutions**
- Initialisation de Xavier et He : Utilisation de stratégies d'initialisation des poids (fan-in, fan-out) pour garantir une propagation correcte des signaux dans les deux directions.

### Optimiseurs

**Optimiseur Adam (Adaptive Moment Estimation)**
- Algorithme à taux d’apprentissage adaptatif.
- Par défaut, η = 0.001.
- Permet d'accélérer l'entraînement du réseau.

**Optimiseur avec Inertie de Nesterov (Nesterov Accelerated Gradient)**
- Utilise le gradient de la fonction de coût légèrement en avant dans le sens de l’inertie.
- Réduit les oscillations et converge plus rapidement.

### Hyperparamètres

**Définition et Importance**
- Taux d'apprentissage (Learning rate)
- Nombre d'époques (Epochs)
- Nombre de neurones et de couches cachées
- Fonction d'activation
- Régularisation
- Minibatches
- Normalisation des entrées et des cibles
- Augmentation et Déformation des données

**Optimisation des Hyperparamètres**
- Utilisation de méthodes systématiques comme la recherche par quadrillage (Grid Search) et la recherche aléatoire (Random Search).

### Régularisation

**Techniques de Régularisation**
- **Arrêt Précoce (Early Stopping)** : Interrompre l’entraînement dès que l’erreur de validation atteint le minimum.
- **Dropout** : Neutraliser aléatoirement une proportion des connexions pour obliger le réseau à mieux performer avec moins de neurones.
- **Injection de Bruit** : Ajouter des perturbations pendant l'entraînement pour rendre le réseau plus robuste aux variations imprévues.

### Apprentissage par Transfert (Transfer Learning)
- Réutiliser les couches inférieures d’un réseau de neurones existant pour accomplir une tâche comparable.
- Accélère l’entraînement et améliore les performances avec des jeux de données d’entraînement relativement petits.

### Normalisation par Lots (Batch Normalization)
- Technique pour traiter le problème de la fuite du gradient en normalisant les calculs dans les couches.
- Considéré comme un régulariseur diminuant le besoin d’autres techniques de régularisation.

### CPU et GPU pour l’Efficacité de Calcul
- Utilisation de Minibatch pour l’efficacité de calcul sur des GPUs.
- Traitement en parallèle sur les processeurs multi-cores.

### Augmentation et Déformation des Données
- Augmenter artificiellement la quantité et la variabilité des données disponibles.
- Utiliser les transformations comme les translations, rotations, et changements de luminosité pour améliorer la performance du réseau en classification.

### Stratégies de Choix des Hyperparamètres

**Stratégies**
- **Ensemble de données d'entraînement** : Choisir les hyperparamètres et tester le réseau sur les données de validation.
- **Découpage des données (k-fold cross validation)** : Répéter les expériences plusieurs fois et calculer la moyenne de l'erreur.
- **Validation Séquentielle** : Entraîner avec les données sur les années n-1 et tester pour l'année n suivante.

**Recherche par Quadrillage (Grid Search)**
- Utilisée pour trouver les hyperparamètres optimaux d’un modèle.
- Parcourt chaque combinaison de paramètres et stocke un modèle pour chaque combinaison.

**Exemple de Code pour Grid Search**
```python
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

# Charger les données
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Créer la régression logistique
logistic = linear_model.LogisticRegression()

# Créer les pénalités de régularisation
penalty = ['l1', 'l2']

# Créer les hyperparamètres de régularisation
C = np.logspace(0, 4, 10)

# Créer les options d'hyperparamètres
hyperparameters = dict(C=C, penalty=penalty)

# Créer la recherche par quadrillage avec validation croisée 5-fold
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# Adapter le modèle
best_model = clf.fit(X, y)

# Afficher les meilleurs hyperparamètres
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

# Prédire la cible
predictions = best_model.predict(X)
```

**Recherche Aléatoire (Random Search)**
- Utilise une combinaison d’hyperparamètres générée aléatoirement pour trouver la meilleure performance.
- Plus efficace que la recherche par quadrillage pour un grand nombre d’hyperparamètres.

**Exemple de Code pour Random Search**
```python
from sklearn.model_selection import RandomizedSearchCV

# Créer la recherche aléatoire avec validation croisée 5-fold
random_search = RandomizedSearchCV(logistic, hyperparameters, cv=5, n_iter=100, verbose=0, random_state=42)

# Adapter le modèle
best_model_random = random_search.fit(X, y)

# Afficher les meilleurs hyperparamètres
print('Best Penalty (Random Search):', best_model_random.best_estimator_.get_params()['penalty'])
print('Best C (Random Search):', best_model_random.best_estimator_.get_params()['C'])

# Prédire la cible
predictions_random = best_model_random.predict(X)
```

---

## Exercices Pratiques

### Exercice 1: Prédiction de Désabonnement de Clients

#### Importation des Bibliothèques et Chargement des Données
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Charger le dataset
dataset = pd.read_csv('desabonnement.csv')
dataset.head()
```

#### Préparation des Données
```python
# Séparation des variables explicatives et de la variable cible
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Division des données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Mise à l'échelle des caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

#### Création et Entraînement du Modèle
```python
# Initialisation du modèle de régression logistique
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
```

#### Prédiction et Évaluation du Modèle
```python
# Prédictions sur les données de test
y_pred = classifier.predict(X_test)

# Création de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calcul de l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### Exercice 2: Analyse des Ventes de Voitures

#### Importation des Bibliothèques et Chargement des Données
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
dataset = pd.read_csv('vente_voitures.csv')
dataset.head()
```

#### Analyse Exploratoire des Données
```python
# Afficher les informations de base sur le dataset
dataset.info()

# Afficher des statistiques descriptives
print(dataset.describe())

# Vérifier les valeurs manquantes
print(dataset.isnull().sum())
```

#### Visualisation des Données
```python
# Répartition des ventes par type de voiture
plt.figure(figsize=(10, 6))
sns.countplot(x='type', data=dataset)
plt.title('Répartition des ventes par type de voiture')
plt.show()

# Relation entre le prix et les ventes
plt.figure(figsize=(10, 6))
sns.scatterplot(x='prix', y='ventes', data=dataset)
plt.title('Relation entre le prix et les ventes')
plt.show()
```

### Exercice 3: Classification des Images de Mode (Fashion MNIST) (suite)

#### Importation des Bibliothèques et Chargement des Données
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.datasets import fashion_mnist

# Charger les données
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalisation des données
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape des données pour le modèle CNN
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
```

#### Création et Entraînement du Modèle CNN
```python
# Création du modèle
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
model.fit(x_train, y_train, batch_size=100, epochs=10, verbose=1)
```

#### Évaluation du Modèle
```python
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Pourcentage de bien classées : {score[1]}")
```

**Augmentation des Données avec ImageDataGenerator**
```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('../dataset/train', target_size=(150, 150), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('../dataset/validation', target_size=(150, 150), batch_size=32, class_mode='binary')

model.fit_generator(train_generator, epochs=50, validation_data=validation_generator)
```

### Exercice 4: Prédiction des Prix des Maisons

#### Importation des Bibliothèques et Chargement des Données
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Charger le dataset
dataset = pd.read_csv('housing.csv')
dataset.head()
```

#### Préparation des Données
```python
# Séparation des variables explicatives et de la variable cible
X = dataset.drop('price', axis=1).values
y = dataset['price'].values

# Division des données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

#### Création et Entraînement du Modèle de Régression Linéaire
```python
# Initialisation du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)
```

#### Prédiction et Évaluation du Modèle
```python
# Prédictions sur les données de test
y_pred = model.predict(X_test)

# Calcul de l'erreur quadratique moyenne
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

### Exercice 5: Analyse des Performances des Étudiants

#### Importation des Bibliothèques et Chargement des Données
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le dataset
dataset = pd.read_csv('student_performance.csv')
dataset.head()
```

#### Analyse Exploratoire des Données
```python
# Afficher les informations de base sur le dataset
dataset.info()

# Afficher des statistiques descriptives
print(dataset.describe())

# Vérifier les valeurs manquantes
print(dataset.isnull().sum())
```

#### Visualisation des Données
```python
# Répartition des notes par matière
plt.figure(figsize=(10, 6))
sns.countplot(x='subject', data=dataset)
plt.title('Répartition des notes par matière')
plt.show()

# Relation entre les heures d'étude et les notes
plt.figure(figsize=(10, 6))
sns.scatterplot(x='study_hours', y='score', data=dataset)
plt.title('Relation entre les heures d\'étude et les notes')
plt.show()
```

---

## Analyse des Données de Désabonnement

### Étapes de l'Analyse

1. **Importation des Bibliothèques**
2. **Chargement et Inspection des Données**
3. **Préparation des Données**
4. **Analyse Exploratoire des Données**
5. **Modélisation et Prédiction**
6. **Évaluation des Modèles**

### Importation des Bibliothèques
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
```

### Chargement et Inspection des Données
```python
# Charger le dataset
dataset = pd.read_csv('desabonnement.csv')

# Afficher les premières lignes du dataset
print(dataset.head())

# Afficher des informations sur le dataset
print(dataset.info())

# Vérifier les valeurs manquantes
print(dataset.isnull().sum())
```

### Préparation des Données
```python
# Encoder les variables catégorielles si nécessaire
labelencoder = LabelEncoder()
dataset['Gender'] = labelencoder.fit_transform(dataset['Gender'])

# Séparation des variables explicatives et de la variable cible
X = dataset.drop('Exited', axis=1).values
y = dataset['Exited'].values

# Division des données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Mise à l'échelle des caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Analyse Exploratoire des Données
```python
# Afficher des statistiques descriptives
print(dataset.describe())

# Répartition des clients par genre
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', data=dataset)
plt.title('Répartition des clients par genre')
plt.show()

# Répartition des désabonnements
plt.figure(figsize=(10, 6))
sns.countplot(x='Exited', data=dataset)
plt.title('Répartition des désabonnements')
plt.show()

# Analyse de la corrélation entre les variables
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
plt.title('Corrélation entre les variables')
plt.show()
```

### Modélisation et Prédiction

#### Régression Logistique
```python
# Initialisation du modèle de régression logistique
logreg = LogisticRegression(random_state=0)
logreg.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred = logreg.predict(X_test)

# Évaluation du modèle
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
```

#### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

# Initialisation du modèle Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred_rf = rf.predict(X_test)

# Évaluation du modèle
print("Confusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))
print("Accuracy (Random Forest):", accuracy_score(y_test, y_pred_rf))
print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))
```

### Évaluation des Modèles
```python
# Comparaison des modèles
models = ['Logistic Regression', 'Random Forest']
accuracy_scores = [accuracy_score(y_test, y_pred), accuracy_score(y_test, y_pred_rf)]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracy_scores)
plt.title('Comparison of Model Accuracy')
plt.ylabel('Accuracy Score')
plt.show()
```

Ce code permet d'analyser et de modéliser les données du fichier desabonnement.csv. Il inclut l'importation des données, leur préparation, une analyse exploratoire et la modélisation à l'aide de la régression logistique et du Random Forest. L'évaluation des modèles permet de comparer leurs performances.

---

Ce cours couvre de manière exhaustive les concepts et les pratiques de l'intelligence artificielle et de l'apprentissage machine, avec une attention particulière aux réseaux de neurones convolutifs (CNN), à l'optimisation pour l'apprentissage profond, et à
