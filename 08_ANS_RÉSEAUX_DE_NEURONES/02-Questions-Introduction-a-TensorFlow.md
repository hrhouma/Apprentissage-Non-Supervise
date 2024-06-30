# 1 - Cours : Introduction à TensorFlow

# Questions

# 1. Concept de division des données en données d'entraînement et de test

1.1. Qu'est-ce que l'entraînement et le test ?

# 2. Différence entre les `x` et les `y` dans la ligne du code ci-haut

2.1. Expliquez la différence entre les `x` et les `y` dans cette ligne de code.

```python
  (all_x_train, all_y_train), (x_test, y_test) = fashion_mnist_data.load_data()
  ``

# 3. Types de données : différence entre `float32` et `float64`

3.1. Quelle est la différence entre `float32` et `float64` ? [Référence](https://stackoverflow.com/questions/43440821/the-real-difference-between-float32-and-float64)

# 4. Concept des variables catégoriques

4.1. Qu'est-ce qu'une variable catégorique et pourquoi est-elle importante en IA ?

4.2. Qu'est-ce que le one-hot encoding ?

#### 5. Données de validation

5.1. Qu'est-ce que les données de validation ? Est-ce la même chose que les données de test ?

#### 6. Normalisation en IA

6.1. Qu'est-ce que la normalisation en intelligence artificielle ?

#### 7. Considérez le code suivant :

```python
fashion_mnist_class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                             "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Création du modèle séquentiel
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(150, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
```

7.1. À l’origine, quel est le type de `y_train` et `y_test` ?

7.2. Pour l'algorithme, est-il mieux d’avoir un type de données numérique ou chaîne de caractères ? Pourquoi ?

7.3. Pour les humains, est-il mieux d’avoir un type de données numérique ou chaîne de caractères ? Pourquoi ?

#### 8. Considérez le code suivant :

```python
# Création du modèle séquentiel
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(150, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
```

##### Partie 1 - Questions simples :

8.1. Qu'est-ce qu'un modèle Sequential ?

8.2. Qu'est-ce que la couche Flatten ? Expliquez par un exemple.

8.3. Qu'est-ce que la couche de sortie ? Pourquoi avons-nous 10 neurones dans la couche de sortie ?

8.4. Dessinez cette architecture manuellement.

##### Partie 2 - Questions intermédiaires :

8.5. Qu'est-ce qu'une couche Dense ?

8.6. Qu'est-ce qu'une fonction d'activation ? Pourquoi utiliser `relu` et `softmax` pour la sortie ?

8.7. Pourquoi avons-nous choisi 300 neurones et 150 neurones ? Pourquoi avons-nous choisi 2 couches ? (Hyperparamètres)

##### Partie 3 - Autres questions :

8.8. Utilisez `model.summary()` pour décrire votre architecture du modèle programmé.

8.9. Qu'est-ce que le surapprentissage et le sous-apprentissage ?

#### 9. Expliquez la ligne de code `model.compile`

9.1. Qu'est-ce que les paramètres `loss`, `optimizer` et `metrics` ?

9.2. Explorez les différentes valeurs du paramètre `loss` que nous pouvons utiliser.

9.3. Explorez les différentes valeurs du paramètre `optimizer` que nous pouvons utiliser.

9.4. Explorez les différentes valeurs du paramètre `metrics` que nous pouvons utiliser.

9.5. Est-ce que `accuracy` et `précision` sont la même chose ?

#### 10. Expliquez la ligne de code `model.fit`

10.1. Quels sont les paramètres de la méthode `fit()` ?

10.2. Quels sont les paramètres `epoch` et `validation` ?

10.3. Pouvons-nous remplacer `validation_data` par des données de test ?

#### Partie optionnelle :

11. Selon votre compréhension, illustrez la différence entre ces deux concepts : (1) `accuracy` d'entraînement et (2) `accuracy` de validation.

12. Qu'est-ce que la validation croisée ? À quoi sert-elle ?

13. Comment améliorer les résultats ? 

14. Quels sont les différents scores qu’on applique en intelligence artificielle et à quoi servent ces scores ? Donnez un tableau détaillé de chaque score.



