# Exemple : Modèle avec biais défini manuellement

- https://drive.google.com/drive/folders/1v-8LrdbyBSP7jhSJMLHguULbwfYVLNnS?usp=sharing
  
```python
import tensorflow as tf
import numpy as np

# Générer des données d'exemple
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]])  # y = 2 * X + 1

# Définir un modèle simple avec un biais initialisé manuellement
class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomDenseLayer, self).__init__()
        self.w = self.add_weight(shape=(1, 1), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(0.5), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

model = tf.keras.Sequential([
    CustomDenseLayer()
])

# Compiler le modèle
model.compile(optimizer='sgd', loss='mean_squared_error')

# Afficher les poids et le biais avant l'entraînement
initial_weights, initial_bias = model.layers[0].get_weights()
print(f"Poids initiaux : {initial_weights}")
print(f"Biais initial : {initial_bias}")

# Entraîner le modèle
model.fit(X, y, epochs=500, verbose=0)

# Afficher les poids et le biais après l'entraînement
final_weights, final_bias = model.layers[0].get_weights()
print(f"Poids finaux : {final_weights}")
print(f"Biais final : {final_bias}")

# Faire une prédiction
new_input = np.array([[6.0]])
prediction = model.predict(new_input)
print(f"Prédiction pour une entrée de 6.0 : {prediction}")
```

### Explication

- **CustomDenseLayer** : Nous avons défini une couche dense personnalisée où le biais est initialisé manuellement à 0,5 en utilisant `tf.keras.initializers.Constant(0.5)`. Cela permet de spécifier la valeur initiale du biais au lieu de laisser TensorFlow l'initialiser de manière aléatoire.
- **Entraînement** : Le modèle est entraîné pour ajuster à la fois les poids et le biais afin de minimiser la perte (erreur quadratique moyenne).
- **Prédiction** : Après l'entraînement, nous faisons une prédiction pour une nouvelle entrée.

### Résultats attendus

- **Biais initial** : Le biais est initialisé à 0,5 avant l'entraînement.
- **Biais final** : Après l'entraînement, le biais devrait se rapprocher de 1, qui est la valeur idéale pour l'équation

$$
y = 2 \times X + 1
$$

- **Prédiction** : La prédiction pour une entrée de 6.0 devrait être proche de 13.0 (soit

$$
2 \times 6 + 1
$$

après l'entraînement.

