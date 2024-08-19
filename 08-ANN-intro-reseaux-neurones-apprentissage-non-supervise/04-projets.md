<hr/> 
<hr/> 
<hr/> 

# Codes et projets 

<hr/> 
<hr/> 
<hr/> 

- https://drive.google.com/drive/folders/1F1XjH8OrQqYDaMZ7_y56zuqxrAnAj6I3?usp=sharing
- https://drive.google.com/drive/folders/1NSIaTpnrByyvy_LXKrmjDwJgx538R09g?usp=sharing
- https://drive.google.com/drive/folders/1Eaf2-q8CcdLu6oENHJb30bnePlRNRnhu?usp=sharing
- https://drive.google.com/drive/folders/1aydRuYvHqyBJ2fkYA6nR1TvZvjEcaKSk?usp=sharing
- https://drive.google.com/drive/folders/1E8i2CdvHVD473TqGRGDZ0NzHeVu3-SHn?usp=sharing

---

# Exemple: 

---

```
Input Layer (Flatten, 28x28)
+----------------------+
|      28x28 Image      |
+----------------------+
              |
              V
Flatten Layer (784 neurons)
+----------------------------+
|        784 Neurons          |
+----------------------------+
              |
              V
Dense Layer (300 neurons, ReLU)
+----------------------------+
|      300 Neurons, ReLU      |
+----------------------------+
              |
              V
Dense Layer (150 neurons, ReLU)
+----------------------------+
|      150 Neurons, ReLU      |
+----------------------------+
              |
              V
Dense Layer (10 neurons, Softmax)
+----------------------------+
|    10 Neurons, Softmax      |
+----------------------------+
              |
              V
Output Layer (10 classes)
+------------------------------+
|  T-shirt/top  |  Trouser      |
|  Pullover     |  Dress        |
|  Coat         |  Sandal       |
|  Shirt        |  Sneaker      |
|  Bag          |  Ankle boot   |
+------------------------------+
```

### Explication du Schéma :

1. **Input Layer (Flatten, 28x28)** :
   - Représente une image d'entrée de taille 28x28 pixels.

2. **Flatten Layer (784 neurons)** :
   - La couche `Flatten` prend l'image 28x28 et la transforme en un vecteur de 784 neurones.

3. **Dense Layer (300 neurons, ReLU)** :
   - Une couche dense avec 300 neurones utilisant la fonction d'activation ReLU. C'est ici que se fait une grande partie de l'apprentissage initial.

4. **Dense Layer (150 neurons, ReLU)** :
   - Une seconde couche dense, avec 150 neurones, qui continue à apprendre les caractéristiques des données tout en réduisant progressivement la complexité.

5. **Dense Layer (10 neurons, Softmax)** :
   - La couche finale a 10 neurones, correspondant aux 10 classes du dataset Fashion MNIST, avec une fonction d'activation Softmax qui transforme les sorties en probabilités.

6. **Output Layer (10 classes)** :
   - La couche de sortie donne la probabilité d'appartenance de l'image à l'une des 10 classes, comme "T-shirt/top", "Trouser", etc.

Ce schéma montre la structure du réseau de neurones utilisé dans le code, depuis l'entrée jusqu'à la sortie, et comment chaque couche contribue au processus de classification dans le projet suivant : 

3 - fashion_mnist-v2.ipynb

- https://drive.google.com/drive/folders/1NSIaTpnrByyvy_LXKrmjDwJgx538R09g?usp=sharing
