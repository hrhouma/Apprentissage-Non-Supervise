Cette représentation montre les étapes essentielles pour :
1. Charger et préparer les données,
2. Construire et entraîner un autoencodeur,
3. Extraire les **embeddings** et les classer avec un **SVM**,
4. Visualiser les résultats avec **t-SNE**,
5. Conclure sur les performances du modèle.

```
+------------------------------------------------------------+
|               Développement d'un autoencodeur              |
|         pour classification d'images avec un SVM           |
+------------------------------------------------------------+
               |
               |
               v
+---------------------------+
|  Étape 1 : Préparation     |
|    - Chargement des images |
|    - Normalisation des     |
|      pixels (0-255 -> 0-1) |
|    - Division en ensemble  |
|      d'entraînement et     |
|      de validation         |
+---------------------------+
               |
               |
               v
+-------------------------------------------------+
|  Étape 2 : Conception de l'autoencodeur         |
|    - Architecture de l'encodeur                |
|        > Conv2D                                |
|        > MaxPooling2D                          |
|    - Architecture du décodeur                  |
|        > Conv2D                                |
|        > UpSampling2D                          |
|    - Compilation avec l'optimiseur Adam        |
+-------------------------------------------------+
               |
               |
               v
+-----------------------------------------------+
|  Étape 3 : Entraînement de l'autoencodeur      |
|    - Utilisation des données d'entraînement    |
|    - Entraînement sur plusieurs époques        |
|    - Validation pendant l'entraînement         |
+-----------------------------------------------+
               |
               |
               v
+---------------------------------------------------+
|  Étape 4 : Évaluation                             |
|    - Reconstruction des images originales         |
|    - Comparaison avec les images d'origine        |
+---------------------------------------------------+
               |
               |
               v
+--------------------------------------------+
|  Étape 5 : Extraction des embeddings       |
|    - Utilisation de l'encodeur pour        |
|      compresser les images                 |
|    - Sortie sous forme d'embeddings        |
+--------------------------------------------+
               |
               |
               v
+-------------------------------------------+
|  Étape 6 : Application du SVM             |
|    - Entrainer le SVM sur les embeddings  |
|    - Utiliser pour classifier les images  |
+-------------------------------------------+
               |
               |
               v
+-----------------------------------------------+
|  Étape 7 : Visualisation avec t-SNE            |
|    - Réduction des dimensions des embeddings   |
|    - Visualisation des points dans un plot 2D  |
+-----------------------------------------------+
               |
               |
               v
+---------------------------------------------+
|  Étape 8 : Conclusion                       |
|    - Analyse des résultats de classification|
|    - Réflexion sur l'architecture choisie   |
|    - Suggestions pour améliorer le modèle   |
+---------------------------------------------+
```


