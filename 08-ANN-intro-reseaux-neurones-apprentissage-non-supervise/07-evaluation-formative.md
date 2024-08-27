### Examen : Analyse et Compréhension de Codes TensorFlow

**Instructions :**
1. Allez sur le lien suivant pour accéder aux scripts de code que vous allez analyser : [Code à analyser](https://drive.google.com/drive/folders/1HvqAzEGI01-Hbex0mBwhYm4i4PzYn4pq?usp=sharing).
2. Analysez chaque script en détail .
3. Répondez aux questions pour chaque script en utilisant à la fois votre compréhension du code et les ressources théoriques nécessaires.
4. Vos réponses doivent être claires, précises et bien argumentées.


# Code à analyser : 
- https://drive.google.com/drive/folders/1HvqAzEGI01-Hbex0mBwhYm4i4PzYn4pq?usp=sharing
  
---

# 1. 1-tensorflow_bases.py
- **Question 1** : Quelle est la différence entre un tableau NumPy et un tenseur TensorFlow, et pourquoi serait-il avantageux de convertir l'un en l'autre ?
- **Question 2** : Expliquez l'utilité de la commande os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' dans ce script.
- **Question 3** : Comment et pourquoi est-il utile d'ajouter 100 à chaque élément d'une matrice TensorFlow ? Comparez les méthodes présentées dans le code commenté.

---

# 2. 2-tensorflow_operations.py
- **Question 1** : Quelle est l'importance de la transposition d'un tenseur dans TensorFlow, et comment cela se reflète-t-il dans la matrice transposée tensor_11 ?
- **Question 2** : Pourquoi utiliser une fonction telle que tf.cast pour changer le type de données d'un tenseur avant d'effectuer une opération mathématique ?
- **Question 3** : Décrivez comment TensorFlow génère des valeurs aléatoires pour des tensors, et discutez de l'importance de l'initialisation aléatoire.

---

# 3. 3-load_data.py
- **Question 1** : Pourquoi est-il nécessaire de normaliser les données d'entraînement (x_train, x_validation) avant de les utiliser dans un modèle ?
- **Question 2** : Expliquez le processus de séparation des données en ensembles de formation et de validation, et pourquoi cette étape est cruciale pour la formation d'un modèle de deep learning.
- **Question 3** : Quelles informations pouvez-vous obtenir en affichant les cinq premières images de l'ensemble de données Fashion MNIST, et comment ces informations peuvent-elles être utilisées pour diagnostiquer des problèmes potentiels dans les données ?

---

# 4. 4-mnist_model.py
- **Question 1** : Pourquoi le modèle séquentiel est-il approprié pour ce type de données, et quelles sont les principales caractéristiques de ce modèle dans ce script ?
- **Question 2** : Décrivez le rôle de la fonction d'activation ReLU dans les couches denses du modèle. Pourquoi est-elle utilisée ici plutôt qu'une autre fonction d'activation ?
- **Question 3** : Que se passe-t-il lors de l'appel à model.summary(), et pourquoi est-il important de comprendre la structure du modèle avant de l'entraîner ?

---

# 5. 5-layer_infos.py
- **Question 1** : Pourquoi pourriez-vous vouloir initialiser manuellement les poids et les biais d'une couche dense, comme montré dans ce script ?
- **Question 2** : Comment pouvez-vous accéder aux poids et aux biais d'une couche spécifique après l'entraînement du modèle, et que pouvez-vous en apprendre ?
- **Question 3** : Comparez l'initialisation aléatoire des poids avec l'initialisation manuelle des poids dans ce script. Quels sont les avantages et les inconvénients de chaque approche ?

---

# 6. 6-model_prediction.py
- **Question 1** : Expliquez le processus de compilation d'un modèle dans TensorFlow. Quels sont les rôles respectifs de la fonction de perte, de l'optimiseur, et des métriques ?
- **Question 2** : Quelle est la signification des prédictions de probabilités de classes produites par le modèle, et comment interprétez-vous les prédictions par rapport aux vraies classes ?
- **Question 3** : Pourquoi est-il important de valider un modèle sur des données qui n'ont pas été utilisées lors de l'entraînement ?

---

# 7. 7-callbacks.py
- **Question 1** : Expliquez le rôle du ModelCheckpoint dans l'entraînement du modèle. Pourquoi est-il utile d'enregistrer uniquement le meilleur modèle ?
- **Question 2** : Pourquoi la conversion des étiquettes en vecteurs one-hot est-elle nécessaire avant d'entraîner un modèle avec categorical_crossentropy ?
- **Question 3** : Comment le shuffle=True affecte-t-il le processus d'entraînement du modèle, et pourquoi est-il souvent recommandé ?

---

# 8. 8-early_stopping.py
- **Question 1** : Décrivez l'utilité de l'arrêt anticipé (EarlyStopping) dans l'entraînement d'un modèle. Quels problèmes cela permet-il de prévenir ?
- **Question 2** : Que se passe-t-il si l'on continue d'entraîner un modèle au-delà du point optimal de validation ? Comment l'arrêt anticipé peut-il éviter ce problème ?
- **Question 3** : Expliquez le processus de sauvegarde d'un modèle avec model.save. Pourquoi est-il important de sauvegarder le modèle une fois l'entraînement terminé ?

---

# 9. 9-save_model.py
- **Question 1** : Quels sont les avantages d'enregistrer un modèle Keras après l'entraînement, et comment ce modèle peut-il être utilisé ultérieurement ?
- **Question 2** : Dans quel scénario pourriez-vous vouloir charger un modèle précédemment sauvegardé plutôt que d'entraîner un nouveau modèle ?
- **Question 3** : Comparez l'approche de sauvegarde de modèle dans ce script avec d'autres méthodes possibles, comme la sérialisation des poids seulement.

---

# 10. 10-hyperparametres.py
- **Question 1** : Expliquez comment GridSearchCV est utilisé dans ce script pour optimiser les hyperparamètres d'un modèle de réseau de neurones.
- **Question 2** : Pourquoi est-il important d'ajuster les hyperparamètres d'un modèle, et quels sont les risques si cette étape est négligée ?
- **Question 3** : Décrivez les avantages de l'utilisation d'un wrapper Keras (KerasClassifier) avec scikit-learn pour effectuer une recherche sur grille.
