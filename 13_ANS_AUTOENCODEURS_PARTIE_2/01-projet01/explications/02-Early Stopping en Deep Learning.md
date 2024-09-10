-----
# **Early Stopping en Deep Learning :**
---------

L'**early stopping** est une technique utilisée pour prévenir le surapprentissage (ou **overfitting**) dans les réseaux de neurones en deep learning. Le surapprentissage survient lorsque le modèle s'ajuste tellement aux données d'entraînement qu'il perd sa capacité à généraliser sur des données nouvelles. Voici comment l'early stopping fonctionne :

1. **Division des données :** Les données sont souvent divisées en trois parties : les données d'entraînement, les données de validation et les données de test.
2. **Suivi des performances :** Pendant l'entraînement, le modèle est évalué périodiquement sur les données de validation. Si la performance sur ces données s'améliore, le modèle continue à s'entraîner. 
3. **Arrêt anticipé :** Si, après un certain nombre d'itérations (ou d'époques), la performance sur les données de validation cesse de s'améliorer ou commence à se dégrader, cela signifie que le modèle est probablement en train de surapprendre. L'entraînement est alors arrêté même si le nombre d'époques spécifié n'a pas été atteint. 

L'idée est d'arrêter l'entraînement avant que le modèle ne devienne trop spécialisé sur les données d'entraînement, améliorant ainsi sa capacité à généraliser.

**LambdaCall :**

- Le terme **LambdaCall** fait souvent référence à une approche spécifique utilisée dans certains frameworks de deep learning pour exécuter des **callbacks** ou des fonctions durant l'entraînement d'un modèle. Un **callback** est une fonction qui peut être appelée à des moments spécifiques du cycle d'entraînement, par exemple à la fin de chaque époque ou après un certain nombre d'itérations. LambdaCall pourrait désigner l'utilisation d'une fonction lambda (une fonction anonyme, légère) pour effectuer une action dans ce cadre.
- En Python et Keras, par exemple, un callback peut être utilisé pour ajuster dynamiquement certains paramètres comme le taux d'apprentissage ou même implémenter des stratégies d'**early stopping**.
- Cela te permet d’intégrer des opérations personnalisées lors de l’entraînement du modèle, par exemple pour sauvegarder le modèle ou surveiller des métriques spécifiques.

-----
# **Pour résumer :**
-----

- **Early Stopping** : Technique pour arrêter l'entraînement d'un modèle lorsque les performances sur les données de validation cessent de s'améliorer, afin d'éviter le surapprentissage.
- **LambdaCall** : Fonction lambda utilisée comme callback pendant l'entraînement pour exécuter des actions spécifiques, comme ajuster des paramètres ou surveiller des métriques.


