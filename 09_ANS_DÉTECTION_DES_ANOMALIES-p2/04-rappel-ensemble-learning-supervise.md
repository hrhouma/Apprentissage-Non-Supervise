# Rappel - ensemble learning en  machine learning supervisé

L'ensemble learning (apprentissage par ensemble) est principalement une technique de machine learning supervisé, bien qu'il puisse également être adapté pour des approches non supervisées dans certains cas.

En machine learning supervisé, les méthodes d'ensemble combinent plusieurs modèles pour améliorer les performances globales. Voici quelques exemples populaires d'ensemble learning supervisé :

1. **Bagging (Bootstrap Aggregating)** :
   - Méthode : Utilise plusieurs échantillons aléatoires avec remplacement pour entraîner des modèles distincts.
   - Exemple : Random Forest, qui utilise des arbres de décision.

2. **Boosting** :
   - Méthode : Construit des modèles séquentiels où chaque modèle corrige les erreurs du précédent.
   - Exemple : AdaBoost, Gradient Boosting Machines (GBM), XGBoost.

3. **Stacking** :
   - Méthode : Combine les prédictions de plusieurs modèles de base à l'aide d'un modèle de niveau supérieur (métamodèle).
   - Exemple : Utiliser des régressions linéaires, SVMs, et des réseaux neuronaux comme modèles de base, et un autre modèle pour combiner leurs prédictions.

Cependant, il existe également des méthodes d'ensemble pour l'apprentissage non supervisé. 
- Par exemple, dans le clustering (regroupement) non supervisé, différentes méthodes de clustering peuvent être combinées pour obtenir des clusters plus stables et robustes.
- Un exemple d'ensemble learning non supervisé est l'utilisation de plusieurs algorithmes de clustering (comme K-means, DBSCAN, et l'agglomération hiérarchique) pour obtenir une meilleure compréhension des structures de données complexes en combinant les résultats des différents algorithmes.
- En résumé, l'ensemble learning est principalement utilisé en apprentissage supervisé pour améliorer la précision et la robustesse des modèles, mais il peut également être appliqué dans des contextes non supervisés, bien que cela soit moins courant.
