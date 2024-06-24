- YOUTUBE : ==> https://www.youtube.com/watch?v=FTtzd31IAOw&t=1065s&ab_channel=MachineLearnia
- CODE : ==> https://github.com/MachineLearnia/Python-Machine-Learning/blob/master/24%20-%20Sklearn%20%3A%20Apprentissage%20Non-supervis%C3%A9.ipynb
- TUTORIAL: 
# Introduction: 

- Bienvenue dans cette vingt-quatrième vidéo de la série Python spéciale machine learning. 
- Aujourd'hui, nous allons découvrir les bases de l'apprentissage non supervisé, une des branches importantes du machine learning et du deep learning. 
- Nous aborderons trois applications principales : le clustering, la détection d'anomalies, et la réduction de dimension. 
- À la fin de cette vidéo, vous serez capable d'écrire vos propres programmes d'apprentissage non supervisé avec Scikit-learn. 
- Prêts ? C'est parti !

# Qu'est-ce que l'apprentissage non supervisé ?

- L'apprentissage non supervisé est une méthode d'apprentissage où, au lieu de montrer à la machine des exemples de ce qu'elle doit apprendre, on lui fournit uniquement des données et on lui demande d'analyser leur structure pour apprendre à réaliser certaines tâches. Par exemple, la machine peut apprendre à classer des données en les regroupant selon leur ressemblance. C'est ce qu'on appelle le clustering ou la classification non supervisée. Cette technique permet de réaliser diverses tâches comme classer des documents, des photos, des tweets, segmenter la clientèle d'une entreprise, etc. Nous verrons comment faire cela avec l'algorithme K-means clustering.

### Application 1 : Clustering

- Le clustering permet à la machine de classer nos données selon leur ressemblance. Un des algorithmes les plus populaires pour cette tâche est le K-means clustering.
- Le principe est simple : on commence par placer un certain nombre de points appelés centroids, au hasard parmi les données.
- Ensuite, on affecte chaque point du dataset au centroid le plus proche, formant ainsi des clusters.
- On déplace ensuite chaque centroid au milieu de son cluster, et on répète le processus jusqu'à ce que les centroids convergent vers une position d'équilibre.
- L'algorithme de K-means clustering fonctionne donc de manière itérative en deux étapes : affectation des points aux centroids les plus proches, puis recalcul de la moyenne de chaque cluster pour déplacer les centroids.
- Pour implémenter cela avec Scikit-learn, il faut importer `KMeans` depuis le module `cluster`. Ensuite, on crée un modèle en spécifiant le nombre de clusters souhaités et on entraîne ce modèle avec la méthode `fit`. Après l'entraînement, on peut visualiser les clusters avec Matplotlib. L'algorithme de K-means cherche à minimiser une fonction appelée inertie, représentant la somme des distances entre les points des clusters et leurs centroids.

### Application 2 : Détection d'anomalies

- La détection d'anomalies consiste à identifier les échantillons dont les caractéristiques sont très éloignées de celles des autres échantillons.
- Cela permet de développer des systèmes de sécurité, de détection de fraude bancaire, de détection de défaillances dans une usine, etc.
- Un algorithme efficace pour cette tâche est l'Isolation Forest. Le principe est d'effectuer une série de découpes aléatoires dans les données et de compter le nombre de découpes nécessaires pour isoler un échantillon.
- Plus ce nombre est petit, plus il est probable que l'échantillon soit une anomalie.
- Pour utiliser l'Isolation Forest avec Scikit-learn, il suffit d'importer `IsolationForest` depuis le module `ensemble` et de préciser le taux de contamination souhaité.
- Ensuite, on entraîne le modèle avec la méthode `fit` et on identifie les anomalies avec la méthode `predict`.

### Application 3 : Réduction de dimension

- La réduction de dimension permet de simplifier la structure des données tout en conservant les principales informations.
- Un des algorithmes les plus populaires pour cette tâche est l'Analyse en Composantes Principales (PCA).
- Le principe est de projeter les données sur des axes appelés composantes principales, en minimisant la distance entre les points et leur projection, tout en maximisant la variance conservée.
- Pour implémenter PCA avec Scikit-learn, il faut importer `PCA` depuis le module `decomposition`, spécifier le nombre de dimensions souhaitées, et transformer les données avec la méthode `fit_transform`.
- Pour choisir le nombre optimal de composantes, on peut examiner le pourcentage de variance préservée par chaque composante et choisir de conserver 95 à 99 % de la variance initiale.

### Conclusion

- Vous savez maintenant comment utiliser les techniques d'apprentissage non supervisé pour le clustering, la détection d'anomalies et la réduction de dimension.
- Ces outils sont essentiels pour analyser et interpréter des données complexes.
- N'oubliez pas de standardiser vos données avant de les utiliser dans PCA et de considérer les méthodes de machine learning adaptées à la nature de vos données.
- Merci de votre attention et à bientôt pour une nouvelle vidéo sur le machine learning !

---

Comme d'habitude, merci de vous abonner si cette vidéo vous a plu, de partager vos questions dans les commentaires, et de rejoindre notre communauté sur Discord. Si vous souhaitez me soutenir, n'hésitez pas à visiter ma page Tipeee. Portez-vous bien et à très vite pour la prochaine vidéo !
