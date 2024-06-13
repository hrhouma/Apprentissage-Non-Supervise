# Partie 1 - Clustering
# 1 - Introduction au clustering :
- Le clustering est une technique d'apprentissage non supervisé qui consiste à regrouper des données en clusters ou groupes en fonction de leur ressemblance. Il permet de découvrir des structures cachées dans les données et d'organiser ces données de manière significative.

# 2 - Algorithme K-Means Clustering :
  - **Fonctionnement du K-Means :** L'algorithme K-Means est l'un des algorithmes de clustering les plus populaires. Il fonctionne en deux étapes itératives : dans la première étape, chaque point de données est affecté au centroïde le plus proche, et dans la deuxième étape, les centroïdes sont recalculés comme la moyenne des points de données de chaque cluster. Ces étapes sont répétées jusqu'à ce que les centroïdes ne changent plus.
  - **Étapes de l'algorithme :** 
    1. Initialisation des centroïdes de manière aléatoire.
    2. Affectation de chaque point de données au centroïde le plus proche.
    3. Recalibrage des centroïdes en calculant la moyenne des points de chaque cluster.
    4. Répétition des étapes 2 et 3 jusqu'à convergence.

  - **Exemple pratique avec Scikit-Learn :** Implémentation de l'algorithme K-Means en utilisant la bibliothèque Scikit-Learn, comprenant l'importation des modules nécessaires, la création et l'entraînement du modèle, et la visualisation des résultats obtenus.

# 3 - Méthodes avancées de clustering :
  - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise) :** Une méthode de clustering qui identifie les clusters en fonction de la densité des points de données. Elle est particulièrement utile pour détecter les clusters de forme arbitraire et les points de données anormaux (outliers).
  - **Agglomerative Clustering :** Une méthode hiérarchique de clustering qui fusionne progressivement les points de données en clusters plus grands en fonction de leur similarité.

# 4 - Cas d'utilisation et visualisation des résultats :

- Présentation de différents cas d'utilisation du clustering, tels que la segmentation de clients ou le classement de documents. Utilisation de bibliothèques de visualisation comme Matplotlib pour représenter graphiquement les clusters et analyser les résultats obtenus.
