# Référence : 
- https://stackoverflow.com/questions/12893492/choosing-eps-and-minpts-for-dbscan-r
- En clustering, notamment avec DBSCAN (Density-Based Spatial Clustering of Applications with Noise), le choix des paramètres `minPts` (points minimum) et `epsilon` (ε) est crucial mais pas simple. Voici une explication plus détaillée de ces paramètres et quelques conseils pour les choisir :

### minPts (Points Minimum)
- **Définition :** Le nombre minimum de points pour former une région dense.
- **Impact :** Une valeur de `minPts` basse peut entraîner la création de clusters à partir de bruit, augmentant le nombre de petits clusters potentiellement insignifiants.
- **Conseil :** `minPts` doit être choisi en fonction de la dimensionnalité des données. Une heuristique courante est de fixer `minPts` à au moins la dimensionnalité du jeu de données plus un (par exemple, pour un jeu de données en 2D, `minPts` pourrait être 3 ou 4).

### Epsilon (ε)
- **Définition :** La distance maximale entre deux points pour qu'ils soient considérés comme voisins.
- **Impact :** Le choix de `ε` dépend de plusieurs facteurs comme le jeu de données, `minPts`, la fonction de distance utilisée et la normalisation des données. Un `ε` trop petit peut entraîner un grand nombre de clusters petits et épars, tandis qu'un `ε` trop grand peut entraîner la fusion de clusters distincts en un seul.
- **Conseil :** Vous pouvez essayer de créer un histogramme des distances `k`-plus proches voisins (kNN) et choisir un "coude" dans le graphique. Cependant, il se peut qu'il n'y ait pas de coude visible ou qu'il y en ait plusieurs.

### OPTICS (Ordering Points To Identify the Clustering Structure)
- **Description :** OPTICS est un successeur de DBSCAN qui ne nécessite pas le paramètre `ε` (sauf pour des raisons de performance avec le support des index, voir Wikipedia). C'est une méthode plus flexible, mais sa mise en œuvre peut être complexe, surtout dans des environnements comme R qui privilégient les opérations matricielles.
- **Fonctionnement :** On peut imaginer OPTICS comme testant toutes les valeurs de `ε` en même temps et plaçant les résultats dans une hiérarchie de clusters.

### Vérification Initiale
- **Importance de la Fonction de Distance et de la Normalisation :** Indépendamment de l'algorithme de clustering que vous utilisez, il est essentiel d'avoir une fonction de distance utile et une normalisation appropriée des données. Si votre distance dégénère, aucun algorithme de clustering ne fonctionnera correctement.

En résumé, le choix des paramètres dans DBSCAN est une tâche empirique qui nécessite des essais et des ajustements en fonction des caractéristiques spécifiques de vos données.
