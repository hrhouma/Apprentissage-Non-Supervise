# OPTIMISATION DBSCAN-eps-min_samples
----
# Partie 1 - Tester dans Colab le Notebbok 02_OPTIMISATION_DBSCAN_02.ipynb
# Lien des fichiers : 

- https://drive.google.com/drive/folders/1Oe9DgDq_64ZlibnWSz1PM2udQ5VtKTdb?usp=sharing
![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/8f2d0525-527f-4ffa-956f-5fbb2fec49ac)

## Objectif du Programme

L'objectif de ce programme est de comprendre et d'optimiser l'utilisation de l'algorithme de clustering DBSCAN pour analyser un jeu de données. Nous souhaitons explorer visuellement et interpréter les résultats des clusters en jouant avec les paramètres de DBSCAN (`eps` et `min_samples`) et en utilisant le score silhouette pour évaluer la qualité des clusters.

## Description et Interprétation des Résultats

### Fonctionnement du Programme

1. **Chargement des Données** :
    - Les données sont chargées à partir du fichier `/content/wholesale_clients.csv`.
    - Les colonnes utilisées pour le clustering sont : 'Calories', 'Protein (g)', 'Fat', 'Sugars', 'Vitamins and Minerals'.

2. **Normalisation des Données** :
    - Les données sont normalisées en utilisant `StandardScaler` pour s'assurer que chaque caractéristique contribue également au clustering.

3. **Application de DBSCAN** :
    - L'algorithme DBSCAN est appliqué avec des paramètres `eps` et `min_samples` initialement définis.
    - Les clusters obtenus sont ajoutés aux données.

4. **Calcul du Score Silhouette** :
    - Le score silhouette est calculé pour évaluer la qualité des clusters. Un score élevé indique des clusters bien définis.

5. **Visualisation des Clusters** :
    - Les résultats sont visualisés à l'aide de graphiques.
    - Un graphique de dispersion montre les clusters formés.
    - Une heatmap montre la répartition des points dans les clusters et les valeurs moyennes des caractéristiques pour chaque cluster.

### Interprétation des Graphiques

#### Graphique de Dispersion

- Les points sont colorés en fonction des clusters auxquels ils appartiennent.
- Les paramètres `eps` et `min_samples` influencent la formation et la séparation des clusters.
- En ajustant ces paramètres, vous pouvez observer comment les clusters changent et comment les points sont regroupés différemment.

#### Heatmap des Clusters

- La heatmap montre la moyenne des valeurs des caractéristiques pour chaque cluster.
- Les lignes représentent les points de données et les colonnes représentent les caractéristiques.
- La couleur indique l'intensité des valeurs des caractéristiques.
- Vous pouvez identifier les caractéristiques dominantes dans chaque cluster.

### Exemple de Meilleure Combinaison

Dans l'image fournie, la meilleure combinaison trouvée est :
- `eps=2.0`
- `min_samples=3.0`

#### Résultats des Clusters

Les clusters formés montrent les moyennes suivantes pour chaque caractéristique :

- **Cluster -1 (Bruit)** :
  - Calories : 115.0
  - Protein (g) : 4.5
  - Fat : 2.5
  - Sugars : 5.5
  - Vitamins and Minerals : 12.5

- **Cluster 0** :
  - Calories : 104.24
  - Protein (g) : 2.42
  - Fat : 0.09
  - Sugars : 6.84
  - Vitamins and Minerals : 22.34

- **Cluster 1** :
  - Calories : 116.67
  - Protein (g) : 2.67
  - Fat : 0.83
  - Sugars : 6.33
  - Vitamins and Minerals : 100.0

#### Distribution des Clusters

- **Cluster 0** : 66 points
- **Cluster 1** : 6 points
- **Cluster -1 (Bruit)** : 2 points

### Score Silhouette

- Le score silhouette pour cette configuration est un indicateur de la qualité du clustering. Un score élevé signifie que les clusters sont bien définis et séparés. Dans cette analyse, le score silhouette permet de choisir les meilleurs paramètres pour `eps` et `min_samples`.

### Conclusion

En jouant avec les paramètres de DBSCAN (`eps` et `min_samples`), nous pouvons observer comment les clusters se forment et évoluent. Le score silhouette est utilisé pour évaluer la qualité des clusters formés. Les visualisations fournissent une compréhension claire de la répartition et des caractéristiques des clusters, ce qui permet d'interpréter les résultats de manière plus intuitive et efficace.



