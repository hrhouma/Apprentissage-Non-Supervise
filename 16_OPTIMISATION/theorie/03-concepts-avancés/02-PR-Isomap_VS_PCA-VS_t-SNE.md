# Comparaison entre  les algorithmes PCA, t-SNE et PR-Isomap

# 1 - Analyse en Composantes Principales (PCA)

#### Vue d'ensemble
L'Analyse en Composantes Principales (PCA) est une technique statistique utilisée pour simplifier un jeu de données en réduisant ses dimensions tout en conservant autant de variabilité que possible. Cela est réalisé en transformant les données originales en un nouvel ensemble de variables orthogonales (non corrélées) appelées composantes principales.

#### Fonctionnement de PCA
1. **Standardisation** : Standardiser les données pour avoir une moyenne de zéro et une variance d'un.
2. **Calcul de la Matrice de Covariance** : Calculer la matrice de covariance pour comprendre comment les variables du jeu de données se rapportent entre elles.
3. **Valeurs Propres et Vecteurs Propres** : Calculer les valeurs propres et les vecteurs propres de la matrice de covariance. Les vecteurs propres déterminent les directions du nouvel espace de caractéristiques, tandis que les valeurs propres déterminent leur magnitude.
4. **Composantes Principales** : Sélectionner les k premiers vecteurs propres (composantes principales) qui correspondent aux plus grandes valeurs propres.
5. **Transformation** : Transformer les données originales dans le nouvel espace de caractéristiques.

#### Applications
- **Visualisation des Données** : Réduction des données à haute dimension en 2D ou 3D pour la visualisation.
- **Réduction du Bruit** : Élimination des dimensions insignifiantes pour réduire le bruit.
- **Extraction de Caractéristiques** : Identification des caractéristiques significatives pour une analyse plus approfondie.

# 2 - t-Distributed Stochastic Neighbor Embedding (t-SNE)

#### Vue d'ensemble
Le t-SNE est une technique de réduction de dimension non linéaire principalement utilisée pour la visualisation des données. Il convertit les données à haute dimension en un espace de faible dimension tout en préservant la structure et les relations entre les points de données.

#### Fonctionnement de t-SNE
1. **Similitudes Par Paires** : Calculer les similitudes par paires entre les points de données dans l'espace de haute dimension.
2. **Distribution de Probabilité Conjointe** : Convertir les similitudes par paires en probabilités conjointes.
3. **Mappage en Faible Dimension** : Initialiser un mappage aléatoire des points de données dans l'espace de faible dimension.
4. **KL-Divergence** : Minimiser la divergence de Kullback-Leibler (KL) entre les probabilités conjointes des espaces de haute et de faible dimension en utilisant la descente de gradient.

#### Applications
- **Exploration des Données** : Visualisation des clusters ou des motifs dans les données à haute dimension.
- **Détection d'Anomalies** : Identification des points de données hors normes ou inhabituels.
- **Prétraitement** : Réduction de la dimension avant d'appliquer d'autres algorithmes d'apprentissage automatique.

# 3 - PR-Isomap

#### Vue d'ensemble
PR-Isomap est une version modifiée de l'Isometric Mapping (Isomap) qui intègre une contrainte de fenêtre Parzen-Rosenblatt (PR). Cette amélioration vise à améliorer l'uniformité du graphe des chemins les plus courts, en particulier pour les données à haute dimension (HD) telles que les biomarqueurs d'imagerie médicale.

#### Fonctionnement de PR-Isomap
1. **Construction du Graphe** : Construire un graphe des k plus proches voisins basé sur les données à haute dimension.
2. **Calcul des Chemins les Plus Courts** : Utiliser la contrainte de fenêtre PR pour modifier l'algorithme des chemins les plus courts, assurant des calculs de distances plus uniformes.
3. **Projection MDS** : Appliquer le Multi-Dimensional Scaling (MDS) pour projeter les données HD dans un espace de faible dimension (LD) tout en préservant les distances géodésiques.

#### Applications
- **Imagerie Médicale** : Réduction de la dimension des biomarqueurs d'imagerie pour des maladies telles que la pneumonie et le cancer du poumon.
- **Médecine de Précision** : Amélioration de la précision de la détection des maladies et de la prédiction des résultats en préservant des informations critiques dans les dimensions réduites.
- **Extraction de Caractéristiques** : Identification et préservation des caractéristiques importantes des données HD.

# 4 - Analyse Comparative

## PCA vs. t-SNE
- **PCA** :
  - Transformation linéaire.
  - Bon pour les grands jeux de données.
  - Préserve la structure globale.
- **t-SNE** :
  - Transformation non linéaire.
  - Meilleur pour visualiser les petits jeux de données.
  - Préserve la structure locale.

## PCA vs. PR-Isomap
- **PCA** :
  - Plus simple et plus rapide.
  - Réduction de dimension linéaire et globale.
- **PR-Isomap** :
  - Gère mieux les structures non linéaires.
  - Plus intensif en calculs en raison des calculs de chemins les plus courts.

## t-SNE vs. PR-Isomap
- **t-SNE** :
  - Supérieur pour visualiser les structures complexes en 2D ou 3D.
  - Coûteux en calculs pour les grands jeux de données.
- **PR-Isomap** :
  - Maintient à la fois les structures locales et globales dans les données HD.
  - Plus adapté aux applications nécessitant la préservation des distances géodésiques.

# 5 - Implémentation en Python

*Exemple simplifié d'implémentation de PCA, t-SNE et PR-Isomap en utilisant Python :*

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Charger un jeu de données d'exemple
digits = load_digits()
X = digits.data

# Appliquer PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Appliquer t-SNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# Visualisation
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, cmap='viridis')
ax[0].set_title('PCA')
ax[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, cmap='viridis')
ax[1].set_title('t-SNE')
plt.show()
```

# Conclusion

Comprendre PCA, t-SNE et PR-Isomap fournit des outils puissants pour la réduction de dimension dans diverses applications, en particulier dans l'imagerie médicale et la médecine de précision. Chaque méthode a ses forces et ses limitations, et le choix de la méthode dépend des exigences spécifiques des données et de l'analyse envisagée. L'intégration de ces techniques peut significativement améliorer la visualisation des données, l'extraction de caractéristiques et l'analyse globale des jeux de données à haute dimension.



# Pour finir: Tableau Comparatif : PCA, t-SNE et PR-Isomap

| Caractéristique                     | PCA                                           | t-SNE                                          | PR-Isomap                                     |
|-------------------------------------|-----------------------------------------------|------------------------------------------------|-----------------------------------------------|
| **Type de Réduction**               | Linéaire                                      | Non linéaire                                   | Non linéaire                                  |
| **Complexité Algorithmique**        | Faible                                        | Moyenne à élevée                               | Élevée                                        |
| **Préservation de la Structure**    | Globale                                       | Locale                                         | Locale et globale                             |
| **Applications Principales**        | Visualisation, réduction du bruit, extraction de caractéristiques | Visualisation de clusters, détection d'anomalies, prétraitement | Imagerie médicale, médecine de précision, extraction de caractéristiques |
| **Taille du Jeu de Données**        | Grands jeux de données                        | Petits à moyens jeux de données                | Moyens à grands jeux de données               |
| **Efficacité Computationnelle**     | Rapide                                        | Lente pour de grands jeux de données           | Plus lente en raison des calculs de chemins les plus courts |
| **Préservation des Distances**      | Euclidiennes                                  | Proximité locale                               | Géodésiques                                   |
| **Scalabilité**                     | Haute                                         | Moyenne à faible                               | Moyenne à faible                              |
| **Visualisation**                   | 2D ou 3D                                      | 2D ou 3D                                       | 2D ou 3D                                       |
| **Paramètres Clés**                 | Nombre de composantes                         | Perplexité, nombre de dimensions               | Nombre de voisins, taille de la fenêtre PR    |
| **Exemples d'Applications**         | Réduction de bruit dans les données, analyse de données, compression de données | Exploration de données complexes, visualisation de clusters, réduction de dimensions avant apprentissage automatique | Analyse de biomarqueurs d'imagerie médicale, prédiction des résultats cliniques, médecine de précision |
| **Robustesse aux Non-Linéarités**   | Faible                                        | Élevée                                         | Élevée                                        |
| **Capacité d'Intégration**          | Facilement intégrable dans des flux de travail existants | Nécessite un traitement intensif, souvent utilisé pour visualisation uniquement | Peut nécessiter des ajustements importants, bonne intégration pour des analyses spécialisées |

# 5 - Conclusion

Chaque méthode présente des avantages et des inconvénients selon l'application envisagée :

- **PCA** est approprié pour des tâches nécessitant une réduction de dimension linéaire rapide et efficace, avec une bonne préservation de la variance globale des données.
- **t-SNE** est excellent pour la visualisation des structures complexes et non linéaires dans les données de petite à moyenne taille, mais peut être computationnellement coûteux.
- **PR-Isomap** offre une solution robuste pour la préservation des structures géodésiques dans les données de grande dimension, particulièrement utile dans des contextes comme l'imagerie médicale, bien que nécessitant des ressources computationnelles importantes.

Le choix de la méthode dépend des besoins spécifiques de l'analyse des données et des contraintes computationnelles.
