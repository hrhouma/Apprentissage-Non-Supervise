# Analyse en Composantes Principales (ACP)

## Introduction

L'Analyse en Composantes Principales (ACP) est une technique statistique qui permet de réduire la dimensionnalité d'un ensemble de données tout en conservant le plus d'information possible. Elle est utilisée pour simplifier des jeux de données complexes, en transformant les variables d'origine en un nouveau jeu de variables appelées "composantes principales".

## Objectif

L'objectif de l'ACP est de :

1. **Réduire la dimensionnalité des données** : Diminuez le nombre de variables tout en préservant le maximum d'information.
2. **Identifier les motifs cachés** : Détectez les structures sous-jacentes dans les données.
3. **Visualiser les données** : Facilitez la visualisation des données en deux ou trois dimensions.

## Fonctionnement

L'ACP fonctionne en suivant ces étapes :

1. **Standardisation des données** : Les données sont normalisées pour avoir une moyenne de 0 et une variance de 1.
2. **Calcul de la matrice de covariance** : Évaluez les relations entre les variables.
3. **Calcul des valeurs propres et des vecteurs propres** : Identifiez les directions de la plus grande variance (composantes principales).
4. **Sélection des composantes principales** : Choisissez les composantes qui expliquent le plus de variance.
5. **Transformation des données** : Projetez les données d'origine sur les nouvelles composantes principales.

## Avantages

- **Réduction de la complexité** : Simplifie les modèles en réduisant le nombre de variables.
- **Amélioration de la visualisation** : Permet de visualiser des données multidimensionnelles en 2D ou 3D.
- **Élimination du bruit** : Réduit l'impact des variables peu informatives ou bruitées.

## Exemple d'application

```python
# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Chargement des données
data = np.array([[2.5, 2.4],
                 [0.5, 0.7],
                 [2.2, 2.9],
                 [1.9, 2.2],
                 [3.1, 3.0],
                 [2.3, 2.7],
                 [2.0, 1.6],
                 [1.0, 1.1],
                 [1.5, 1.6],
                 [1.1, 0.9]])

# Standardisation des données
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Application de l'ACP
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_standardized)

# Visualisation des composantes principales
plt.figure()
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.title('Projection des données sur les deux premières composantes principales')
plt.show()
```

## Conclusion

L'ACP est un outil puissant pour réduire la dimensionnalité des données et révéler des structures cachées. Elle est largement utilisée dans des domaines variés comme la finance, la biologie, et le traitement du signal.
