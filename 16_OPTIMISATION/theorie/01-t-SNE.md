# Table des Matières - t-SNE

- [Niveau 1 - Vulgarisation](#niveau-1---vulgarisation)
  - [Vue d'ensemble](#vue-densemble)
  - [Comment ça marche ?](#comment-ca-marche)
    - [Similitudes Par Paires](#similitudes-par-paires)
      - [Calcul des Similitudes](#calcul-des-similitudes)
      - [Création de Probabilités](#creation-de-probabilites)
    - [Distribution de Probabilité Conjointe](#distribution-de-probabilite-conjointe)
    - [Mappage en Faible Dimension](#mappage-en-faible-dimension)
      - [Initialisation](#initialisation)
      - [Calcul des Similitudes en 2D](#calcul-des-similitudes-en-2d)
    - [KL-Divergence](#kl-divergence)
      - [Réajustement](#reajustement)
  - [Applications](#applications)
    - [Exploration des Données](#exploration-des-donnees)
      - [Visualisation](#visualisation)
    - [Détection d'Anomalies](#detection-danomalies)
      - [Identification](#identification)
    - [Prétraitement](#pretraitement)
      - [Réduction de Dimension](#reduction-de-dimension)
  - [Exemple Simple](#exemple-simple)

- [Niveau 2 - Avec une petite touche de mathématiques](#niveau-2---avec-une-petite-touche-de-mathématiques)
  - [Vue d'ensemble](#vue-densemble-1)
  - [Comment ça marche ?](#comment-ca-marche-1)
    - [Similitudes Par Paires](#similitudes-par-paires-1)
      - [Calcul des Similitudes](#calcul-des-similitudes-1)
    - [Distribution de Probabilité Conjointe](#distribution-de-probabilite-conjointe-1)
    - [Mappage en Faible Dimension](#mappage-en-faible-dimension-1)
    - [KL-Divergence](#kl-divergence-1)
      - [Réajustement](#reajustement-1)
      - [Descente de Gradient](#descente-de-gradient)
  - [Applications](#applications-1)
    - [Exploration des Données](#exploration-des-donnees-1)
    - [Détection d'Anomalies](#detection-danomalies-1)
    - [Prétraitement](#pretraitement-1)
  - [Exemple avec Mathématiques Simplifiées](#exemple-avec-mathématiques-simplifiées)

- [Niveau 3 - C'est quoi le t-SNE ? (La bonne définition)](#niveau-3---cest-quoi-le-t-sne--la-bonne-défintion)
  - [Équations](#équations)
    - [Équation 1](#équation-1)
    - [Équation 2](#équation-2)
    - [Équation 3](#équation-3)
    - [Équation 4](#équation-4)
    - [Équation 5](#équation-5)
    - [Équation 6](#équation-6)
    - [Équation 7](#équation-7)
  - [Vue d'ensemble](#vue-densemble-2)
  - [Fonctionnement de t-SNE](#fonctionnement-de-t-sne)
    - [Similitudes Par Paires](#similitudes-par-paires-2)
    - [Distribution de Probabilité Conjointe](#distribution-de-probabilite-conjointe-2)
    - [Mappage en Faible Dimension](#mappage-en-faible-dimension-2)
    - [KL-Divergence](#kl-divergence-2)
      - [Minimisation de la Divergence de KL](#minimisation-de-la-divergence-de-kl)
      - [Descente de Gradient](#descente-de-gradient-1)
  - [Applications](#applications-2)
    - [Exploration des Données](#exploration-des-donnees-2)
    - [Détection d'Anomalies](#detection-danomalies-2)
    - [Prétraitement](#pretraitement-2)
  - [Exemple Concret](#exemple-concret)

## Niveau 1 - Vulgarisation
[Retour en haut](#table-des-matières---t-sne)

### Vue d'ensemble
Le t-SNE est une méthode utilisée pour rendre les données complexes plus faciles à comprendre. Imagine que tu as un grand tableau avec beaucoup d'informations (par exemple, les goûts musicaux de milliers de personnes). Le t-SNE aide à transformer ce grand tableau en une image simple que tu peux regarder et comprendre plus facilement.

### Comment ça marche ?
[Retour en haut](#table-des-matières---t-sne)

#### Similitudes Par Paires
##### Calcul des Similitudes
##### Création de Probabilités

#### Distribution de Probabilité Conjointe

#### Mappage en Faible Dimension
##### Initialisation
##### Calcul des Similitudes en 2D

#### KL-Divergence
##### Réajustement

### Applications
[Retour en haut](#table-des-matières---t-sne)

#### Exploration des Données
##### Visualisation

#### Détection d'Anomalies
##### Identification

#### Prétraitement
##### Réduction de Dimension

### Exemple Simple
[Retour en haut](#table-des-matières---t-sne)

Imagine que tu as une liste de fruits avec leurs couleurs et tailles (pomme rouge petite, banane jaune grande, cerise rouge petite, etc.). Le t-SNE peut transformer cette liste en une image où les fruits similaires (comme les pommes et les cerises rouges petites) sont proches les uns des autres, et les fruits différents (comme une banane jaune grande) sont éloignés. Cela te permet de voir rapidement quels fruits sont similaires.

En résumé, le t-SNE est comme un traducteur qui prend des tableaux compliqués et les transforme en images simples et compréhensibles, permettant ainsi de voir facilement les relations et les groupes dans les données.

## Niveau 2 - Avec une petite touche de mathématiques
[Retour en haut](#table-des-matières---t-sne)

### Vue d'ensemble
Le t-SNE (t-distributed Stochastic Neighbor Embedding) est une méthode qui aide à transformer des données complexes en une représentation visuelle simple. Imagine que tu as une liste de fruits avec leurs caractéristiques (couleur et taille). Le t-SNE peut transformer cette liste en une image où les fruits similaires apparaissent proches les uns des autres.

### Comment ça marche ?
[Retour en haut](#table-des-matières---t-sne)

#### Similitudes Par Paires
##### Calcul des Similitudes

#### Distribution de Probabilité Conjointe

#### Mappage en Faible Dimension

#### KL-Divergence
##### Réajustement
##### Descente de Gradient

### Applications
[Retour en haut](#table-des-matières---t-sne)

#### Exploration des Données

#### Détection d'Anomalies

#### Prétraitement

### Exemple avec Mathématiques Simplifiées
[Retour en haut](#table-des-matières---t-sne)

Imagine que tu as les fruits suivants :
- Pomme rouge petite (x1 = 1, y1 = 1)
- Banane jaune grande (x2 = 5, y2 = 5)
- Cerise rouge petite (x3 = 1, y3 = 1)

1. **Calcul des Distances** :
   - Pomme et Cerise : Distance = sqrt((1-1)² + (1-1)²) = 0
   - Pomme et Banane : Distance = sqrt((1-5)² + (1-5)²) = sqrt(32) ≈ 5.66

2. **Création de Probabilités** :
   - Pomme et Cerise : Probabilité élevée (proche de 1)
   - Pomme et Banane : Probabilité faible (proche de 0)

3. **Initialisation en 2D** :
   - Placer les fruits aléatoirement sur une feuille.

4. **Réajustement** :
   - Utiliser la descente de gradient pour ajuster les positions et minimiser la divergence de KL, en s'assurant que les distances en 2D reflètent les distances calculées.

En résumé, le t-SNE prend des données complexes (comme des listes de fruits avec leurs caractéristiques) et les transforme en une image simple, en utilisant des concepts mathématiques pour s'assurer que les relations originales entre les données sont préservées.

## Niveau 3 - C'est quoi le t-SNE ? (La bonne définition)
[Retour en haut](#table-des-matières---t-sne)

### Équations
#### Équation 1
#### Équation 2
#### Équation 3
#### Équation 4
#### Équation 5
#### Équation 6
#### Équation 7

### Vue d'ensemble
Le t-SNE (t-distributed Stochastic Neighbor Embedding) est une technique de réduction de dimension non liné

aire principalement utilisée pour la visualisation des données à haute dimension. Il transforme des données complexes en une représentation de faible dimension tout en préservant la structure des relations entre les points de données.

### Fonctionnement de t-SNE
[Retour en haut](#table-des-matières---t-sne)

#### Similitudes Par Paires

#### Distribution de Probabilité Conjointe

#### Mappage en Faible Dimension

#### KL-Divergence
##### Minimisation de la Divergence de KL
##### Descente de Gradient

### Applications
[Retour en haut](#table-des-matières---t-sne)

#### Exploration des Données

#### Détection d'Anomalies

#### Prétraitement

### Exemple Concret
[Retour en haut](#table-des-matières---t-sne)

Prenons un ensemble de fruits avec des caractéristiques (couleur et taille) :
- Pomme rouge petite (x1 = [1, 1])
- Banane jaune grande (x2 = [5, 5])
- Cerise rouge petite (x3 = [1, 1])

1. **Calcul des Distances** :
   - Distance entre Pomme et Cerise : voir équation 6.
   - Distance entre Pomme et Banane : voir équation 7.

2. **Création de Probabilités** :
   - Pomme et Cerise : Probabilité élevée (proche de 1)
   - Pomme et Banane : Probabilité faible (proche de 0)

3. **Initialisation en 2D** :
   - Placer les fruits aléatoirement sur une feuille (espace 2D).

4. **Réajustement** :
   - Utiliser la descente de gradient pour ajuster les positions et minimiser la divergence de KL, assurant que les distances en 2D reflètent les distances calculées.

En conclusion, le t-SNE transforme des données complexes en représentations visuelles simplifiées, tout en préservant les relations entre les points de données, ce qui facilite l'analyse et la compréhension des structures sous-jacentes.
