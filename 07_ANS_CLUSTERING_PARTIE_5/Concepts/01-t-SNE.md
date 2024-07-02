# Niveau 1 - vulgarisation
### Vue d'ensemble
Le t-SNE est une méthode utilisée pour rendre les données complexes plus faciles à comprendre. Imagine que tu as un grand tableau avec beaucoup d'informations (par exemple, les goûts musicaux de milliers de personnes). Le t-SNE aide à transformer ce grand tableau en une image simple que tu peux regarder et comprendre plus facilement.

### Comment ça marche ?

1. **Similitudes Par Paires** :
   - **Calcul des Similitudes** : Imagine que tu as une classe avec plein d'élèves. Le t-SNE commence par regarder à quel point chaque élève est proche des autres en fonction de leurs goûts musicaux.
   - **Création de Probabilités** : Ensuite, il transforme ces "proximités" en chiffres entre 0 et 1. Plus les élèves ont des goûts similaires, plus le chiffre est proche de 1.

2. **Distribution de Probabilité Conjointe** :
   - Il combine tous ces chiffres pour créer une "carte" montrant à quel point chaque élève est proche de chaque autre élève dans le grand tableau.

3. **Mappage en Faible Dimension** :
   - **Initialisation** : Ensuite, le t-SNE place ces élèves aléatoirement sur une feuille de papier (c'est-à-dire dans une image en 2D).
   - **Calcul des Similitudes en 2D** : Il vérifie à quel point les élèves sont proches les uns des autres sur cette feuille de papier.

4. **KL-Divergence** :
   - **Réajustement** : Maintenant, il compare les "proximités" des élèves sur la feuille avec celles du grand tableau initial. Si deux élèves qui étaient très proches dans le tableau sont éloignés sur la feuille, il les rapproche. Il fait cela plein de fois jusqu'à ce que la "carte" sur la feuille ressemble le plus possible au grand tableau initial.

### Applications

1. **Exploration des Données** :
   - **Visualisation** : Imagine que tu as un grand tableau de résultats scolaires. Le t-SNE peut transformer ce tableau en une image où les élèves ayant des résultats similaires sont proches les uns des autres. Cela permet de voir rapidement des groupes d'élèves avec des résultats semblables.

2. **Détection d'Anomalies** :
   - **Identification** : Si un élève a des résultats très différents des autres, il apparaîtra isolé sur l'image. Cela aide à repérer rapidement les cas particuliers ou anomalies.

3. **Prétraitement** :
   - **Réduction de Dimension** : Avant d'utiliser d'autres méthodes pour analyser les données, on peut utiliser le t-SNE pour simplifier le tableau, en gardant seulement les informations les plus importantes. Cela rend l'analyse plus facile et plus rapide.

### Exemple Simple

Imagine que tu as une liste de fruits avec leurs couleurs et tailles (pomme rouge petite, banane jaune grande, cerise rouge petite, etc.). Le t-SNE peut transformer cette liste en une image où les fruits similaires (comme les pommes et les cerises rouges petites) sont proches les uns des autres, et les fruits différents (comme une banane jaune grande) sont éloignés. Cela te permet de voir rapidement quels fruits sont similaires.

En résumé, le t-SNE est comme un traducteur qui prend des tableaux compliqués et les transforme en images simples et compréhensibles, permettant ainsi de voir facilement les relations et les groupes dans les données.


# Niveau 2  - avec une petite touche de mathématiques

D'accord, reprenons l'exemple des fruits et ajoutons une touche de mathématiques simplifiées pour expliquer le fonctionnement du t-SNE.

### Vue d'ensemble
Le t-SNE (t-distributed Stochastic Neighbor Embedding) est une méthode qui aide à transformer des données complexes en une représentation visuelle simple. Imagine que tu as une liste de fruits avec leurs caractéristiques (couleur et taille). Le t-SNE peut transformer cette liste en une image où les fruits similaires apparaissent proches les uns des autres.

### Comment ça marche ?

1. **Similitudes Par Paires** :
   - **Calcul des Similitudes** : Supposons que tu as une pomme rouge petite, une banane jaune grande et une cerise rouge petite. Le t-SNE calcule à quel point chaque fruit est proche des autres en utilisant une mesure de distance. Par exemple, pour simplifier, on pourrait utiliser la distance Euclidienne.
     - Pomme et Cerise : Elles sont toutes deux rouges et petites, donc très proches.
     - Pomme et Banane : Elles ont des couleurs et des tailles différentes, donc plus éloignées.
   - **Création de Probabilités** : On convertit ces distances en probabilités qui représentent la chance qu'un fruit choisisse un autre fruit comme voisin. Par exemple :
     - Pomme et Cerise : Probabilité élevée (disons 0.9)
     - Pomme et Banane : Probabilité faible (disons 0.1)

2. **Distribution de Probabilité Conjointe** :
   - On crée une distribution de probabilités qui montre à quel point chaque fruit est proche de chaque autre fruit dans la liste originale.

3. **Mappage en Faible Dimension** :
   - **Initialisation** : On place les fruits de manière aléatoire sur une feuille de papier (2D).
   - **Calcul des Similitudes en 2D** : On calcule à nouveau les distances et les probabilités, mais cette fois-ci dans l'espace 2D.

4. **KL-Divergence** :
   - **Réajustement** : Le t-SNE compare les probabilités dans l'espace original avec celles dans l'espace 2D. S'il y a une grande différence, il ajuste les positions des fruits pour minimiser cette différence. La mesure utilisée pour cette comparaison est appelée la divergence de Kullback-Leibler (KL), qui est une façon de mesurer à quel point deux distributions de probabilités sont différentes.
   - **Descente de Gradient** : Pour minimiser la divergence de KL, le t-SNE utilise une méthode appelée descente de gradient. C'est un peu comme essayer de trouver le point le plus bas dans une vallée en descendant progressivement.

### Applications

1. **Exploration des Données** :
   - **Visualisation** : Si tu as une grande liste de fruits avec différentes caractéristiques, le t-SNE peut transformer cette liste en une image où les fruits similaires sont regroupés. Cela permet de voir rapidement les groupes de fruits similaires.

2. **Détection d'Anomalies** :
   - **Identification** : Si un fruit a des caractéristiques très différentes des autres, il apparaîtra isolé sur l'image. Cela aide à repérer les fruits inhabituels.

3. **Prétraitement** :
   - **Réduction de Dimension** : Avant d'utiliser d'autres méthodes pour analyser les données, le t-SNE peut simplifier la liste en conservant seulement les informations les plus importantes. Cela rend l'analyse plus facile et plus rapide.

### Exemple avec Mathématiques Simplifiées

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

# Niveau 3 - C'est quoi le t-SNE ?  ( La bonne définition)
### Équations

# Équation 1 : 
   $$
   \text{similarité}_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma_i^2}\right)
   $$

# Équation 2 : 
   $$
   P_{j|i} = \frac{\exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma_i^2}\right)}{\sum_{k \neq i} \exp\left(-\frac{\|x_i - x_k\|^2}{2\sigma_i^2}\right)}
   $$

# Équation 3 : 
   $$
   P_{ij} = \frac{P_{j|i} + P_{i|j}}{2N}
   $$

# Équation 4 : 
   $$
   Q_{ij} = \frac{\left(1 + \|y_i - y_j\|^2\right)^{-1}}{\sum_{k \neq l} \left(1 + \|y_k - y_l\|^2\right)^{-1}}
   $$

# Équation 5 : 
   $$
   KL(P||Q) = \sum_{i \neq j} P_{ij} \log \frac{P_{ij}}{Q_{ij}}
   $$

# Équation 6 : 
$$
\|x_1 - x_3\| = \sqrt{(1-1)^2 + (1-1)^2} = 0
$$

# Équation 7 : 
$$ 
\|x_1 - x_2\| = \sqrt{(1-5)^2 + (1-5)^2} = \sqrt{32} \approx 5.66
$$

### Vue d'ensemble
Le t-SNE (t-distributed Stochastic Neighbor Embedding) est une technique de réduction de dimension non linéaire principalement utilisée pour la visualisation des données à haute dimension. Il transforme des données complexes en une représentation de faible dimension tout en préservant la structure des relations entre les points de données.

### Fonctionnement de t-SNE

1. **Similitudes Par Paires** :
   - **Calcul des Similitudes** : Pour chaque paire de points de données dans l'espace de haute dimension, t-SNE calcule une mesure de similarité en utilisant l'équation 1.

2. **Distribution de Probabilité Conjointe** :
   - **Probabilités Conditionnelles** : Les similarités sont ensuite converties en probabilités conditionnelles $$P_{j|i}$$ en utilisant l'équation 2.
   - **Probabilités Symétriques** : Ces probabilités conditionnelles sont symétrisées pour obtenir une distribution de probabilité conjointe $$P_{ij}$$, comme décrit par l'équation 3.

3. **Mappage en Faible Dimension** :
   - **Initialisation** : Les points de données sont initialement placés aléatoirement dans un espace de faible dimension (2D ou 3D).
   - **Probabilités en 2D/3D** : On calcule également des probabilités conjointes $$Q_{ij}$$ dans cet espace de faible dimension en utilisant l'équation 4.

4. **KL-Divergence** :
   - **Minimisation de la Divergence de KL** : L'objectif est de minimiser la divergence de Kullback-Leibler (KL) entre les distributions $$P$$ et $$Q$$, comme défini par l'équation 5.
   - **Descente de Gradient** : On utilise des techniques de descente de gradient pour ajuster les positions des points dans l'espace de faible dimension afin de minimiser cette divergence. Cela implique de déplacer les points de manière itérative pour réduire l'écart entre $$P$$ et $$Q$$.

### Applications

1. **Exploration des Données** :
   - **Visualisation** : Le t-SNE est utilisé pour visualiser des structures et des motifs dans des données complexes, révélant des clusters naturels qui ne sont pas visibles dans l'espace de haute dimension.

2. **Détection d'Anomalies** :
   - **Identification** : Les points de données qui sont éloignés des clusters principaux peuvent être identifiés comme des anomalies ou des outliers, facilitant la détection de cas exceptionnels.

3. **Prétraitement** :
   - **Réduction de Dimension** : Avant d'appliquer d'autres algorithmes de machine learning, le t-SNE peut réduire la dimension des données tout en conservant les relations importantes, améliorant ainsi les performances des modèles en réduisant la complexité.

### Exemple Concret

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
