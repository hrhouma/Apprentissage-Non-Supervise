
$$ \theta $$ 
est le vecteur des paramètres.

### Exemple simple

Supposons que notre fonction de coût soit une simple parabole $$ J(\theta) = \theta^2 $$. Le gradient de cette fonction est $$ \frac{dJ}{d\theta} = 2\theta $$. Si on commence avec un $$ \theta $$ initial de 3 et un taux d'apprentissage de 0.1, la mise à jour des paramètres se fait comme suit :

1. Calcul du gradient : $$ \frac{dJ}{d\theta} = 2 \times 3 = 6 $$
2. Mise à jour de $$ \theta $$ : $$ \theta := 3 - 0.1 \times 6 = 3 - 0.6 = 2.4 $$

On répète ce processus jusqu'à ce que le gradient soit proche de zéro, ce qui signifie que la fonction de coût est minimisée.

Cette petite touche de mathématiques montre comment la descente du gradient ajuste les paramètres du modèle pour minimiser les erreurs en utilisant des concepts de dérivées et de mises à jour itératives.

# Niveau 01
La descente du gradient est une méthode d'optimisation très utilisée en apprentissage automatique, notamment pour entraîner des réseaux de neurones. Imagine que tu es sur une montagne et que tu veux descendre jusqu'à la vallée la plus basse, mais qu'il fait nuit noire et que tu ne peux pas voir où tu vas. Voici une explication simple pour comprendre ce processus.

### Concept de base

1. **Position actuelle**: Imagine que tu te tiens sur un point quelque part sur la montagne. Ce point représente ta position actuelle (dans le contexte de l'apprentissage automatique, c'est la valeur actuelle des paramètres de ton modèle).

2. **La pente (le gradient)**: La pente est comme la direction et l'inclinaison de la montagne à l'endroit où tu te trouves. Si la pente est forte, c'est que le terrain est très incliné, et si elle est faible, le terrain est plus plat. Le gradient te dit dans quelle direction monter (ou descendre).

3. **Descendre**: Pour descendre la montagne, tu vas dans la direction opposée à celle de la pente. Si tu sens que le terrain monte en face de toi, tu vas dans la direction opposée pour descendre. C'est comme ça que fonctionne la descente du gradient : tu ajustes tes paramètres dans la direction opposée au gradient pour diminuer l'erreur de ton modèle.

### Étapes de la descente du gradient

1. **Choisir un point de départ**: Commence quelque part sur la montagne. En termes de modèle, cela signifie initialiser les paramètres de ton modèle de manière aléatoire ou avec une certaine heuristique.

2. **Calculer le gradient**: À chaque point, calcule la pente (le gradient) de la montagne. En apprentissage automatique, cela signifie calculer la dérivée de la fonction de coût par rapport à chaque paramètre du modèle.

3. **Faire un petit pas**: Fais un petit pas dans la direction opposée à la pente. Ce petit pas est appelé "taux d'apprentissage". Il détermine la taille du pas que tu fais à chaque étape.

4. **Répéter**: Répète les étapes 2 et 3 jusqu'à ce que tu atteignes un point où tu ne peux plus descendre (ou très peu). En termes de modèle, c'est le moment où l'erreur de ton modèle est minimisée autant que possible.

### Illustration

Imagine que tu es sur une colline et que tu veux descendre. Tu fermes les yeux et tu touches le sol avec tes mains pour sentir dans quelle direction la pente descend. Tu fais un petit pas dans cette direction, puis tu répètes le processus jusqu'à atteindre la vallée.

### Exemple simple

Supposons que tu essaies de deviner le bon nombre dans un jeu, et chaque fois que tu fais une supposition, tu reçois un indice disant si tu es trop haut ou trop bas. La descente du gradient, c'est comme ajuster ta supposition petit à petit dans la direction qui te rapproche de la bonne réponse, en utilisant les indices que tu reçois.

Voilà, la descente du gradient expliquée de manière simple sans que j'utilise de mathématiques complexes. 
L'idée principale est de toujours ajuster ta position de manière à descendre progressivement jusqu'à atteindre le point le plus bas, ou la meilleure solution possible pour ton modèle.

# Niveau 2

Pour ajouter une touche de mathématiques, on peut introduire la notion de fonction de coût et de gradient de manière simple.

### Fonction de coût

En apprentissage automatique, on utilise une fonction de coût (ou fonction de perte) pour mesurer à quel point notre modèle fait des erreurs. Par exemple, si on entraîne un modèle de régression linéaire, la fonction de coût courante est l'erreur quadratique moyenne (MSE).

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 
$$

où $$ y_i $$ est la valeur réelle, $$ \hat{y}_i $$ est la valeur prédite par le modèle, et *n* est le nombre total d'exemples.

### Gradient

Le gradient est un vecteur qui contient les dérivées partielles de la fonction de coût par rapport à chaque paramètre du modèle. Ces dérivées nous indiquent comment la fonction de coût change si on modifie légèrement chaque paramètre.

Si on a une fonction de coût $$ J(\theta) $$ dépendant des paramètres $$ \theta $$, le gradient est :

$$ 
\nabla J(\theta) = \left( \frac{\partial J}{\partial \theta_1}, \frac{\partial J}{\partial \theta_2}, \dots, \frac{\partial J}{\partial \theta_n} \right) 
$$

### Mise à jour des paramètres

Lors de chaque itération de la descente du gradient, on met à jour les paramètres en soustrayant une fraction du gradient. Cette fraction est appelée le taux d'apprentissage (\( \alpha \)).

$$ 
\theta := \theta - \alpha \nabla J(\theta) 
$$



# Niveau 3

### Équations

# Équation 1 :
$$
\theta_{j} = \theta_{j} - \alpha \frac{\partial J(\theta)}{\partial \theta_{j}}
$$

# Équation 2 :
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)^2
$$

# Équation 3 :
$$
h_{\theta}(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n
$$

# Équation 4 :
$$
\frac{\partial J(\theta)}{\partial \theta_{j}} = \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_{j}^{(i)}
$$

### Vue d'ensemble
La descente de gradient est une méthode d'optimisation utilisée pour minimiser une fonction de coût, souvent employée dans le cadre des algorithmes d'apprentissage automatique. L'objectif est d'ajuster les paramètres du modèle afin de minimiser l'erreur entre les prédictions du modèle et les valeurs réelles des données d'entraînement.

### Fonctionnement de la descente de gradient

1. **Mise à jour des paramètres :**
   - **Formule de mise à jour** : Pour chaque paramètre $$\theta_{j}$$, on ajuste la valeur en utilisant l'équation 1. Cela permet de déplacer les paramètres dans la direction de la pente descendante de la fonction de coût.

2. **Fonction de coût :**
   - **Définition de la fonction de coût** : La fonction de coût $$J(\theta)$$ mesure l'erreur entre les prédictions du modèle et les valeurs réelles, comme décrit par l'équation 2.

3. **Hypothèse du modèle :**
   - **Formule de l'hypothèse** : L'hypothèse $$h_{\theta}(x)$$ est une combinaison linéaire des paramètres et des caractéristiques des données, définie par l'équation 3.

4. **Calcul du gradient :**
   - **Formule du gradient** : Le gradient de la fonction de coût par rapport à chaque paramètre $$\theta_{j}$$ est donné par l'équation 4. Il s'agit de la moyenne des produits des erreurs et des caractéristiques correspondantes.

### Applications

1. **Régression Linéaire** :
   - **Ajustement des paramètres** : La descente de gradient est utilisée pour trouver les paramètres optimaux qui minimisent l'erreur quadratique entre les prédictions et les valeurs réelles.

2. **Réseaux de Neurones** :
   - **Entraînement des modèles** : Dans les réseaux de neurones, la descente de gradient est utilisée pour ajuster les poids et les biais afin de minimiser la fonction de coût, souvent par le biais de la rétropropagation.

3. **Autres Algorithmes d'Apprentissage** :
   - **Optimisation** : De nombreux algorithmes d'apprentissage automatique utilisent la descente de gradient pour optimiser leurs paramètres, y compris les machines à vecteurs de support et les modèles de clustering.

### Exemple Concret

Prenons un exemple simple de régression linéaire avec un seul paramètre $$\theta$$ et une seule caractéristique $$x$$.

1. **Initialisation :**
   - Initialiser le paramètre $$\theta$$ à une valeur arbitraire.

2. **Calcul de l'erreur :**
   - Utiliser l'équation 3 pour calculer l'hypothèse $$h_{\theta}(x)$$.
   - Calculer l'erreur entre l'hypothèse et les valeurs réelles des données d'entraînement.

3. **Mise à jour du paramètre :**
   - Calculer le gradient en utilisant l'équation 4.
   - Mettre à jour le paramètre $$\theta$$ en utilisant l'équation 1.

4. **Répétition :**
   - Répéter les étapes 2 et 3 jusqu'à ce que la fonction de coût $$J(\theta)$$ converge à une valeur minimale.

En conclusion, la descente de gradient est une technique essentielle en apprentissage automatique pour optimiser les paramètres des modèles en minimisant la fonction de coût, ce qui conduit à des prédictions plus précises.
