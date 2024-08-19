## **Table des Matières**

1. [Introduction aux Réseaux de Neurones](#intro)

2. [Apprentissage Supervisé vs Non Supervisé](#supervised-vs-unsupervised)

   - [Entraînement et Test](#training-testing)
     
   - [Importance de la Division](#data-split-importance)
     
3. [Type de Données pour l'IA](#data-types)

   - [Types de Données : float32 vs float64](#float32-vs-float64)
     
   - [Numérique vs Chaîne de Caractères](#numeric-vs-string)
     
4. [Variables Catégoriques](#categorical-variables)

   - [Définition des Variables Catégoriques](#categorical-definition)
     
   - [Importance pour l'IA](#categorical-importance)
     
   - [One-Hot Encoding](#one-hot-encoding)
     
5. [Données de Validation](#validation-data)

   - [Qu'est-ce que les Données de Validation ?](#what-is-validation-data)
     
   - [Validation vs Test](#validation-vs-test)
     
6. [Normalisation en IA](#normalization)

   - [Qu'est-ce que la Normalisation ?](#what-is-normalization)
     
   - [Processus et Importance](#normalization-importance)
     
7. [Modèle Séquentiel en Keras](#keras-sequential)

   - [Création et Compréhension](#sequential-model-creation)
     
8. [Couches dans un Réseau de Neurones](#neural-layers)

   - [Couche Flatten](#flatten-layer)
     
   - [Couche de Sortie](#output-layer)
     
   - [Couche Dense](#dense-layer)
     
9. [Fonction d'Activation](#activation-functions)

   - [Qu'est-ce qu'une Fonction d'Activation ?](#activation-definition)
     
   - [ReLU vs Softmax](#relu-vs-softmax)
     
10. [Architecture de Réseau de Neurones](#network-architecture)

    - [Choix des Neurones et des Couches](#neuron-layer-choices)
      
    - [Choisir le Nombre de Neurones](#choosing-neurons)

---

## **Cours**

<a id="intro"></a>

### 1. Introduction aux Réseaux de Neurones 


Les réseaux de neurones artificiels (RNA) sont des modèles inspirés par le fonctionnement des neurones biologiques dans le cerveau humain. Leur objectif est de simuler la capacité du cerveau à reconnaître des motifs et à prendre des décisions basées sur des données d'entrée. Les RNA sont à la base de nombreuses avancées récentes dans le domaine de l'intelligence artificielle (IA), notamment dans les applications de reconnaissance d'image, de traitement du langage naturel, et bien d'autres.

Un réseau de neurones est composé de couches de neurones connectées entre elles. Chaque neurone reçoit des entrées, effectue un calcul (généralement une somme pondérée des entrées), applique une fonction d'activation, et transmet le résultat en sortie. Les principales composantes d'un réseau de neurones incluent :

1. **Les neurones** : Unité de base du réseau, chaque neurone prend des entrées, les transforme via une fonction d'activation, et génère une sortie.
2. **Les couches** : Les neurones sont organisés en couches. On distingue généralement trois types de couches :
   - **Couche d'entrée** : La première couche qui reçoit les données brutes.
   - **Couches cachées** : Couches intermédiaires où les calculs complexes se produisent.
   - **Couche de sortie** : La couche finale qui produit la sortie du réseau.
3. **Les poids** : Les connexions entre les neurones sont caractérisées par des poids, qui déterminent l'importance des signaux passant d'un neurone à un autre.
4. **Fonctions d'activation** : Elles déterminent si un neurone doit être activé ou non en appliquant une transformation mathématique sur les entrées.

Les réseaux de neurones peuvent être entraînés à reconnaître des motifs à partir de données étiquetées. Ce processus d'entraînement consiste à ajuster les poids des connexions pour minimiser l'erreur entre la sortie prédite et la sortie attendue. Une fois entraîné, le réseau peut être utilisé pour prédire des résultats sur de nouvelles données.

L'une des forces des réseaux de neurones est leur capacité à apprendre des représentations complexes des données, ce qui les rend particulièrement efficaces pour les tâches de classification, de régression, et même de génération de données.


   [Retour en haut](#table-des-matières)

<a id="supervised-vs-unsupervised"></a>

<hr/> 
<hr/> 
<hr/> 

### 2. Apprentissage Supervisé vs Non Supervisé

<hr/> 
<hr/> 
<hr/> 


### 2. Apprentissage Supervisé vs Non Supervisé

L'apprentissage supervisé et l'apprentissage non supervisé sont deux des principales approches utilisées en apprentissage automatique pour entraîner des modèles, y compris les réseaux de neurones.

**Apprentissage Supervisé :**

Dans l'apprentissage supervisé, le modèle est entraîné sur un ensemble de données étiquetées, ce qui signifie que chaque exemple de données d'entrée est associé à une sortie connue (ou étiquette). L'objectif est que le modèle apprenne à mapper les entrées aux sorties de manière précise. Au fur et à mesure que le modèle est entraîné, il ajuste ses paramètres pour minimiser l'écart entre les prédictions et les valeurs réelles.

Exemples courants d'apprentissage supervisé :
- **Classification** : Le modèle apprend à classer les données en catégories, par exemple, reconnaître des images de chats ou de chiens.
- **Régression** : Le modèle prédit des valeurs continues, comme estimer le prix d'une maison en fonction de ses caractéristiques.
- **Réseaux de Neurones** : Souvent utilisés dans un cadre supervisé pour des tâches comme la reconnaissance d'images ou le traitement du langage naturel, où le réseau apprend à partir de grandes quantités de données étiquetées.

**Apprentissage Non Supervisé :**

Contrairement à l'apprentissage supervisé, l'apprentissage non supervisé ne dispose pas de données étiquetées. Le modèle est exposé à un ensemble de données où les étiquettes ne sont pas connues, et il doit découvrir des structures ou des motifs sous-jacents dans les données. L'objectif est d'explorer les relations entre les données d'entrée sans supervision explicite.

Exemples courants d'apprentissage non supervisé :
- **Clustering** : Le modèle regroupe les données en clusters basés sur des similitudes, comme segmenter des clients en groupes ayant des comportements d'achat similaires.
- **Réduction de dimensionnalité** : Le modèle réduit le nombre de variables dans un ensemble de données tout en préservant les informations les plus importantes, comme le fait la méthode d'Analyse en Composantes Principales (PCA).
- **Autoencodeurs** : Un type spécifique de réseau de neurones utilisé en apprentissage non supervisé. Les autoencodeurs apprennent à compresser les données d'entrée en une représentation plus compacte, puis à reconstruire les données d'origine à partir de cette représentation. Ils sont souvent utilisés pour la réduction de dimensionnalité, la détection d'anomalies, ou l'apprentissage des caractéristiques importantes des données sans avoir besoin d'étiquettes.

**Principales Différences :**

1. **Données d'entraînement** :
   - **Supervisé** : Données étiquetées.
   - **Non supervisé** : Données non étiquetées.
   
2. **Objectif** :
   - **Supervisé** : Apprendre à prédire une sortie précise basée sur les entrées.
   - **Non supervisé** : Découvrir des motifs cachés ou des structures dans les données.

3. **Applications** :
   - **Supervisé** : Classification, régression.
   - **Non supervisé** : Clustering, réduction de dimensionnalité, autoencodeurs.

Les réseaux de neurones jouent un rôle dans les deux types d'apprentissage. Dans l'apprentissage supervisé, ils sont utilisés pour des tâches prédictives complexes. Dans l'apprentissage non supervisé, les autoencodeurs sont un exemple de l'utilisation des réseaux de neurones pour découvrir des représentations latentes dans les données sans supervision explicite.


---


   [Retour en haut](#table-des-matières)

<a id="training-testing"></a>






### Entraînement et Test

Dans le contexte des réseaux de neurones, le processus d'entraînement et de test est fondamental pour développer un modèle qui peut faire des prédictions précises sur des données nouvelles. Ce processus est généralement divisé en plusieurs étapes, chacune ayant son propre rôle pour garantir que le modèle fonctionne correctement.

#### **Entraînement**

L'entraînement d'un réseau de neurones consiste à ajuster les poids des connexions entre les neurones en utilisant un ensemble de données d'entraînement. Ce processus est itératif et se déroule en plusieurs étapes, appelées époques.

**Étapes de l'entraînement :**

1. **Propagation avant (Forward Propagation)** :
   - Les données d'entrée passent à travers le réseau, couche par couche, jusqu'à la couche de sortie.
   - À chaque couche, les neurones effectuent des calculs en utilisant les poids et les biais, puis appliquent une fonction d'activation pour produire une sortie.

2. **Calcul de l'erreur (Loss Calculation)** :
   - La sortie prédite par le réseau est comparée à la sortie réelle (ou étiquette) dans les données d'entraînement.
   - Une fonction de coût (comme l'erreur quadratique moyenne pour la régression ou l'entropie croisée pour la classification) est utilisée pour calculer l'erreur entre la sortie prédite et la sortie réelle.

3. **Rétropropagation (Backward Propagation)** :
   - Le réseau ajuste ses poids pour minimiser l'erreur. Ce processus se fait en calculant les gradients de la fonction de coût par rapport aux poids et en utilisant ces gradients pour mettre à jour les poids. C'est le cœur de l'algorithme d'entraînement du réseau de neurones, souvent réalisé avec la méthode de descente de gradient.

4. **Mise à jour des poids (Weight Update)** :
   - Les poids des connexions sont ajustés selon les gradients calculés, souvent avec un facteur de taux d'apprentissage qui contrôle la magnitude de la mise à jour.
   - Ce processus est répété pour plusieurs itérations (époques) jusqu'à ce que l'erreur soit minimisée.

L'objectif de l'entraînement est de minimiser la différence entre les prédictions du modèle et les valeurs réelles en ajustant les poids du réseau de neurones.

#### **Test**

Après l'entraînement, le modèle est évalué sur un ensemble de données de test qui n'a jamais été utilisé pendant l'entraînement. Cela permet de mesurer la capacité du modèle à généraliser, c'est-à-dire à faire des prédictions correctes sur des données qu'il n'a pas vues auparavant.

**Étapes du test :**

1. **Propagation avant uniquement (Forward Propagation)** :
   - Comme pendant l'entraînement, les données de test passent à travers le réseau. Cependant, ici, les poids sont fixés (ils ne sont pas ajustés) et seule la propagation avant est effectuée.

2. **Évaluation des performances (Performance Evaluation)** :
   - Les prédictions générées par le modèle sont comparées aux vraies valeurs pour calculer des métriques de performance comme l'exactitude, la précision, le rappel, le F1-score, etc.
   - Aucune rétropropagation ou mise à jour des poids n'est effectuée pendant la phase de test.

**Importance de l'étape de test** :

- **Généralisation** : Le test sur un ensemble de données indépendant permet de s'assurer que le modèle n'est pas simplement "mémorisé" les données d'entraînement, mais qu'il a réellement appris à généraliser à de nouvelles données.
- **Comparaison des modèles** : Les performances sur l'ensemble de test sont souvent utilisées pour comparer différents modèles ou différentes configurations de modèles.

### Importance de la Division

Pour garantir que le modèle de réseau de neurones fonctionne bien à la fois sur les données d'entraînement et sur des données nouvelles, il est essentiel de diviser correctement les données disponibles en ensembles distincts. Les trois ensembles les plus couramment utilisés sont :

1. **Ensemble d'entraînement** :
   - Utilisé pour entraîner le modèle. Le modèle ajuste ses poids en fonction de cet ensemble pour minimiser l'erreur.

2. **Ensemble de validation** :
   - Utilisé pendant l'entraînement pour ajuster les hyperparamètres et pour prévenir le surapprentissage. Cet ensemble aide à déterminer quand arrêter l'entraînement pour éviter le surapprentissage.

3. **Ensemble de test** :
   - Utilisé après l'entraînement pour évaluer la performance finale du modèle. Cet ensemble fournit une estimation de la performance du modèle sur des données réellement nouvelles.

**Pourquoi cette division est-elle importante ?**

- **Prévention du surapprentissage (Overfitting)** : En utilisant un ensemble de validation séparé, il est possible de détecter et d'éviter le surapprentissage, où le modèle s'ajuste trop étroitement aux données d'entraînement.
- **Évaluation de la généralisation** : L'ensemble de test, distinct des ensembles d'entraînement et de validation, fournit une évaluation impartiale de la capacité du modèle à généraliser.
- **Répartition équilibrée** : Pour que les résultats soient fiables, il est important que les ensembles soient représentatifs et équilibrés, c'est-à-dire qu'ils contiennent une distribution similaire de classes ou de caractéristiques que l'on retrouve dans l'ensemble global.

Une bonne division des données est essentielle pour développer des modèles de réseaux de neurones robustes et généralisables.

---

[Retour en haut](#table-des-matières)

<a id="data-types"></a>


<hr/>
<hr/>
<hr/>

### 3. Type de Données pour l'IA

<hr/>
<hr/>
<hr/>

Le type de données utilisé dans les réseaux de neurones a un impact significatif sur la manière dont le modèle traite les informations. Différents types de données, tels que les nombres flottants, les chaînes de caractères, et les valeurs catégoriques, sont gérés de différentes manières par le modèle.

---

[Retour en haut](#table-des-matières)

<a id="float32-vs-float64"></a>

#### Types de Données : float32 vs float64

Les types de données numériques sont souvent représentés par des nombres flottants, tels que `float32` et `float64`, qui diffèrent principalement par leur précision.

- **float32** : Ce type de données utilise 32 bits pour représenter un nombre flottant. Il offre une précision suffisante pour la plupart des tâches d'apprentissage automatique et est largement utilisé en raison de son efficacité en termes de mémoire et de calcul. Par exemple, une température mesurée en degrés Celsius pourrait être stockée en tant que `float32` : `23.56`.

- **float64** : Utilise 64 bits pour représenter un nombre flottant, offrant ainsi une précision supérieure. Cependant, il est plus gourmand en mémoire et en puissance de calcul. Il est utilisé lorsque des calculs très précis sont nécessaires. Par exemple, une valeur monétaire avec plusieurs décimales pourrait être stockée en tant que `float64` : `123456.789012`.

**Exemples :**
- **float32** : `23.56` (température en degrés Celsius)
- **float64** : `123456.789012` (valeur monétaire précise)

---

[Retour en haut](#table-des-matières)

<a id="numeric-vs-string"></a>

#### Numérique vs Chaîne de Caractères

Les données numériques, comme les `float32` et `float64`, sont directement exploitables par les réseaux de neurones car elles peuvent être utilisées dans des calculs mathématiques. Cependant, les chaînes de caractères (ou `strings`) nécessitent une pré-traitement avant de pouvoir être utilisées par les modèles.

- **Numérique** : Ce type de données inclut les nombres entiers et les flottants. Ils sont directement utilisés dans les réseaux de neurones pour effectuer des calculs et ajuster les poids du modèle. Par exemple, une valeur représentant l'âge d'une personne : `30`.

- **Chaîne de Caractères (String)** : Une chaîne de caractères est une séquence de caractères utilisée pour représenter des données textuelles. Les réseaux de neurones ne peuvent pas utiliser directement les chaînes de caractères dans leur forme brute; elles doivent d'abord être converties en une forme numérique, souvent via des techniques comme l'encodage one-hot ou l'intégration de mots (word embeddings). Par exemple, un nom de ville : `"Montreal"`.

**Exemples :**
- **Numérique** : `30` (âge d'une personne)
- **Chaîne de caractères** : `"Montreal"` (nom d'une ville)

---

[Retour en haut](#table-des-matières)

<a id="categorical-variables"></a>

#### Variables Catégoriques

Les variables catégoriques sont un type de données qui représentent des catégories ou des groupes. Contrairement aux données numériques, elles ne représentent pas une quantité, mais plutôt une qualité ou une caractéristique. Les variables catégoriques doivent être converties en un format numérique avant d'être utilisées par un réseau de neurones. Ceci est souvent fait via des techniques comme l'encodage one-hot ou l'encodage ordinal.

- **Exemple de Variable Catégorique** : Supposons une variable qui représente une catégorie de couleur : `"Rouge"`, `"Bleu"`, `"Vert"`. Pour qu'un réseau de neurones puisse traiter ces informations, elles sont souvent converties en un format numérique via l'encodage one-hot, par exemple :
  - `"Rouge"` → `[1, 0, 0]`
  - `"Bleu"` → `[0, 1, 0]`
  - `"Vert"` → `[0, 0, 1]`

Une autre approche peut être l'encodage ordinal si les catégories ont un ordre naturel (par exemple : `"Petit"`, `"Moyen"`, `"Grand"`).

**Exemple :**
- **Variable Catégorique** : `"Montreal"` (comme nom de ville)

---

Ces différents types de données sont traités de manière spécifique par les réseaux de neurones pour extraire les informations pertinentes et faire des prédictions. Les données numériques peuvent être directement intégrées dans les calculs du réseau, tandis que les chaînes de caractères et les variables catégoriques nécessitent des étapes de pré-traitement pour être utilisées efficacement.

---

[Retour en haut](#table-des-matières)










<hr/> 
<hr/> 
<hr/> 

### 4. Variables Catégoriques

<hr/> 
<hr/> 
<hr/> 

Les variables catégoriques sont un type de données qui représentent des catégories ou des groupes discrets, plutôt que des valeurs numériques continues. Elles jouent un rôle essentiel dans de nombreuses applications d'intelligence artificielle (IA), car elles permettent de représenter des attributs qualitativement distincts, comme les noms de villes, les couleurs, ou les types de produits.

---

[Retour en haut](#table-des-matières)

<a id="categorical-definition"></a>

#### Définition des Variables Catégoriques

Une variable catégorique est une variable qui prend des valeurs limitées et fixes, correspondant à différentes catégories ou classes. Contrairement aux variables numériques, qui peuvent prendre une large gamme de valeurs, les variables catégoriques se limitent à un ensemble prédéfini d'options.

**Exemples de Variables Catégoriques :**
- **Couleur** : `"Rouge"`, `"Bleu"`, `"Vert"`
- **Ville** : `"Montreal"`, `"Toronto"`, `"Vancouver"`
- **Type de produit** : `"Électronique"`, `"Vêtements"`, `"Alimentaire"`

Ces catégories n'ont pas nécessairement d'ordre naturel (par exemple, les couleurs ne sont pas ordonnées) et sont donc traitées différemment des variables ordinales, qui, elles, ont un ordre (par exemple, `"Petit"`, `"Moyen"`, `"Grand"`).

---

[Retour en haut](#table-des-matières)

<a id="categorical-importance"></a>

#### Importance pour l'IA

Les variables catégoriques sont cruciales pour les modèles d'IA car elles permettent de représenter des informations qualitatives qui ne peuvent pas être capturées par des variables numériques. En IA, de nombreux problèmes requièrent l'analyse de données catégoriques pour faire des prédictions ou pour classifier des objets.

**Raisons pour lesquelles les variables catégoriques sont importantes :**

1. **Représentation d'attributs qualitatifs** : Elles permettent de capturer des aspects qualitatifs des données, comme le type de produit, la couleur d'un objet, ou le lieu d'origine.
  
2. **Segmentation des données** : Les variables catégoriques aident à diviser les données en groupes distincts pour des analyses plus ciblées. Par exemple, on peut segmenter les ventes d'un produit par région (représentée par une variable catégorique telle que `"Ville"`).

3. **Entrée pour les modèles de classification** : Les modèles de classification utilisent souvent des variables catégoriques pour prédire des classes ou des catégories. Par exemple, un modèle pourrait prédire le type de produit qu'un client est susceptible d'acheter basé sur des variables catégoriques comme la catégorie du produit ou la région du client.

---

[Retour en haut](#table-des-matières)

<a id="one-hot-encoding"></a>

#### One-Hot Encoding

Le one-hot encoding est une méthode couramment utilisée pour convertir des variables catégoriques en un format numérique que les modèles de réseaux de neurones et autres algorithmes d'apprentissage automatique peuvent utiliser. Plutôt que d'assigner un numéro arbitraire à chaque catégorie (ce qui pourrait introduire un ordre implicite non désiré), le one-hot encoding crée une nouvelle variable binaire pour chaque catégorie.

**Comment ça fonctionne :**

- Supposons une variable catégorique `"Ville"` avec trois valeurs possibles : `"Montreal"`, `"Toronto"`, `"Vancouver"`.
- Le one-hot encoding va transformer cette variable en trois nouvelles variables binaires :
  - `"Montreal"` → `[1, 0, 0]`
  - `"Toronto"` → `[0, 1, 0]`
  - `"Vancouver"` → `[0, 0, 1]`

**Avantages du One-Hot Encoding :**

1. **Évite l'ordre implicite** : Contrairement à l'encodage ordinal, le one-hot encoding ne donne pas d'ordre implicite entre les catégories, ce qui est crucial lorsque les catégories ne sont pas naturellement ordonnées.
  
2. **Compatibilité avec les modèles** : Les réseaux de neurones et autres modèles d'apprentissage automatique nécessitent des données numériques en entrée. Le one-hot encoding permet de convertir les variables catégoriques en un format utilisable par ces modèles.

3. **Flexibilité** : Cette méthode est flexible et s'adapte bien à un large éventail de problèmes de classification.

**Inconvénients :**
- Le one-hot encoding peut entraîner une explosion du nombre de variables si la variable catégorique a un grand nombre de catégories, ce qui peut rendre les modèles plus complexes et gourmands en mémoire.

En résumé, le one-hot encoding est une méthode essentielle pour manipuler les variables catégoriques dans les réseaux de neurones et les autres modèles d'IA, permettant de transformer des informations qualitatives en données numériques utilisables.

---

[Retour en haut](#table-des-matières)

<a id="validation-data"></a>




















<hr/> 
<hr/> 
<hr/> 

### 5. Données de Validation  


<hr/> 
<hr/> 
<hr/> 


   [Retour en haut](#table-des-matières)

<a id="what-is-validation-data"></a>



### Qu'est-ce que les Données de Validation ?

Dans le cadre des réseaux de neurones, les données de validation jouent un rôle crucial pour évaluer la performance du modèle pendant l'entraînement. Les données de validation sont un sous-ensemble des données d'entraînement qui ne sont pas utilisées pour ajuster les poids du modèle, mais plutôt pour vérifier comment le modèle se généralise à des données qu'il n'a pas encore vues. Ce processus permet d'éviter le surapprentissage (ou overfitting), où le modèle s'adapte trop étroitement aux données d'entraînement et perd sa capacité à bien généraliser à de nouvelles données.

**Pourquoi les Données de Validation sont-elles importantes ?**

1. **Suivi de la performance** : Les données de validation permettent de suivre la performance du modèle en temps réel pendant l'entraînement. Si la performance sur les données de validation commence à se détériorer alors que celle sur les données d'entraînement continue de s'améliorer, cela peut indiquer que le modèle est en train de surapprendre.

2. **Ajustement des hyperparamètres** : Les données de validation sont utilisées pour ajuster les hyperparamètres du modèle, tels que le taux d'apprentissage, la taille des couches, le nombre de neurones, etc. Les hyperparamètres sont ajustés pour maximiser la performance sur les données de validation, car cela est plus représentatif de la performance sur des données non vues.

3. **Précision de la généralisation** : En utilisant les données de validation, on obtient une estimation de la capacité du modèle à généraliser, c'est-à-dire à bien performer sur des données nouvelles qui ne faisaient pas partie du jeu d'entraînement.

### Validation vs Test

**Données de Validation :**

Les données de validation, comme mentionné précédemment, sont utilisées pendant l'entraînement pour ajuster les hyperparamètres et pour prévenir le surapprentissage. Elles aident à déterminer quand arrêter l'entraînement pour éviter que le modèle ne devienne trop complexe et commence à surapprendre les détails spécifiques des données d'entraînement.

**Données de Test :**

Contrairement aux données de validation, les données de test sont un autre sous-ensemble de données, totalement séparées, qui ne sont jamais utilisées pendant l'entraînement ou la validation du modèle. Les données de test sont uniquement utilisées à la fin de l'entraînement pour évaluer de manière objective la performance finale du modèle. Elles fournissent une estimation de la performance du modèle sur des données réellement non vues, ce qui simule son comportement en production.

**Différences clés entre Validation et Test :**

1. **Utilisation** :
   - **Validation** : Utilisées pendant l'entraînement pour ajuster le modèle.
   - **Test** : Utilisées après l'entraînement pour évaluer la performance finale.

2. **Rôle dans l'entraînement** :
   - **Validation** : Permet d'ajuster les hyperparamètres et de surveiller le surapprentissage.
   - **Test** : Fournit une mesure finale de la qualité du modèle, souvent utilisée pour comparer différents modèles.

3. **Séparation des données** :
   - Les données de validation sont souvent un petit sous-ensemble des données d'entraînement, tandis que les données de test sont complètement distinctes et réservées uniquement à l'évaluation finale.

Dans un pipeline typique de réseaux de neurones, les données sont souvent divisées en trois ensembles : **entraînement**, **validation**, et **test**. Cette séparation est essentielle pour s'assurer que le modèle non seulement apprend bien, mais qu'il est aussi capable de généraliser à des situations qu'il n'a jamais rencontrées auparavant.


---

[Retour en haut](#table-des-matières)

<a id="normalization"></a>


















### 6. Normalisation en IA

La normalisation est une étape cruciale dans la préparation des données pour les réseaux de neurones et autres modèles d'intelligence artificielle. Elle permet de redimensionner les caractéristiques (ou features) des données d'entrée afin que toutes soient sur la même échelle. Cela facilite l'apprentissage du modèle et améliore souvent la vitesse de convergence et la précision du modèle.

---

[Retour en haut](#table-des-matières)

<a id="what-is-normalization"></a>

#### Qu'est-ce que la Normalisation ?

La normalisation consiste à transformer les valeurs des caractéristiques des données pour les amener dans une plage spécifique, souvent entre 0 et 1 ou -1 et 1. Cette transformation est particulièrement importante lorsque les caractéristiques des données ont des plages de valeurs différentes, ce qui peut affecter les performances du modèle de réseau de neurones.

**Méthodes de Normalisation :**
- **Min-Max Scaling (Échelle Min-Max)** : Une méthode courante de normalisation où les valeurs de chaque caractéristique sont redimensionnées pour se situer entre 0 et 1. Cela se fait en utilisant la formule suivante :
  
$$
X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

  Ici, *X* est la valeur d'origine, \( X_{min} \) est la valeur minimale de la caractéristique, et \( X_{max} \) est la valeur maximale. 

**Exemple :**
- Si une caractéristique "Prix" varie entre 10 et 200 dollars, une valeur de 150 dollars serait normalisée comme suit :
  
$$
X_{norm} = \frac{150 - 10}{200 - 10} = \frac{140}{190} \approx 0.737
$$

---

[Retour en haut](#table-des-matières)

<a id="normalization-importance"></a>

#### Processus et Importance

**Processus de Normalisation :**

1. **Identification des caractéristiques** : Déterminer quelles caractéristiques des données doivent être normalisées.
  
2. **Calcul des valeurs minimales et maximales** : Pour chaque caractéristique, calculer les valeurs minimale et maximale pour appliquer la transformation.

3. **Application de la transformation** : Redimensionner les valeurs en utilisant la méthode choisie (par exemple, Min-Max Scaling).

**Importance de la Normalisation :**

1. **Équilibrer les caractéristiques** : Si les caractéristiques ont des plages de valeurs très différentes, certaines pourraient dominer les calculs, ce qui rendrait l'entraînement moins efficace. La normalisation évite ce problème en mettant toutes les caractéristiques sur un pied d'égalité.

2. **Amélioration de la convergence** : Les modèles de réseaux de neurones convergent souvent plus rapidement et plus efficacement lorsque les données sont normalisées, car cela facilite le processus d'optimisation.

3. **Prévention des biais** : Sans normalisation, le modèle pourrait être biaisé en faveur des caractéristiques avec des valeurs plus grandes, ce qui pourrait conduire à des performances inférieures.

---

[Retour en haut](#table-des-matières)

<a id="keras-sequential"></a>

### 6.1. Standardisation vs Normalisation

Outre la normalisation, la standardisation est une autre technique couramment utilisée pour la mise à l'échelle des caractéristiques. La standardisation transforme les données pour qu'elles aient une moyenne de 0 et un écart-type de 1. Contrairement à la normalisation, qui redimensionne les valeurs dans une plage fixe, la standardisation ajuste les données en fonction de leur distribution statistique.

**Standardisation :**
- La standardisation utilise la formule suivante :
  
$$
X_{std} = \frac{X - \mu}{\sigma}
$$

  Ici, \( \mu \) est la moyenne des données, et \( \sigma \) est l'écart-type.

**Exemple :**
- Supposons une caractéristique "Taille" avec une moyenne de 170 cm et un écart-type de 10 cm. Une valeur de 180 cm serait standardisée comme suit :
  
$$
X_{std} = \frac{180 - 170}{10} = \frac{10}{10} = 1
$$

---

**Quand utiliser la Normalisation vs la Standardisation ?**

| **Critère**                     | **Normalisation (Min-Max Scaling)**                            | **Standardisation**                                 |
|---------------------------------|----------------------------------------------------------------|-----------------------------------------------------|
| **Objectif**                    | Redimensionner les caractéristiques entre une plage spécifique | Centrer les caractéristiques autour de 0 et 1       |
| **Utilisation typique**         | Lorsque les données ont des valeurs limites connues           | Lorsque les données suivent une distribution normale |
| **Plage des données**           | 0 à 1 (ou une autre plage fixée)                               | Pas de plage fixe                                   |
| **Impact sur les performances** | Peut améliorer la convergence dans les réseaux de neurones     | Peut améliorer la précision dans les modèles linéaires|
| **Exemples d'application**      | Images (valeurs de pixels), signaux audio                      | Données financières, variables biologiques          |

**Résumé :**
- **Utilisez la normalisation** lorsque vous souhaitez mettre toutes les caractéristiques sur une échelle similaire, surtout lorsque les valeurs extrêmes des données sont bien définies.
- **Utilisez la standardisation** lorsque vos données ont une distribution normale ou lorsque vous travaillez avec des modèles qui supposent une distribution normale des caractéristiques.

---

Ces techniques sont fondamentales pour préparer les données en vue d'une utilisation efficace dans les modèles d'intelligence artificielle, notamment les réseaux de neurones, et permettent d'améliorer à la fois la vitesse d'entraînement et la précision des prédictions.

---

[Retour en haut](#table-des-matières)









<hr/> 
<hr/> 
<hr/> 


### 7. Modèle Séquentiel en Keras

<hr/> 
<hr/> 
<hr/> 


Keras est une bibliothèque d'apprentissage profond hautement modulable et facile à utiliser, souvent utilisée pour construire et entraîner des modèles de réseaux de neurones. Le modèle séquentiel est l'une des structures les plus simples disponibles dans Keras, permettant de construire un modèle couche par couche de manière linéaire.

---

[Retour en haut](#table-des-matières)

<a id="sequential-model-creation"></a>

#### Création et Compréhension

Le modèle séquentiel en Keras est conçu pour permettre une construction simple et intuitive des réseaux de neurones. Il convient particulièrement bien aux modèles où les couches sont empilées les unes après les autres de manière séquentielle.

**Création d'un Modèle Séquentiel :**

1. **Initialisation** :
   - Pour commencer, vous initialisez un modèle séquentiel avec `Sequential()`.

   ```python
   from keras.models import Sequential

   model = Sequential()
   ```

2. **Ajout de Couches** :
   - Les couches sont ajoutées au modèle séquentiel une par une, en utilisant la méthode `add()`. Par exemple, une couche dense (Fully Connected Layer) peut être ajoutée comme suit :

   ```python
   from keras.layers import Dense

   model.add(Dense(units=64, activation='relu', input_shape=(100,)))
   ```

   Ici, `units=64` spécifie le nombre de neurones dans la couche, `activation='relu'` spécifie la fonction d'activation, et `input_shape=(100,)` définit la forme des données d'entrée.

3. **Compilation du Modèle** :
   - Après avoir ajouté toutes les couches, vous devez compiler le modèle en spécifiant la fonction de perte, l'optimiseur, et les métriques d'évaluation :

   ```python
   model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
   ```

4. **Entraînement du Modèle** :
   - Une fois le modèle compilé, vous pouvez l'entraîner en utilisant la méthode `fit()` avec les données d'entraînement :

   ```python
   model.fit(x_train, y_train, epochs=10, batch_size=32)
   ```

   Ici, `epochs=10` indique que le modèle va parcourir les données d'entraînement 10 fois, et `batch_size=32` spécifie la taille des lots de données utilisés pour l'entraînement.

5. **Évaluation et Prédiction** :
   - Après l'entraînement, vous pouvez évaluer le modèle sur des données de test ou faire des prédictions avec `evaluate()` et `predict()` :

   ```python
   model.evaluate(x_test, y_test)
   predictions = model.predict(x_new)
   ```

**Compréhension du Modèle Séquentiel :**

Le modèle séquentiel est linéaire, ce qui signifie que les données passent par chaque couche une à une, dans l'ordre où les couches ont été ajoutées. Cette simplicité fait du modèle séquentiel un excellent choix pour les débutants et pour des réseaux de neurones où cette structure convient, comme les réseaux fully connected.

**Limitations :**
- Le modèle séquentiel ne peut pas gérer des architectures complexes qui nécessitent des connexions multiples entre les couches (comme les réseaux de neurones convolutifs complexes ou les architectures avec des chemins parallèles).
- Pour des architectures plus complexes, il est préférable d'utiliser l'API fonctionnelle de Keras.

# Pour plus de détails, voir l'annexe 01.

---

[Retour en haut](#table-des-matières)

<a id="neural-layers"></a>






<hr/> 
<hr/> 
<hr/> 
### 8. Couches dans un Réseau de Neurones

Les réseaux de neurones sont construits à partir de différentes couches, chacune jouant un rôle spécifique dans le traitement des données et l'apprentissage des caractéristiques. Ces couches sont combinées pour former une architecture qui peut résoudre des problèmes complexes, comme la classification d'images, la reconnaissance de la parole, ou la prédiction de séries temporelles.

---

[Retour en haut](#table-des-matières)

<a id="flatten-layer"></a>

#### Couche Flatten

La couche Flatten est utilisée pour transformer une matrice de données multidimensionnelle en un vecteur unidimensionnel, qui peut ensuite être utilisé comme entrée pour des couches denses (fully connected). Cela est particulièrement utile dans les réseaux de neurones convolutifs (CNN), où les sorties des couches convolutives et de pooling sont en forme de tenseurs (par exemple, 2D ou 3D).

**Exemple d'utilisation :**
- Supposons que vous ayez une sortie de taille `(28, 28, 64)` après plusieurs couches convolutives. La couche Flatten convertit cela en un vecteur de `28 * 28 * 64 = 50176` valeurs, qui peut ensuite être utilisé comme entrée pour une couche dense.

```python
from keras.layers import Flatten

model.add(Flatten())
```

**Fonction :**
- La couche Flatten ne modifie pas les données en elles-mêmes, mais réorganise leur forme pour qu'elles puissent être traitées par des couches suivantes dans le réseau.

---

[Retour en haut](#table-des-matières)

<a id="output-layer"></a>

#### Couche de Sortie

La couche de sortie est la dernière couche d'un réseau de neurones, celle qui produit les prédictions finales du modèle. Le choix de la couche de sortie dépend du type de problème à résoudre :

- **Classification binaire** : Une couche dense avec un seul neurone et une fonction d'activation `sigmoid` est souvent utilisée.
  
  ```python
  model.add(Dense(1, activation='sigmoid'))
  ```

- **Classification multi-classes** : Une couche dense avec autant de neurones que de classes, et une fonction d'activation `softmax`, est couramment utilisée.
  
  ```python
  model.add(Dense(10, activation='softmax'))
  ```

- **Régression** : Pour les problèmes de régression, la couche de sortie est généralement une couche dense sans fonction d'activation ou avec une activation linéaire.

  ```python
  model.add(Dense(1))  # Activation linéaire par défaut
  ```

**Fonction :**
- La couche de sortie transforme les caractéristiques apprises par le réseau en une prédiction exploitable, adaptée au type de tâche (classification, régression, etc.).

---

[Retour en haut](#table-des-matières)

<a id="dense-layer"></a>

#### Couche Dense

La couche Dense, également appelée fully connected layer, est l'une des couches les plus couramment utilisées dans les réseaux de neurones. Chaque neurone dans une couche dense est connecté à tous les neurones de la couche précédente. C'est dans cette couche que se déroulent les calculs principaux, où les caractéristiques extraites par les couches précédentes sont combinées pour produire une décision ou une prédiction.

**Exemple d'utilisation :**

```python
from keras.layers import Dense

model.add(Dense(units=128, activation='relu'))
```

- **units=128** : Indique le nombre de neurones dans la couche.
- **activation='relu'** : Spécifie la fonction d'activation à utiliser pour chaque neurone.

**Fonction :**
- Les couches denses sont responsables de l'intégration des informations extraites des données pour former des prédictions complexes. Elles sont généralement placées après les couches convolutives ou récurrentes et jouent un rôle clé dans la décision finale du modèle.

---

Les couches dans un réseau de neurones travaillent ensemble pour transformer les données brutes en prédictions utiles. Chaque type de couche, de la couche Flatten à la couche Dense en passant par la couche de sortie, remplit un rôle spécifique, contribuant à la complexité et à la précision du modèle.

---

[Retour en haut](#table-des-matières)

<a id="activation-functions"></a>






<hr/> 
<hr/> 
<hr/> 


### 9. Fonction d'Activation

Les fonctions d'activation jouent un rôle crucial dans les réseaux de neurones en déterminant si un neurone doit être activé ou non, c'est-à-dire si l'information doit être transmise à travers le réseau. Elles introduisent de la non-linéarité dans le modèle, ce qui permet aux réseaux de neurones d'apprendre et de représenter des fonctions complexes.

---

[Retour en haut](#table-des-matières)

<a id="activation-definition"></a>

#### Qu'est-ce qu'une Fonction d'Activation ?

Une fonction d'activation est une fonction mathématique appliquée à la sortie de chaque neurone dans un réseau de neurones. Après qu'un neurone a calculé une somme pondérée de ses entrées, la fonction d'activation décide du signal final envoyé à la couche suivante. Sans ces fonctions, un réseau de neurones ne serait qu'une simple combinaison linéaire, incapable de capturer les relations complexes dans les données.

**Exemples de Fonctions d'Activation :**

- **Sigmoid** : 
  - Fonction utilisée principalement pour les modèles de classification binaire.
  - Convertit la sortie en une probabilité comprise entre 0 et 1.
  - Formule : 

$$
\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$

- **Tanh (Tangente hyperbolique)** : 
  - Fonction qui échelonne la sortie entre -1 et 1.
  - Utilisée lorsque des sorties négatives sont également importantes.
  - Formule : 

$$
\text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

- **ReLU (Rectified Linear Unit)** : 
  - Fonction d'activation la plus couramment utilisée dans les réseaux de neurones modernes.
  - Transforme les valeurs négatives en 0 et laisse passer les valeurs positives telles quelles.
  - Formule :
  
$$
\text{ReLU}(x) = \max(0, x)
$$

- **Softmax** :
  - Fonction utilisée dans la couche de sortie pour les problèmes de classification multi-classes.
  - Convertit les sorties en probabilités qui totalisent 1.
  - Formule : 

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

---

[Retour en haut](#table-des-matières)

<a id="relu-vs-softmax"></a>

#### ReLU vs Softmax

**ReLU (Rectified Linear Unit) :**

ReLU est une fonction d'activation simple mais extrêmement efficace. Elle est définie comme étant 0 pour toute entrée négative et égale à l'entrée elle-même pour toute entrée positive. Cette simplicité lui permet de surmonter certains des problèmes associés à d'autres fonctions d'activation, comme la disparition du gradient (vanishing gradient problem) rencontrée avec Sigmoid ou Tanh.

**Caractéristiques de ReLU :**

- **Non-linéarité** : Permet au réseau de capturer des relations complexes.
- **Simplicité** : Très rapide à calculer, ce qui améliore la vitesse d'entraînement.
- **Évitement de la saturation** : Contrairement à Sigmoid ou Tanh, ReLU ne sature pas pour des valeurs positives, ce qui aide à maintenir un gradient non nul.

**Usage :**
- ReLU est souvent utilisée dans les couches cachées des réseaux de neurones pour introduire de la non-linéarité tout en étant efficace en termes de calcul.

**Softmax :**

Softmax est une fonction d'activation utilisée principalement dans les couches de sortie pour les problèmes de classification multi-classes. Elle transforme un vecteur de valeurs arbitraires en un vecteur de probabilités, où chaque valeur représente la probabilité que l'entrée appartienne à une certaine classe.

**Caractéristiques de Softmax :**

- **Probabilités** : Convertit les sorties en probabilités, ce qui est utile pour les tâches de classification.
- **Normalisation** : La somme de toutes les probabilités est toujours égale à 1, ce qui permet une interprétation directe comme distribution de probabilités.

**Usage :**
- Softmax est utilisée dans la couche de sortie d'un réseau de neurones lorsqu'on veut classifier une entrée parmi plusieurs classes possibles. Par exemple, dans un modèle de classification d'images qui distingue entre 10 catégories, Softmax produira un vecteur de probabilités de taille 10.

**Résumé des Différences :**

- **ReLU** : Utilisée dans les couches cachées pour maintenir un gradient efficace, fonctionne bien pour des problèmes où la saturation n'est pas souhaitée.
- **Softmax** : Utilisée dans la couche de sortie pour des problèmes de classification multi-classes, où il est nécessaire de prédire une classe parmi plusieurs.

---

Les fonctions d'activation, qu'il s'agisse de ReLU ou de Softmax, jouent un rôle crucial dans le fonctionnement des réseaux de neurones. Elles permettent au réseau d'apprendre des relations complexes et d'effectuer des prédictions qui peuvent être interprétées et utilisées dans des applications réelles.



# Table de comparaison entre les différentes fonctions d'activation couramment utilisées dans les réseaux de neurones :



| **Fonction d'Activation** | **Formule**                                                | **Plage de Valeurs**  | **Usage Typique**                       | **Avantages**                                        | **Inconvénients**                                     |
|---------------------------|------------------------------------------------------------|-----------------------|------------------------------------------|-----------------------------------------------------|------------------------------------------------------|
| **Sigmoid**                | Equation 1                                                 | (0, 1)                | Couches de sortie pour la classification binaire | Sortie entre 0 et 1, interprétable comme probabilité | Saturation pour les grandes valeurs de x (vanishing gradient), computationnellement coûteuse |
| **Tanh**                   | Equation 2                                                 | (-1, 1)               | Couches cachées, certaines tâches de régression | Centre les données autour de 0, bonne convergence | Saturation pour les grandes valeurs de x (vanishing gradient), computationnellement coûteuse |
| **ReLU**                   | Equation 3                                                 | [0, +∞)               | Couches cachées dans des réseaux profonds | Simple, computation rapide, évite le vanishing gradient | Peut entraîner des neurones morts (dead neurons), pas de sortie négative |
| **Leaky ReLU**             | Equation 4                                                 | (-∞, +∞)              | Couches cachées, alternative à ReLU      | Évite le problème des neurones morts de ReLU, maintien du gradient pour les valeurs négatives | La valeur de \(\alpha\) doit être choisie manuellement |
| **Softmax**                | Equation 5                                                 | (0, 1)                | Couches de sortie pour la classification multi-classes | Sortie interprétable comme probabilité, normalisation | Saturation des petites valeurs de x, computationnellement coûteuse |
| **Linear**                 | Equation 6                                                 | (-∞, +∞)              | Couches de sortie pour la régression      | Simplicité, pas de saturation                        | Pas de non-linéarité, limité aux problèmes linéaires  |

---

### Équations

- **Equation 1 (Sigmoid)** :

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

- **Equation 2 (Tanh)** :

$$
\text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

- **Equation 3 (ReLU)** :

$$
  \text{ReLU}(x) = \max(0, x)
$$

- **Equation 4 (Leaky ReLU)** :

$$
\text{Leaky ReLU}(x) = \max(\alpha x, x)
$$

- **Equation 5 (Softmax)** :

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

- **Equation 6 (Linear)** :

$$
f(x) = x
$$


Voici la suite :

---

**Résumé :**

- **Sigmoid** (Equation 1) et **Tanh** (Equation 2) sont principalement utilisées pour des tâches où la sortie doit être comprise dans une plage limitée (0,1 pour Sigmoid ou -1,1 pour Tanh) et où une interprétation probabiliste est utile. Cependant, elles peuvent souffrir de la saturation, ce qui entraîne un problème de vanishing gradient, rendant l'entraînement des réseaux de neurones plus difficile.

- **ReLU** (Equation 3) est la fonction d'activation la plus courante dans les réseaux profonds modernes en raison de sa simplicité et de sa performance. Elle est particulièrement efficace pour les réseaux profonds car elle évite le problème de vanishing gradient, mais elle peut parfois entraîner des neurones morts (c'est-à-dire des neurones qui cessent de s'activer), surtout si les valeurs négatives dominent l'entraînement.

- **Leaky ReLU** (Equation 4) est une variante de ReLU qui tente de résoudre le problème des neurones morts en permettant une petite pente pour les valeurs négatives. Cela maintient le gradient même pour les valeurs négatives de l'entrée, ce qui peut aider à améliorer la robustesse du modèle.

- **Softmax** (Equation 5) est indispensable pour les problèmes de classification multi-classes. Elle convertit les sorties du réseau en probabilités, où la somme de toutes les sorties est égale à 1, permettant de prédire à quelle classe appartient une entrée donnée. Softmax est généralement utilisée dans la couche de sortie des réseaux de neurones pour les tâches de classification.

- **Linear** (Equation 6) est utilisée dans les couches de sortie pour les tâches de régression, où l'objectif est de prédire une valeur continue. Cette fonction d'activation est simple et n'introduit aucune non-linéarité, ce qui la rend adaptée pour les problèmes où une relation linéaire entre l'entrée et la sortie est suffisante.



---

[Retour en haut](#table-des-matières)

<a id="network-architecture"></a>



<hr/> 
<hr/> 
<hr/> 




### 10. Architecture de Réseau de Neurones

L'architecture d'un réseau de neurones est un élément clé qui détermine sa capacité à apprendre et à généraliser les connaissances à partir des données. La conception de l'architecture implique plusieurs décisions critiques, telles que le choix du nombre de couches, le nombre de neurones par couche, et la manière dont ces couches sont connectées. Une architecture bien conçue peut améliorer considérablement la performance du modèle, tandis qu'une mauvaise conception peut limiter sa capacité à apprendre efficacement.

---

[Retour en haut](#table-des-matières)

<a id="neuron-layer-choices"></a>

#### Choix des Neurones et des Couches

Le choix du nombre de couches et de neurones dans chaque couche dépend largement de la complexité du problème à résoudre et de la quantité de données disponibles. Voici les principales considérations à prendre en compte lors de la conception de l'architecture d'un réseau de neurones :

1. **Couches d'Entrée :**
   - La couche d'entrée est la première couche du réseau, et son rôle est de recevoir les données brutes. Le nombre de neurones dans la couche d'entrée correspond au nombre de caractéristiques dans les données d'entrée. Par exemple, pour une image de taille 28x28 pixels en niveaux de gris, la couche d'entrée aura 784 neurones (28*28).

2. **Couches Cachées :**
   - Les couches cachées sont situées entre la couche d'entrée et la couche de sortie. Ce sont elles qui effectuent la majeure partie du traitement dans le réseau, extrayant les caractéristiques pertinentes des données d'entrée.
   - **Nombre de Couches** : Un réseau de neurones peut avoir une ou plusieurs couches cachées. Plus le problème est complexe, plus il peut être utile d'ajouter des couches cachées, mais cela augmente aussi le risque de surapprentissage.
   - **Nombre de Neurones par Couche** : Chaque couche cachée peut avoir un nombre différent de neurones. Généralement, on commence par un grand nombre de neurones dans les premières couches cachées, puis on réduit progressivement leur nombre dans les couches suivantes.

3. **Couche de Sortie :**
   - La couche de sortie produit la prédiction finale du réseau. Le nombre de neurones dans la couche de sortie dépend du type de problème :
     - **Classification binaire** : Un seul neurone avec une fonction d'activation sigmoid.
     - **Classification multi-classes** : Un neurone par classe avec une fonction d'activation softmax.
     - **Régression** : Un seul neurone avec une activation linéaire (ou aucune activation).

**Exemple de Conception :**

- Pour un problème de classification d'images (10 classes), une architecture typique pourrait ressembler à ceci :
  - **Couche d'entrée** : 784 neurones (pour une image 28x28)
  - **Couche cachée 1** : 512 neurones, activation ReLU
  - **Couche cachée 2** : 256 neurones, activation ReLU
  - **Couche de sortie** : 10 neurones, activation softmax

---

[Retour en haut](#table-des-matières)

<a id="choosing-neurons"></a>

#### Choisir le Nombre de Neurones

Le choix du nombre de neurones dans chaque couche est crucial car il affecte directement la capacité du réseau à apprendre des relations complexes dans les données, mais aussi sa tendance à surapprendre.

1. **Sous-dimensionnement :**
   - Si vous choisissez trop peu de neurones, le réseau peut ne pas avoir la capacité d'apprendre les relations complexes présentes dans les données. Cela conduit souvent à un problème appelé **underfitting**, où le modèle est trop simple pour capturer les nuances des données.

2. **Sur-dimensionnement :**
   - Si vous choisissez trop de neurones, le réseau peut devenir trop complexe, avec un grand nombre de paramètres à ajuster. Cela peut entraîner un **overfitting**, où le modèle s'adapte trop étroitement aux données d'entraînement, perdant sa capacité à généraliser à de nouvelles données.

3. **Règles Empiriques :**
   - Il n'existe pas de formule magique pour choisir le nombre de neurones, mais quelques règles empiriques peuvent guider la conception :
     - Commencez par un nombre de neurones proportionnel au nombre de caractéristiques dans les données.
     - Augmentez le nombre de neurones si le modèle semble underfitting.
     - Réduisez le nombre de neurones ou utilisez des techniques de régularisation (comme le dropout) si le modèle semble overfitting.

4. **Expérimentation :**
   - La conception optimale de l'architecture nécessite souvent des expérimentations répétées. Il est courant de tester différentes configurations en utilisant la validation croisée pour déterminer l'architecture qui offre le meilleur compromis entre performance et complexité.

**Exemple :**
- Pour un réseau de neurones simple destiné à la reconnaissance de chiffres manuscrits (comme le dataset MNIST), vous pourriez commencer avec une architecture comme suit :
  - **Couche d'entrée** : 784 neurones
  - **Couche cachée 1** : 128 neurones, activation ReLU
  - **Couche cachée 2** : 64 neurones, activation ReLU
  - **Couche de sortie** : 10 neurones, activation softmax

Si ce modèle montre des signes d'underfitting, vous pourriez augmenter le nombre de neurones à 256 et 128 dans les couches cachées, respectivement.

---

Le choix de l'architecture du réseau de neurones, y compris le nombre de couches et de neurones, est une étape critique dans le développement de modèles performants. Cela nécessite une compréhension approfondie des données et des objectifs, ainsi qu'une expérimentation attentive pour trouver le meilleur équilibre entre complexité et généralisation.




----

# Exemple de Choix du Nombre de Neurones Basé sur les Caractéristiques des Données

Supposons que vous travaillez sur un problème de classification où vous avez un jeu de données avec 20 caractéristiques (features). Les caractéristiques peuvent représenter des attributs comme l'âge, le revenu, la taille, etc. Vous souhaitez créer un réseau de neurones pour prédire une classe parmi trois possibles.

**Règle empirique** : Une règle simple consiste à commencer par un nombre de neurones dans la première couche cachée qui est proportionnel au nombre de caractéristiques. Une approche courante est de choisir un nombre de neurones dans la première couche cachée qui est égal ou légèrement supérieur au nombre de caractéristiques.

**Exemple :**

- **Nombre de caractéristiques** : 20
- **Nombre de neurones dans la première couche cachée** : 20 à 40 neurones

### Architecture proposée :

1. **Couche d'entrée** : 20 neurones (une pour chaque caractéristique)
2. **Couche cachée 1** : 30 neurones (nombre proportionnel au nombre de caractéristiques)
3. **Couche cachée 2** : 15 neurones (réduction pour capturer des relations plus abstraites)
4. **Couche de sortie** : 3 neurones (une pour chaque classe possible)

### Schéma en ASCII :

```
Input Layer (20 neurones)
+--------------------+
|  O  O  O  O  O  O  |
|  O  O  O  O  O  O  |
|  O  O  O  O  O  O  |
+--------------------+
          |
Hidden Layer 1 (30 neurones)
+-------------------------------+
|  O  O  O  O  O  O  O  O  O  O  |
|  O  O  O  O  O  O  O  O  O  O  |
|  O  O  O  O  O  O  O  O  O  O  |
+-------------------------------+
          |
Hidden Layer 2 (15 neurones)
+------------------------+
|  O  O  O  O  O  O  O  O |
|  O  O  O  O  O  O  O  O |
|  O  O  O  O  O  O  O  O |
+------------------------+
          |
Output Layer (3 neurones)
+-----+
|  O  |
|  O  |
|  O  |
+-----+
```

### Explication :

- **Couche d'entrée** : La couche d'entrée a 20 neurones, chacun correspondant à une caractéristique de l'entrée.
- **Couche cachée 1** : Basée sur la règle empirique, nous avons choisi 30 neurones, légèrement plus que le nombre de caractéristiques, pour capturer les relations complexes.
- **Couche cachée 2** : Le nombre de neurones est réduit à 15 pour capter des abstractions plus fines après la première couche.
- **Couche de sortie** : Trois neurones, chacun correspondant à l'une des classes à prédire.

Cette approche permet de démarrer avec une architecture simple et d'augmenter ou de réduire le nombre de neurones en fonction des performances obtenues après quelques essais.





---

[Retour en haut](#table-des-matières)





-----

# Annexe 01 - Limitations de réseaux de neurones

---

### Exemple de Limitation avec un Réseau de Neurones Convolutifs Complexe

Le modèle séquentiel est idéal pour des architectures simples où les couches sont empilées les unes après les autres. Cependant, il devient limité lorsqu'il s'agit de réseaux de neurones convolutifs (CNN) complexes qui nécessitent des connexions plus sophistiquées entre les couches, comme des chemins parallèles ou des couches qui partagent des entrées et se rejoignent plus tard dans le réseau.

**Exemple d'Architecture Complexe : Réseau Inception**

Un exemple typique est l'architecture Inception, utilisée dans des réseaux de neurones convolutifs avancés comme **InceptionNet**. Dans cette architecture, plusieurs convolutions de différentes tailles (par exemple, 1x1, 3x3, 5x5) sont appliquées en parallèle sur la même entrée, puis les résultats sont concaténés en une seule sortie.

```python
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate
from keras.models import Model

input_img = Input(shape=(256, 256, 3))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = concatenate([tower_1, tower_2, tower_3], axis=1)

model = Model(inputs=input_img, outputs=output)
```

Dans cet exemple, trois branches distinctes (ou "tours") sont appliquées en parallèle sur la même image d'entrée, et leurs sorties sont ensuite fusionnées (concatenated) pour former la sortie du bloc Inception. Ce type de structure ne peut pas être implémenté avec un modèle séquentiel car il nécessite des connexions multiples et des fusions complexes entre les couches.



```
                      +----------------------------+
                      |         Input Layer        |
                      |     Shape: (256, 256, 3)   |
                      +----------------------------+
                                   |
     +-----------------------------+--------------------------------+
     |                             |                                |
+----v----+                   +----v----+                      +----v----+
|  Conv2D |                   |  Conv2D |                      | MaxPool |
| 1x1, 64 |                   | 1x1, 64 |                      | 3x3     |
+---------+                   +---------+                      +---------+
     |                             |                                |
+----v----+                   +----v----+                      +----v----+
|  Conv2D |                   |  Conv2D |                      |  Conv2D |
| 3x3, 64 |                   | 5x5, 64 |                      | 1x1, 64 |
+---------+                   +---------+                      +---------+
     |                             |                                |
     +-----------------------------+--------------------------------+
                                   |
                      +------------v------------+
                      |        Concatenate       |
                      +--------------------------+
                                   |
                      +------------v------------+
                      |       Output Layer       |
                      +--------------------------+
```

**Description du schéma :**

1. **Input Layer** : C'est l'entrée du réseau, qui reçoit une image de taille 256x256 avec 3 canaux (par exemple, une image en couleur RGB).

2. **Branch 1 (à gauche)** :
   - Une première convolution `1x1` avec 64 filtres.
   - Suivie d'une autre convolution `3x3` avec 64 filtres.

3. **Branch 2 (au centre)** :
   - Une première convolution `1x1` avec 64 filtres.
   - Suivie d'une convolution `5x5` avec 64 filtres.

4. **Branch 3 (à droite)** :
   - Un max pooling `3x3` avec un stride de `1x1`.
   - Suivi d'une convolution `1x1` avec 64 filtres.

5. **Concatenate** : Les sorties des trois branches sont ensuite fusionnées (concatenated) pour former une seule sortie.

6. **Output Layer** : La sortie finale qui résulte de la concaténation des trois chemins.





**Conclusion :**
- Pour des architectures simples où les couches sont simplement empilées, le modèle séquentiel de Keras est suffisant.
- Pour des architectures complexes comme Inception, où les couches peuvent avoir des connexions parallèles ou fusionner à différents points, l'API fonctionnelle de Keras est nécessaire pour créer et gérer ces structures.
- C'est pourquoi le modèle séquentiel n'est pas adapté à des réseaux de neurones convolutifs complexes et pourquoi l'API fonctionnelle de Keras est une meilleure option pour ces cas.



---

# Annexe 02 - Qu'est-ce que les Hyperparamètres dans un Réseau de Neurones ?

---

### Qu'est-ce que les Hyperparamètres dans un Réseau de Neurones ?

Les hyperparamètres sont des paramètres qui contrôlent la structure et l'apprentissage d'un réseau de neurones. Contrairement aux paramètres internes du modèle (comme les poids des connexions entre neurones), les hyperparamètres ne sont pas appris à partir des données, mais sont définis à l'avance par l'utilisateur et influencent le comportement du modèle. Leur réglage correct est essentiel pour obtenir de bonnes performances du modèle.

Voici quelques exemples d'hyperparamètres couramment utilisés dans les réseaux de neurones :

#### 1. Nombre de Couches (Architecture du Réseau)
Le nombre de couches dans un réseau de neurones, également appelé profondeur du réseau, est l'un des hyperparamètres les plus importants. Un réseau peut avoir une seule couche cachée (ce qu'on appelle un perceptron multicouche simple) ou plusieurs couches cachées, formant un réseau profond.

- **Réseau peu profond** : Un réseau avec une ou deux couches cachées. Convient aux problèmes simples où les relations dans les données ne sont pas trop complexes.
- **Réseau profond** : Un réseau avec de nombreuses couches cachées. Utilisé pour des problèmes complexes comme la reconnaissance d'images, où plusieurs niveaux d'abstraction sont nécessaires.

#### 2. Nombre de Neurones par Couche
Le nombre de neurones dans chaque couche cachée détermine la capacité du réseau à capturer des informations. Plus il y a de neurones, plus le modèle peut potentiellement apprendre des relations complexes, mais cela augmente également le risque de surapprentissage.

- **Trop peu de neurones** : Le modèle risque de sous-apprendre (underfitting) car il ne dispose pas de suffisamment de capacité pour capturer toutes les nuances des données.
- **Trop de neurones** : Le modèle peut surapprendre (overfitting), s'adaptant trop étroitement aux données d'entraînement et perdant sa capacité à généraliser à de nouvelles données.

#### 3. Taux d'Apprentissage (Learning Rate)
Le taux d'apprentissage est un hyperparamètre qui détermine la taille des ajustements apportés aux poids du réseau à chaque étape de l'apprentissage. Un taux d'apprentissage élevé peut accélérer l'entraînement, mais risque de dépasser le minimum global. Un taux d'apprentissage faible, en revanche, permet des ajustements plus fins mais peut rendre l'entraînement très lent.

- **Taux d'apprentissage élevé** : Risque de ne pas converger ou de sauter par-dessus le minimum.
- **Taux d'apprentissage faible** : Convergence plus stable mais peut nécessiter un grand nombre d'itérations.

#### 4. Taille des Mini-Lots (Batch Size)
Lors de l'entraînement d'un réseau de neurones, les données d'entraînement sont souvent divisées en petits lots, ou mini-batches, pour permettre une mise à jour des poids plus fréquente. La taille des mini-batches est un hyperparamètre qui peut affecter la vitesse d'entraînement et la qualité du modèle.

- **Mini-batch de grande taille** : Moins de bruit dans les mises à jour des poids, mais nécessite plus de mémoire.
- **Mini-batch de petite taille** : Mises à jour plus fréquentes et plus de bruit, ce qui peut aider le modèle à sortir des minima locaux.

#### 5. Nombre d'Époques (Epochs)
Une époque correspond à un passage complet sur l'ensemble des données d'entraînement. Le nombre d'époques est un hyperparamètre qui détermine combien de fois le modèle passe par les données pour ajuster ses poids.

- **Trop peu d'époques** : Le modèle peut ne pas avoir suffisamment appris, conduisant à un underfitting.
- **Trop d'époques** : Le modèle peut surapprendre les données d'entraînement, conduisant à un overfitting.

#### 6. Fonction d'Activation
La fonction d'activation est l'hyperparamètre qui détermine comment les sorties des neurones sont transformées avant d'être transmises à la couche suivante. Des fonctions comme ReLU, Sigmoid, et Tanh sont couramment utilisées.

- **ReLU** : Rapide et efficace pour les couches cachées.
- **Softmax** : Utilisé dans la couche de sortie pour la classification multi-classes.
- **Sigmoid** : Utilisé dans la couche de sortie pour la classification binaire.

#### 7. Méthode de Régularisation
La régularisation est un ensemble de techniques visant à prévenir le surapprentissage en pénalisant les poids du modèle. Les méthodes de régularisation sont des hyperparamètres qui peuvent être ajustés pour équilibrer l'apprentissage.

- **Dropout** : Désactive de manière aléatoire un certain pourcentage de neurones à chaque étape d'apprentissage, ce qui aide à éviter la co-dépendance entre les neurones.
- **L2 Regularization** : Ajoute une pénalité aux poids élevés, encourageant des poids plus petits.

### Schéma en ASCII des Hyperparamètres dans un Réseau de Neurones

```
+--------------------------------+
|         Hyperparamètres         |
+--------------------------------+
|  Nombre de Couches : 3          |
|  Nombre de Neurones (C1) : 64   |
|  Nombre de Neurones (C2) : 32   |
|  Nombre de Neurones (C3) : 16   |
|  Taux d'Apprentissage : 0.001   |
|  Taille des Mini-Batches : 32   |
|  Nombre d'Époques : 50          |
|  Fonction d'Activation : ReLU   |
|  Régularisation : Dropout 0.5   |
+--------------------------------+
         |
Input Layer  ---> Hidden Layer 1 (64) ---> Hidden Layer 2 (32) ---> Hidden Layer 3 (16) ---> Output Layer
         |
+--------------------------------+
|    Ajustement des poids         |
|         (Learning Rate)         |
+--------------------------------+
```

### Conclusion

Les hyperparamètres sont essentiels pour configurer et entraîner un réseau de neurones efficacement. Leur choix et leur ajustement nécessitent souvent une série d'expérimentations et de validations croisées pour trouver la meilleure combinaison qui offre un compromis entre performance et généralisation. L'optimisation des hyperparamètres peut impliquer des techniques comme la recherche en grille (grid search), la recherche aléatoire (random search), ou l'optimisation bayésienne pour automatiser le processus et obtenir des résultats optimaux.


----

# Annexe 3 - exemple d'hyperparamètres

---



- Je vous présente des exemples de code Python utilisant Keras pour illustrer comment chaque hyperparamètre peut être modifié et comment cela affecte la conception et l'entraînement d'un réseau de neurones.
- Pour chaque hyperparamètre, j'ai inclus un exemple de code où il est modifié pour montrer sa variabilité.

### 1. Nombre de Couches (Architecture du Réseau)

**Exemple de code avec un réseau peu profond :**

```python
from keras.models import Sequential
from keras.layers import Dense

# Modèle avec une seule couche cachée
model = Sequential()
model.add(Dense(64, input_shape=(20,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sortie pour la classification binaire

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**Exemple de code avec un réseau profond :**

```python
# Modèle avec plusieurs couches cachées
model = Sequential()
model.add(Dense(128, input_shape=(20,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### 2. Nombre de Neurones par Couche

**Exemple de code avec peu de neurones :**

```python
# Modèle avec moins de neurones par couche
model = Sequential()
model.add(Dense(16, input_shape=(20,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**Exemple de code avec plus de neurones :**

```python
# Modèle avec plus de neurones par couche
model = Sequential()
model.add(Dense(128, input_shape=(20,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### 3. Taux d'Apprentissage (Learning Rate)

**Exemple de code avec un taux d'apprentissage élevé :**

```python
from keras.optimizers import Adam

# Modèle avec un taux d'apprentissage élevé
model = Sequential()
model.add(Dense(64, input_shape=(20,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.01)  # Taux d'apprentissage élevé
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
```

**Exemple de code avec un taux d'apprentissage faible :**

```python
# Modèle avec un taux d'apprentissage faible
optimizer = Adam(learning_rate=0.0001)  # Taux d'apprentissage faible
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
```

### 4. Taille des Mini-Lots (Batch Size)

**Exemple de code avec une taille de mini-batch petite :**

```python
# Entraînement avec une petite taille de mini-batch
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))
```

**Exemple de code avec une taille de mini-batch grande :**

```python
# Entraînement avec une grande taille de mini-batch
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val))
```

### 5. Nombre d'Époques (Epochs)

**Exemple de code avec peu d'époques :**

```python
# Entraînement avec un petit nombre d'époques
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))
```

**Exemple de code avec beaucoup d'époques :**

```python
# Entraînement avec un grand nombre d'époques
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
```

### 6. Fonction d'Activation

**Exemple de code avec ReLU :**

```python
# Modèle utilisant ReLU comme fonction d'activation
model = Sequential()
model.add(Dense(64, input_shape=(20,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**Exemple de code avec Tanh :**

```python
# Modèle utilisant Tanh comme fonction d'activation
model = Sequential()
model.add(Dense(64, input_shape=(20,), activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### 7. Méthode de Régularisation

**Exemple de code avec Dropout :**

```python
from keras.layers import Dropout

# Modèle avec Dropout
model = Sequential()
model.add(Dense(128, input_shape=(20,), activation='relu'))
model.add(Dropout(0.5))  # 50% des neurones désactivés aléatoirement
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**Exemple de code avec L2 Regularization :**

```python
from keras.regularizers import l2

# Modèle avec L2 regularization
model = Sequential()
model.add(Dense(128, input_shape=(20,), activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

# Résumé : 


- La table ci-dessous présente des exemples d'hyperparamètres couramment utilisés dans les réseaux de neurones, avec une brève description et des exemples de valeurs typiques ou configurations :

| **Hyperparamètre**           | **Description**                                                                 | **Exemples de Valeurs/Configurations**                                      |
|------------------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Nombre de Couches**        | Détermine la profondeur du réseau, c'est-à-dire combien de couches cachées il contient. | Réseau peu profond : 1 ou 2 couches cachées<br>Réseau profond : 5 à 20 couches cachées |
| **Nombre de Neurones par Couche** | Nombre de neurones dans chaque couche cachée. Plus de neurones peuvent capturer des relations complexes, mais risquent de surapprendre. | Couches cachées : 32, 64, 128, 256 neurones<br>Couche de sortie : 1 (binaire), n (multi-classes) |
| **Taux d'Apprentissage (Learning Rate)** | Contrôle la taille des ajustements des poids à chaque itération. Un taux d'apprentissage trop élevé peut empêcher la convergence, tandis qu'un taux trop faible peut ralentir l'apprentissage. | Haut : 0.01<br>Moyen : 0.001<br>Bas : 0.0001 |
| **Taille des Mini-Lots (Batch Size)** | Nombre de données d'entraînement utilisées pour une seule mise à jour des poids. Les petits lots offrent plus de bruit mais des mises à jour plus fréquentes. | Petite : 16, 32<br>Grande : 64, 128, 256 |
| **Nombre d'Époques (Epochs)** | Nombre de fois que le modèle voit l'ensemble des données d'entraînement. Trop d'époques peuvent entraîner un surapprentissage. | Peu : 5, 10<br>Beaucoup : 50, 100, 200 |
| **Fonction d'Activation**    | Fonction appliquée à la sortie de chaque neurone pour introduire de la non-linéarité. | Couches cachées : ReLU, Tanh<br>Couches de sortie : Sigmoid (binaire), Softmax (multi-classes) |
| **Méthode de Régularisation**| Techniques pour prévenir le surapprentissage en pénalisant les poids ou en désactivant certains neurones pendant l'entraînement. | Dropout : 0.2, 0.5<br>L2 Regularization : 0.01, 0.001 |
| **Optimiseur**               | Algorithme qui ajuste les poids du modèle en fonction de la fonction de perte.  | SGD (Gradient Descent), Adam, RMSprop                                        |
| **Initialisation des Poids** | Méthode pour initialiser les poids des neurones. Une mauvaise initialisation peut ralentir ou empêcher la convergence. | Initialisation aléatoire, He Initialization, Xavier Initialization          |
| **Taille de la Fenêtre**     | Pour les réseaux convolutifs, la taille de la fenêtre de convolution détermine combien de pixels sont utilisés pour calculer chaque neurone de sortie. | 3x3, 5x5, 7x7                                                               |

### Explication de la Table

- **Nombre de Couches** : Ce paramètre définit combien de transformations non-linéaires les données subissent avant d'atteindre la sortie. Un réseau plus profond peut modéliser des fonctions plus complexes.
- **Nombre de Neurones par Couche** : Le nombre de neurones dans chaque couche cachée détermine la capacité du réseau à capturer des relations dans les données. Un nombre trop faible peut limiter la capacité du modèle, tandis qu'un nombre trop élevé peut conduire à un surapprentissage.
- **Taux d'Apprentissage** : Ce paramètre contrôle la vitesse à laquelle le modèle apprend. Un taux trop élevé risque de rendre l'apprentissage instable, tandis qu'un taux trop faible peut ralentir l'apprentissage de manière significative.
- **Taille des Mini-Lots** : La taille des mini-lots influence la stabilité et la vitesse de l'apprentissage. Les petits lots peuvent introduire du bruit, ce qui aide parfois à sortir des minima locaux, tandis que les grands lots fournissent des mises à jour plus stables.
- **Nombre d'Époques** : Détermine combien de fois l'algorithme d'apprentissage passera par l'ensemble des données d'entraînement. Trop d'époques peuvent entraîner un surapprentissage, tandis que trop peu peuvent conduire à un sous-apprentissage.
- **Fonction d'Activation** : Les fonctions d'activation déterminent comment les signaux sont propagés dans le réseau, en introduisant des non-linéarités qui permettent au modèle de capturer des relations complexes.
- **Méthode de Régularisation** : Les techniques comme le Dropout ou la régularisation L2 aident à prévenir le surapprentissage en rendant le modèle moins sensible aux particularités du jeu de données d'entraînement.
- **Optimiseur** : L'optimiseur ajuste les poids en fonction de la fonction de perte. Différents optimisateurs conviennent mieux à différentes architectures et types de données.
- **Initialisation des Poids** : Les poids initiaux du réseau influencent la vitesse de convergence et la probabilité de trouver une solution optimale. Une bonne initialisation est essentielle pour un entraînement efficace.
- **Taille de la Fenêtre** : Pour les réseaux convolutifs, la taille de la fenêtre détermine la zone d'entrée regardée par chaque neurone, ce qui influence la manière dont les caractéristiques spatiales sont capturées.


