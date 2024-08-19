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

#### - Entraînement et Test  
   [Retour en haut](#table-des-matières)

<a id="data-split-importance"></a>

#### - Importance de la Division  
   [Retour en haut](#table-des-matières)

<a id="data-types"></a>



<hr/> 
<hr/> 
<hr/> 

### 3. Type de Données pour l'IA  

<hr/> 
<hr/> 
<hr/> 


   [Retour en haut](#table-des-matières)

<a id="float32-vs-float64"></a>

#### - Types de Données : float32 vs float64  
   [Retour en haut](#table-des-matières)

<a id="numeric-vs-string"></a>

#### - Numérique vs Chaîne de Caractères  
   [Retour en haut](#table-des-matières)

<a id="categorical-variables"></a>


<hr/> 
<hr/> 
<hr/> 

### 4. Variables Catégoriques  


<hr/> 
<hr/> 
<hr/> 


   [Retour en haut](#table-des-matières)

<a id="categorical-definition"></a>

#### - Définition des Variables Catégoriques  
   [Retour en haut](#table-des-matières)

<a id="categorical-importance"></a>

#### - Importance pour l'IA  
   [Retour en haut](#table-des-matières)

<a id="one-hot-encoding"></a>

#### - One-Hot Encoding  
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

#### - Qu'est-ce que les Données de Validation ?  
   [Retour en haut](#table-des-matières)

<a id="validation-vs-test"></a>

#### - Validation vs Test  
   [Retour en haut](#table-des-matières)

<a id="normalization"></a>


<hr/> 
<hr/> 
<hr/> 

### 6. Normalisation en IA  
   [Retour en haut](#table-des-matières)

<a id="what-is-normalization"></a>

<hr/> 
<hr/> 
<hr/> 


#### - Qu'est-ce que la Normalisation ?  
   [Retour en haut](#table-des-matières)

<a id="normalization-importance"></a>

#### - Processus et Importance  
   [Retour en haut](#table-des-matières)

<a id="keras-sequential"></a>


<hr/> 
<hr/> 
<hr/> 

### 7. Modèle Séquentiel en Keras 


<hr/> 
<hr/> 
<hr/> 

   [Retour en haut](#table-des-matières)

<a id="sequential-model-creation"></a>

#### - Création et Compréhension  
   [Retour en haut](#table-des-matières)

<a id="neural-layers"></a>


<hr/> 
<hr/> 
<hr/> 

### 8. Couches dans un Réseau de Neurones  

<hr/> 
<hr/> 
<hr/> 

   [Retour en haut](#table-des-matières)

<a id="flatten-layer"></a>

#### - Couche Flatten  
   [Retour en haut](#table-des-matières)

<a id="output-layer"></a>

#### - Couche de Sortie  
   [Retour en haut](#table-des-matières)

<a id="dense-layer"></a>

#### - Couche Dense  
   [Retour en haut](#table-des-matières)

<a id="activation-functions"></a>


<hr/> 
<hr/> 
<hr/> 

### 9. Fonction d'Activation  

<hr/> 
<hr/> 
<hr/> 

   [Retour en haut](#table-des-matières)

<a id="activation-definition"></a>

#### - Qu'est-ce qu'une Fonction d'Activation ?  
   [Retour en haut](#table-des-matières)

<a id="relu-vs-softmax"></a>

#### - ReLU vs Softmax  
   [Retour en haut](#table-des-matières)

<a id="network-architecture"></a>


<hr/> 
<hr/> 
<hr/> 

### 10. Architecture de Réseau de Neurones  

<hr/> 
<hr/> 
<hr/> 

   [Retour en haut](#table-des-matières)

<a id="neuron-layer-choices"></a>

#### - Choix des Neurones et des Couches  
   [Retour en haut](#table-des-matières)

<a id="choosing-neurons"></a>

#### - Choisir le Nombre de Neurones  
   [Retour en haut](#table-des-matières)

