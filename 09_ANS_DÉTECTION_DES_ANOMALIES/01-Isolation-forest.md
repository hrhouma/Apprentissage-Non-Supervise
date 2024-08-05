# Introduction
- Isolation Forest est un algorithme d'apprentissage automatique non supervisé pour la détection d'anomalies. 
- Comme son nom l'indique, Isolation Forest est une méthode d'ensemble (similaire à la forêt aléatoire).
- En d'autres termes, il utilise la moyenne des prédictions de plusieurs arbres de décision pour attribuer le score d'anomalie final à un point de données donné.
- Contrairement à d'autres algorithmes de détection d'anomalies, qui définissent d'abord ce qui est "normal" et rapportent ensuite tout le reste comme étant anormal, Isolation Forest tente d'isoler les points de données anormaux dès le départ.

# Algorithme

- Supposons que nous ayons les points de données suivants :

![image](https://github.com/user-attachments/assets/6c10502f-d7d4-4283-b183-84502d3be1f0)


- L'algorithme Isolation Forest sélectionne une dimension aléatoire (dans ce cas, la dimension associée à l'axe des x) et divise aléatoirement les données le long de cette dimension.


![image](https://github.com/user-attachments/assets/b178ed3d-e532-411b-a223-437fcaf4baa9)

- Les deux sous-espaces résultants définissent leur propre sous-arbre.


- Dans cet exemple, la coupure sépare un point isolé du reste du jeu de données.
- Le premier niveau de l'arbre binaire résultant se compose de deux nœuds, l'un contenant le sous-arbre des points à gauche de la coupure initiale et l'autre représentant le point unique à droite.


![image](https://github.com/user-attachments/assets/02d1715e-90fd-4dea-9e84-140d0df21dfb)

Il est important de noter que les autres arbres de l'ensemble sélectionneront des coupures initiales différentes. Dans l'exemple suivant, la première coupure n'isole pas l'anomalie.

![image](https://github.com/user-attachments/assets/70b0b335-0cd6-48c8-8cfd-3082d1c6ff31)

Nous obtenons un arbre composé de deux nœuds, l'un contenant les points à gauche de la ligne et l'autre représentant les points à droite de la ligne. Le processus est répété jusqu'à ce que chaque feuille de l'arbre représente un point de données unique du jeu de données. Dans notre exemple, la deuxième itération parvient à isoler l'anomalie. Après cette étape, l'arbre ressemblerait à ceci :



![image](https://github.com/user-attachments/assets/02bfb42c-f95f-4672-a29d-0507b9bac644)











Rappelez-vous qu'une coupure peut également se produire le long de l'autre dimension, comme c'est le cas pour ce troisième arbre de décision.

![image](https://github.com/user-attachments/assets/50be8390-30be-4f81-a18f-0080c774fe19)
![image](https://github.com/user-attachments/assets/c8ebf98a-46d6-492f-b448-a57c2100d28c)
![image](https://github.com/user-attachments/assets/95716ccc-9ee7-46e6-a58e-992a92af97c5)
![image](https://github.com/user-attachments/assets/643121d0-2c9f-4943-851d-fef31f260607)


En moyenne, un point de données anormal sera isolé dans une boîte englobante à une profondeur d'arbre plus petite que les autres points. Lors de l'inférence avec un modèle Isolation Forest entraîné, le score d'anomalie final est reporté comme la moyenne des scores reportés par chaque arbre de décision individuel.


# Variables catégorielles
Si vous êtes comme moi, vous vous demandez probablement comment cela fonctionnerait avec des variables catégorielles. En supposant qu'une valeur moins observée soit anormale, l'algorithme Isolation Forest peut utiliser des variables catégorielles en les représentant comme des rectangles où la taille du rectangle est proportionnelle à la fréquence d'occurrence.


![image](https://github.com/user-attachments/assets/e39f4045-3052-40b3-8579-861160aac57d)

Nous considérons l'ensemble des valeurs possibles entre le milieu de la première valeur et le milieu de la dernière valeur. Nous sélectionnons un point aléatoire le long du domaine, puis déterminons le bord le plus proche d'un rectangle donné. Ceci est utilisé pour notre coupure.

![image](https://github.com/user-attachments/assets/ada3c218-58b1-498d-9b00-66bcb546be20)

Pour garantir l'équité, les autres arbres de la forêt utiliseront un ordre différent.

![image](https://github.com/user-attachments/assets/836751cf-2133-4e39-b1df-3407dae178aa)

Python
Pour commencer, importez les bibliothèques suivantes :

```python
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import pandas as pd
```

Dans le tutoriel suivant, nous allons travailler avec le jeu de données sur le cancer du sein de l'UCI machine learning repository. Heureusement, la bibliothèque scikit-learn fournit une fonction wrapper pour télécharger les données.

```python
breast_cancer = load_breast_cancer()
df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
df["benign"] = breast_cancer.target
```

Comme nous pouvons le voir, le jeu de données contient 30 caractéristiques numériques et une valeur cible de 0 et 1 pour les tumeurs bénignes et malignes, respectivement.

```python
df.head()
```

Pour notre cas d'utilisation, nous allons supposer qu'une étiquette maligne est anormale. Le jeu de données contient un nombre relativement élevé de tumeurs malignes. Ainsi, nous utilisons un échantillonnage aléatoire.

```python
majority_df = df[df["benign"] == 1]
minority_df = df[df["benign"] == 0]
minority_downsampled_df = resample(minority_df, replace=True, n_samples=30, random_state=42)
downsampled_df = pd.concat([majority_df, minority_downsampled_df])
```

Après l'échantillonnage, il y a plus de 10 fois plus d'échantillons de la classe majoritaire que de la classe minoritaire.

```python
downsampled_df["benign"].value_counts()
1    357
0     30
Name: benign, dtype: int64
```

Nous sauvegardons les caractéristiques et la cible comme variables séparées.

```python
y = downsampled_df["benign"]
X = downsampled_df.drop("benign", axis=1)
```

Nous mettons de côté une partie des données totales pour les tests.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

Ensuite, nous créons une instance de la classe IsolationForest.

```python
model = IsolationForest(random_state=42)
```

Nous entraînons le modèle.

```python
model.fit(X_train, y_train)
```

Nous prédisons les données dans l'ensemble de test.

```python
y_pred = model.predict(X_test)
```

IsolationForest attribue une valeur de -1 au lieu de 0. Par conséquent, nous la remplaçons pour nous assurer que nous n'avons que 2 valeurs distinctes dans notre matrice de confusion.

```python
y_pred[y_pred == -1] = 0
```

Comme nous pouvons le voir, l'algorithme fait un bon travail de prédiction des points de données anormaux.

```python
confusion_matrix(y_test, y_pred)
```

```plaintext
array([[ 7,  2],
       [ 5, 83]])
```


# Annexe : 

### Isolation Forest : Une Explication Vulgarisée avec des Exemples de la Vie Professionnelle

#### Qu'est-ce qu'Isolation Forest ?
Isolation Forest est un algorithme conçu pour détecter les anomalies dans un ensemble de données. Les anomalies sont des points de données qui diffèrent de la majorité, comme un employé portant une chemise rouge dans un bureau où tout le monde porte des chemises blanches.

#### Comment fonctionne Isolation Forest ?

##### Analogie de l'Entreprise

Imaginez que vous gérez une entreprise où la plupart des employés arrivent au bureau entre 8h et 9h. Un jour, vous remarquez qu'un employé arrive à 6h du matin. Cet employé est une anomalie. Isolation Forest aide à identifier ces comportements anormaux.

##### Étape 1 : Sélectionner une Dimension
L'algorithme commence par choisir une dimension aléatoire. C'est comme décider de regarder les heures d'arrivée des employés pour identifier les comportements anormaux.

![Sélection d'une dimension](file-J8AhTSpWdC3HC7sgY3OULmlA)

##### Étape 2 : Diviser les Données
Ensuite, l'algorithme divise les données selon cette dimension. Imaginez que vous séparez les employés en deux groupes : ceux qui arrivent avant 8h et ceux qui arrivent après 8h.

![Diviser les données](file-q0Ebvm11xbA2YAgVlgeg2Vek)

##### Étape 3 : Créer des Sous-Arbres
Chaque groupe devient alors un sous-arbre. C'est comme organiser les employés dans différentes équipes en fonction de leurs heures d'arrivée.

![Créer des sous-arbres](file-ZW5hpnRITsfGZajS9rnBSYTL)

##### Étape 4 : Répéter le Processus
Le processus de division continue jusqu'à ce que chaque employé soit dans son propre groupe. C'est comme continuer à diviser les équipes jusqu'à ce que chaque employé soit seul.

![Répéter le processus](file-3Qifo63z1fGniryLRShtiESJ)

##### Exemple Pratique : Identifier l'Anomalie
Imaginez que vous avez un groupe d'employés (points bleus) et un employé qui arrive à 6h du matin (point rouge). Au début, l'algorithme peut ne pas isoler cet employé immédiatement. Mais après quelques itérations, il réussit à le faire.

![Exemple pratique](file-2ggQUFzBT1vSEl0pCHop8fNO)

#### Pourquoi est-ce Efficace ?
Dans Isolation Forest, un point de données anormal (comme l'employé qui arrive très tôt) sera isolé plus rapidement avec moins de divisions qu'un point normal (les employés qui arrivent entre 8h et 9h). C'est comme dire qu'il est plus facile de trouver l'employé qui arrive à 6h parmi ceux qui arrivent à 8h parce qu'il se démarque immédiatement.

##### Comparaison avec les Horaires de Travail
Supposons que dans une entreprise, la majorité des employés travaillent de 9h à 17h. Si un employé commence à travailler à 3h du matin, il sera plus facile à repérer. De la même manière, Isolation Forest isole rapidement les comportements rares ou inhabituels.

#### Variables Catégorielles
Isolation Forest peut aussi traiter des variables catégorielles. Imaginez des rectangles de différentes tailles représentant différentes équipes dans l'entreprise. La taille de chaque rectangle est proportionnelle au nombre d'employés dans chaque équipe. L'algorithme utilise ces rectangles pour diviser les données de manière équitable.

![Variables catégorielles](file-4Zuttub2b3kDBzKKMyaIRLR1)

### Implémentation en Python : Un Exemple Concret

Voyons comment utiliser Isolation Forest avec un exemple en Python.

#### Étape 1 : Importer les Bibliothèques
```python
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import pandas as pd
```

#### Étape 2 : Charger le Jeu de Données
Nous utiliserons le jeu de données sur le cancer du sein de l'UCI.
```python
breast_cancer = load_breast_cancer()
df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
df["benign"] = breast_cancer.target
```

#### Étape 3 : Préparer les Données
Nous allons supposer qu'une tumeur maligne (étiquetée 0) est une anomalie.
```python
majority_df = df[df["benign"] == 1]
minority_df = df[df["benign"] == 0]
minority_downsampled_df = resample(minority_df, replace=True, n_samples=30, random_state=42)
downsampled_df = pd.concat([majority_df, minority_downsampled_df])
```

#### Étape 4 : Séparer les Caractéristiques et la Cible
```python
y = downsampled_df["benign"]
X = downsampled_df.drop("benign", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

#### Étape 5 : Créer et Entraîner le Modèle
```python
model = IsolationForest(random_state=42)
model.fit(X_train, y_train)
```

#### Étape 6 : Prédire et Évaluer
```python
y_pred = model.predict(X_test)
y_pred[y_pred == -1] = 0
print(confusion_matrix(y_test, y_pred))
```

#### Interprétation des Résultats
La matrice de confusion nous montre comment l'algorithme a fait pour prédire les anomalies (tumeurs malignes) par rapport aux données normales (tumeurs bénignes).

### Comparaison avec d'Autres Algorithmes

#### Détection d'Anomalies avec des Règles Fixes
Certains algorithmes définissent d'abord ce qui est normal et tout le reste est considéré comme anormal. C'est comme dire que tous les employés qui ne travaillent pas de 9h à 17h sont des anomalies. Cette méthode peut être moins efficace car elle ne tient pas compte de la variété des anomalies possibles.

#### Comparaison avec la Détection de Fraude
Pensez à la détection de fraude par carte de crédit. Isolation Forest peut détecter des transactions inhabituelles (comme des achats très élevés) sans avoir besoin de définir ce qu'est une transaction normale au préalable. C'est comme trouver des transactions frauduleuses parmi des transactions régulières.

### Conclusion
Isolation Forest est un outil puissant et flexible pour détecter les anomalies dans divers types de données. Grâce à ses arbres de décision aléatoires, il peut isoler rapidement et efficacement les points de données inhabituels, même dans des jeux de données complexes. Avec des exemples de la vie professionnelle et des comparaisons pratiques, nous pouvons mieux comprendre comment cet algorithme fonctionne et pourquoi il est si utile.
