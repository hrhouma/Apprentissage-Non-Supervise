# Introduction aux Convolutional Neural Networks (CNNs)

## Table des matières
1. [Introduction](#introduction)
2. [Comprendre les images numériques](#comprendre-les-images-numériques)
   - 2.1 [Qu'est-ce qu'une image ?](#qu-est-ce-qu-une-image-)
   - 2.2 [Images en niveaux de gris](#images-en-niveaux-de-gris)
   - 2.3 [Images en couleur (RGB)](#images-en-couleur-rgb)
   - 2.4 [Résolution d'image](#résolution-d-image)
   - 2.5 [Dimensionnalité des images](#dimensionnalité-des-images)
   - 2.6 [Problèmes de dimensionnalité](#problèmes-de-dimensionnalité)
3. [Qu'est-ce qu'un réseau de neurones ?](#qu-est-ce-qu-un-réseau-de-neurones-)
4. [Introduction aux Convolutional Neural Networks (CNNs)](#introduction-aux-convolutional-neural-networks-cnns)
   - 4.1 [Définition et rôle des CNNs](#définition-et-rôle-des-cnns)
   - 4.2 [L'apprentissage supervisé](#l-apprentissage-supervisé)
5. [Fonctionnement détaillé des CNNs](#fonctionnement-détaillé-des-cnns)
   - 5.1 [Filtres (ou noyaux)](#filtres-ou-noyaux)
   - 5.2 [Opération de convolution](#opération-de-convolution)
   - 5.3 [Padding : pourquoi c'est important](#padding-pourquoi-c-est-important)
   - 5.4 [Pooling : réduction de la dimensionnalité](#pooling-réduction-de-la-dimensionnalité)
6. [Applications concrètes des CNNs](#applications-concrètes-des-cnns)
7. [Préparation aux concepts avancés : réseaux de neurones et autoencodeurs](#préparation-aux-concepts-avancés-réseaux-de-neurones-et-autoencodeurs)
8. [Conclusion](#conclusion)

## 1. Introduction <a name="introduction"></a>
Les réseaux de neurones convolutionnels (CNNs) sont une technologie révolutionnaire dans le domaine de l'intelligence artificielle et du traitement des images. Ce document vise à introduire les concepts fondamentaux liés aux CNNs, en les expliquant de manière détaillée et accessible aux débutants. Nous commencerons par comprendre ce qu'est une image numérique, avant d'explorer comment les CNNs peuvent analyser ces images pour réaliser des tâches complexes comme la reconnaissance d'objets.

## 2. Comprendre les images numériques <a name="comprendre-les-images-numériques"></a>

### 2.1 Qu'est-ce qu'une image ? <a name="qu-est-ce-qu-une-image-"></a>
Une image est une représentation visuelle du monde réel ou d'une idée, capturée ou générée numériquement. Dans le contexte des ordinateurs, une image est composée de nombreux petits points appelés **pixels**. Chacun de ces pixels contient des informations sur la couleur ou la luminosité.

### 2.2 Images en niveaux de gris <a name="images-en-niveaux-de-gris"></a>
Une **image en niveaux de gris** est une image où chaque pixel représente une intensité lumineuse, qui varie du noir (0) au blanc (255) pour une image de 8 bits. Cela signifie que l'image ne contient pas de couleur, mais uniquement des variations de gris, ce qui simplifie le traitement tout en conservant les informations essentielles sur les formes et les motifs.

**Exemple :** Imaginez une photo de nuit en noir et blanc. Chaque pixel de la photo représente à quel point une partie de l'image est claire ou sombre.

### 2.3 Images en couleur (RGB) <a name="images-en-couleur-rgb"></a>
Une **image en couleur** utilise souvent le modèle **RGB** (Rouge, Vert, Bleu). Chaque pixel est décrit par trois valeurs distinctes correspondant aux composantes de rouge, de vert et de bleu. En combinant ces trois couleurs à différentes intensités, on peut créer une large gamme de couleurs.

**Exemple :** Pensez à un écran de télévision. Il utilise de minuscules lumières rouges, vertes et bleues pour afficher toutes les couleurs que vous voyez à l'écran.

### 2.4 Résolution d'image <a name="résolution-d-image"></a>
La **résolution d'une image** fait référence au nombre de pixels que l'image contient. Plus il y a de pixels, plus l'image est détaillée. Par exemple, une image de résolution 1920x1080 (Full HD) contient 2 073 600 pixels. Une résolution plus élevée signifie que l'image peut montrer plus de détails, mais nécessite également plus d'espace de stockage et plus de puissance de calcul pour être traitée.

### 2.5 Dimensionnalité des images <a name="dimensionnalité-des-images"></a>
La **dimensionnalité** d'une image fait référence au nombre total de valeurs nécessaires pour représenter l'image. Pour une image en niveaux de gris de 28x28 pixels, la dimensionnalité est de 784 (28 * 28). Pour une image en couleur de la même taille, la dimensionnalité est de 2352 (28 * 28 * 3, en tenant compte des trois canaux de couleur).

**Pourquoi c'est important ?** Une dimensionnalité plus élevée signifie plus de complexité pour l'analyse et le traitement des images. Les ordinateurs doivent manipuler et analyser toutes ces valeurs pour comprendre ce que représente l'image, ce qui peut devenir très coûteux en termes de calcul.

### 2.6 Problèmes de dimensionnalité <a name="problèmes-de-dimensionnalité"></a>
À mesure que la dimensionnalité augmente, la quantité d'informations à traiter augmente également. Cela pose plusieurs défis :
- **Besoin de plus de données :** Plus la dimensionnalité est élevée, plus il faut de données pour entraîner efficacement un modèle, car chaque dimension supplémentaire augmente la complexité.
- **Surapprentissage (overfitting) :** Avec trop de dimensions, un modèle peut devenir trop spécialisé et mémoriser les détails des données d'entraînement au lieu d'apprendre des motifs généralisables. Cela rend le modèle moins performant sur de nouvelles données.

**Exemple :** Imaginez un modèle qui apprend à reconnaître des visages. Si on lui donne des images de très haute résolution, il pourrait apprendre des détails insignifiants (comme une petite ombre sur le visage d'une seule personne) au lieu d'apprendre des traits plus généraux comme la forme des yeux ou la position du nez.

## 3. Qu'est-ce qu'un réseau de neurones ? <a name="qu-est-ce-qu-un-réseau-de-neurones-"></a>
Un **réseau de neurones** est un modèle informatique inspiré du cerveau humain. Il est constitué de **neurones artificiels** qui sont organisés en couches. Chaque neurone reçoit des entrées, effectue un calcul (comme une somme pondérée) et produit une sortie. Ces réseaux sont capables d'apprendre à partir de données pour réaliser diverses tâches, comme la classification des images.

**Exemple :** Imaginez que vous essayez de reconnaître des lettres dans une image. Votre cerveau associe les formes aux lettres que vous connaissez. Un réseau de neurones fait quelque chose de similaire en associant des motifs dans les données aux catégories auxquelles ils appartiennent (par exemple, "A", "B", "C").

## 4. Introduction aux Convolutional Neural Networks (CNNs) <a name="introduction-aux-convolutional-neural-networks-cnns"></a>

### 4.1 Définition et rôle des CNNs <a name="définition-et-rôle-des-cnns"></a>
Un **réseau de neurones convolutionnel (CNN)** est un type de réseau de neurones conçu spécifiquement pour traiter les images. Contrairement aux réseaux de neurones traditionnels, les CNNs exploitent la structure en grille des images, ce qui leur permet de détecter des motifs visuels complexes tout en réduisant la dimensionnalité des données.

**Exemple :** Pensez à un filtre que vous appliqueriez sur une photo pour mettre en évidence les contours. Un CNN utilise des filtres similaires pour détecter automatiquement des motifs dans les images, comme les bords ou les textures.

### 4.2 L'apprentissage supervisé <a name="l-apprentissage-supervisé"></a>
L'apprentissage supervisé est une méthode où le modèle est entraîné sur un ensemble de données étiquetées. Cela signifie que pour chaque image d'entraînement, le modèle sait quelle est la bonne réponse. Par exemple, un CNN pourrait être entraîné à reconnaître des chats en lui montrant des milliers d'images de chats avec l'étiquette "chat".

## 5. Fonctionnement détaillé des CNNs <a name="fonctionnement-détaillé-des-cnns"></a>

### 5.1 Filtres (ou noyaux) <a name="filtres-ou-noyaux"></a> (suite)
Les **filtres**, ou **noyaux**, sont des petites matrices (par exemple 3x3 ou 5x5) appliquées sur l'image d'entrée pour extraire des caractéristiques spécifiques telles que les contours, les textures, ou d'autres motifs importants dans l'image. Ces filtres sont glissés sur l'ensemble de l'image, et à chaque position, un produit entre les valeurs du filtre et les pixels correspondants de l'image est calculé. Le résultat est une nouvelle image appelée **carte de caractéristiques** (ou *feature map*), qui met en évidence les motifs détectés.

**Exemple :** Imaginez un filtre de détection de bords appliqué à une image. Ce filtre va détecter les zones où il y a un changement abrupt de luminosité, ce qui correspond souvent aux bords des objets dans l'image. Ainsi, la carte de caractéristiques résultante mettra en évidence les contours des objets dans l'image.

### 5.2 Opération de convolution <a name="opération-de-convolution"></a>
L'opération de convolution est le processus de glissement du filtre sur l'image. À chaque position, les valeurs du filtre sont multipliées par les valeurs des pixels correspondants, puis additionnées pour donner une valeur unique dans la carte de caractéristiques. Cette opération permet de transformer l'image d'entrée en une nouvelle représentation qui met en avant les caractéristiques les plus importantes.

**Pourquoi c'est important ?** La convolution permet au CNN de capturer des motifs visuels qui sont cruciaux pour la tâche de reconnaissance. Par exemple, dans une tâche de reconnaissance faciale, les premières couches du CNN pourraient détecter des bords et des textures, tandis que les couches plus profondes pourraient détecter des parties du visage comme les yeux ou la bouche.

### 5.3 Padding : pourquoi c'est important <a name="padding-pourquoi-c-est-important"></a>
Le **padding** est une technique utilisée pour ajouter des pixels artificiels autour des bords de l'image avant d'appliquer le filtre. Cela permet de préserver la taille de l'image d'origine ou de contrôler la taille de la carte de caractéristiques. Sans padding, l'application répétée de filtres réduirait progressivement la taille de l'image, ce qui pourrait entraîner une perte d'information importante, notamment sur les bords de l'image.

**Exemple :** Si vous appliquez un filtre 3x3 sur une image 5x5 sans padding, la carte de caractéristiques résultante sera de taille 3x3. En ajoutant du padding, vous pouvez maintenir la taille de l'image à 5x5, assurant ainsi que les informations des bords de l'image ne sont pas perdues.

### 5.4 Pooling : réduction de la dimensionnalité <a name="pooling-réduction-de-la-dimensionnalité"></a>
Le **pooling** est une opération qui consiste à réduire la dimensionnalité des cartes de caractéristiques tout en conservant les informations essentielles. L'opération de pooling la plus courante est le **max pooling**, où l'on divise la carte de caractéristiques en petites régions (par exemple 2x2) et où l'on remplace chaque région par la valeur maximale de cette région.

**Pourquoi c'est important ?** Le pooling permet de réduire le nombre de paramètres dans le réseau, ce qui diminue la complexité computationnelle et aide à prévenir le surapprentissage. Il permet également de rendre le modèle plus robuste aux variations dans l'image, comme des changements mineurs de position des objets.

**Exemple :** Si vous avez une carte de caractéristiques 4x4 et que vous appliquez un max pooling 2x2, la carte résultante sera de taille 2x2. Cela réduit la dimensionnalité tout en conservant les informations les plus importantes (les valeurs maximales).

## 6. Applications concrètes des CNNs <a name="applications-concrètes-des-cnns"></a>
Les CNNs sont utilisés dans une multitude d'applications pratiques, notamment :
- **Reconnaissance d'images** : Identifier des objets, des animaux, des personnes, etc., dans des images.
- **Analyse médicale** : Détection de maladies dans des images médicales (comme des radiographies).
- **Vision par ordinateur pour véhicules autonomes** : Reconnaissance des obstacles, des panneaux de signalisation, et des piétons.
- **Sécurité et surveillance** : Identification d'intrus ou de comportements suspects dans des vidéos.

**Exemple concret :** Les systèmes de sécurité utilisent souvent des CNNs pour détecter des intrus en temps réel via des caméras de surveillance. Les images des caméras sont analysées en continu par le CNN, qui peut déclencher une alerte si un intrus est détecté.

## 7. Préparation aux concepts avancés : réseaux de neurones et autoencodeurs <a name="préparation-aux-concepts-avancés-réseaux-de-neurones-et-autoencodeurs"></a>
Avant de plonger dans les concepts plus avancés comme les réseaux de neurones traditionnels et les autoencodeurs, il est essentiel de comprendre comment les CNNs traitent la dimensionnalité élevée des images. Les autoencodeurs, en particulier, sont utilisés pour apprendre une représentation plus compacte des données, ce qui est crucial pour réduire la dimensionnalité et éliminer le bruit dans les images.

**Transition vers les réseaux de neurones et les autoencodeurs :** Une fois que vous comprenez comment les CNNs extraient des caractéristiques importantes tout en réduisant la dimensionnalité, vous serez mieux préparés à aborder les concepts des autoencodeurs, qui sont utilisés pour la compression des données et la détection d'anomalies, et des réseaux de neurones plus traditionnels, qui se concentrent sur d'autres types de données et de tâches.

## 8. Conclusion <a name="conclusion"></a>
Les Convolutional Neural Networks sont un pilier du traitement moderne des images et de l'intelligence artificielle. Leur capacité à gérer efficacement la dimensionnalité élevée des images et à détecter des motifs complexes en fait des outils puissants pour une large gamme d'applications. Ce document a pour objectif de fournir une base solide pour comprendre ces réseaux et préparer les étudiants à aborder des concepts plus avancés, comme les réseaux de neurones traditionnels et les autoencodeurs.
