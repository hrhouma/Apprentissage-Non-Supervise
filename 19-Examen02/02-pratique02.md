# **Première Partie : Quiz (10 questions)**

**Choisissez la réponse la plus appropriée et la plus correcte. Il n’y a qu’une seule bonne réponse.**

1. **Qu'est-ce que l'apprentissage non supervisé ?**  
   a) Apprentissage avec des données étiquetées  
   b) Apprentissage avec des données non étiquetées  
   c) Apprentissage avec supervision humaine  
   d) Apprentissage uniquement basé sur des algorithmes de régression

2. **Quelle méthode est utilisée pour détecter les anomalies dans un ensemble de données non supervisées ?**  
   a) Régression linéaire  
   b) Isolation Forest  
   c) Réseau de neurones supervisé  
   d) Apprentissage par renforcement

3. **Le clustering K-means appartient à quel type d'apprentissage ?**  
   a) Supervisé  
   b) Non supervisé  
   c) Semi-supervisé  
   d) Apprentissage par renforcement

4. **L'algorithme DBSCAN est particulièrement adapté pour :**  
   a) Les données non étiquetées et le bruit  
   b) Les données étiquetées uniquement  
   c) Les réseaux de neurones  
   d) La réduction de dimensionnalité

5. **Quel est l'objectif principal de l'apprentissage non supervisé ?**  
   a) Prédire une étiquette  
   b) Découvrir des structures cachées dans les données  
   c) Surveiller les performances d'un modèle  
   d) Tester un modèle d'apprentissage supervisé

6. **Dans un clustering, que signifie "centroïde" ?**  
   a) Un point aléatoire dans l'espace de données  
   b) Le centre d'un groupe de points  
   c) Un point de données bruyant  
   d) Un point éloigné de tous les autres points

7. **Un autoencodeur est utilisé pour :**  
   a) Classer des données  
   b) Réduire la dimensionnalité des données  
   c) Augmenter la taille des données  
   d) Réparer des données manquantes

8. **Quelle technique est utilisée pour projeter les données dans un espace de dimension réduite ?**  
   a) DBSCAN  
   b) PCA (Analyse en Composantes Principales)  
   c) K-means  
   d) Réseau de neurones convolutif

9. **Le clustering hiérarchique produit :**  
   a) Des clusters prédéfinis  
   b) Une hiérarchie de clusters  
   c) Des clusters basés sur des graphes  
   d) Des clusters basés sur des réseaux de neurones

10. **Quels algorithmes sont typiquement utilisés pour la détection d'anomalies ?**  
   a) K-means et PCA  
   b) Isolation Forest et DBSCAN  
   c) Réseaux convolutifs et régressions logistiques  
   d) K-means et régressions linéaires

# **Deuxième Partie : Questions de développement (2 questions)**

1. **Expliquez brièvement le fonctionnement de l'algorithme DBSCAN. En quoi se différencie-t-il de K-means pour la détection de clusters ?**

2. **Quelle est l'importance de la réduction de dimensionnalité en apprentissage non supervisé ? Donnez deux exemples de technique couramment utilisée et comparez les.**

# **Troisième Partie : Exercice pratique**

**Consigne :** Vous travaillez avec un ensemble de données non étiquetées contenant des transactions financières. 
Votre tâche est de détecter les anomalies dans cet ensemble de données à l'aide de l'algorithme Isolation Forest.

1. Chargez l'ensemble de données (un fichier CSV vous est fourni).  
2. Utilisez l'algorithme `Isolation Forest` pour identifier les transactions suspectes.  
3. Affichez les 10 transactions ayant le plus grand score d'anomalie.

**Indications :**  
- Utilisez la bibliothèque `scikit-learn`.  
- Vous pouvez normaliser les données avant d'appliquer l'algorithme.

------------------------------------------------
# **Études de cas 
------------------------------------------------

------------------------------------------------
## Étude de cas 1 : Détection d'anomalies dans un système de sécurité bancaire
------------------------------------------------

Une grande banque internationale fait face à un défi de taille : détecter efficacement les transactions frauduleuses parmi des millions d'opérations quotidiennes. L'équipe de sécurité dispose d'un vaste ensemble de données comprenant des transactions légitimes et quelques cas connus de fraude. Cependant, les fraudeurs évoluent constamment, rendant difficile la définition précise de ce qu'est une transaction suspecte.

**Contexte :**
- Les données sont non étiquetées (on ne sait pas a priori quelles transactions sont frauduleuses).
- Les cas de fraude sont rares par rapport au volume total de transactions.
- Les patterns de fraude changent fréquemment, rendant les approches supervisées moins efficaces.

**Défis :**
1. Identifier les transactions anormales sans avoir une définition claire de ce qu'est une "anomalie".
2. Gérer un grand volume de données avec des ressources de calcul limitées.
3. S'adapter à l'évolution des comportements frauduleux sans nécessiter de réentraînement constant.

**Solution mise en place :**
L'équipe de data science a implémenté un algorithme qui :

- Construit un modèle de normalité basé sur la majorité des données.
- Attribue un score d'anomalie à chaque transaction.
- Fonctionne de manière non supervisée, sans nécessiter d'étiquettes.
- Est capable de détecter des anomalies dans des espaces de grande dimension.
- Utilise une structure d'arbre pour une exécution rapide, même sur de grands ensembles de données.

**Résultats :**
- L'algorithme a permis de détecter 95% des fraudes connues.
- Il a également identifié de nouveaux patterns de fraude non détectés auparavant.
- Le temps de traitement a été réduit de 70% par rapport aux méthodes précédentes.
- Le taux de faux positifs a diminué de 60%, réduisant la charge de travail des analystes.

**Question  :**
Parmi les algorithmes suivants, lequel correspond le mieux à la solution décrite dans cette étude de cas ?

A) KMeans

B) DBSCAN

C) Clustering hiérarchique

D) Isolation Forest

E) PCA

F) Autoencodeurs

------------------------------------------------
# Étude de cas 2 : Analyse des zones de criminalité urbaine
------------------------------------------------

Le département de police d'une grande métropole cherche à améliorer son efficacité en identifiant les zones à forte criminalité pour mieux allouer ses ressources. Ils disposent de données géolocalisées sur les incidents criminels survenus au cours des dernières années.

**Contexte :**
- Les données comprennent les coordonnées GPS de chaque incident criminel.
- La densité des incidents varie considérablement selon les quartiers.
- Certaines zones présentent des formes irrégulières de concentration d'incidents.

**Défis :**
1. Identifier des clusters de criminalité sans connaître à l'avance leur nombre ou leur taille.
2. Détecter des zones de forme irrégulière, pas nécessairement circulaires.
3. Distinguer les zones à forte densité d'incidents des zones à faible densité.
4. Gérer efficacement les données bruitées (incidents isolés).

**Solution mise en place :**
L'équipe de data science a implémenté un algorithme qui :

- Groupe les incidents en fonction de leur proximité géographique.
- Ne nécessite pas de spécifier à l'avance le nombre de clusters.
- Peut identifier des clusters de forme arbitraire.
- Est capable de distinguer les zones denses des zones moins denses.
- Identifie automatiquement les points isolés comme du bruit.

**Résultats :**
- L'algorithme a identifié 17 zones distinctes de forte criminalité dans la ville.
- Il a révélé des formes de clusters non circulaires, suivant par exemple des axes routiers.
- 5% des incidents ont été classés comme bruit, correspondant à des événements isolés.
- La visualisation des résultats a permis une meilleure compréhension de la répartition spatiale de la criminalité.

**Question :**
Parmi les algorithmes suivants, lequel correspond le mieux à la solution décrite dans cette étude de cas ?

A) KMeans

B) DBSCAN

C) Clustering hiérarchique

D) Isolation Forest

E) PCA

F) Autoencodeurs



------------------------------------------------
## Étude de cas 3 : Optimisation de la qualité d'image dans l'industrie pharmaceutique
------------------------------------------------


Une entreprise pharmaceutique utilise des systèmes de vision par ordinateur pour inspecter la qualité des comprimés sur ses lignes de production. Le système capture des images haute résolution de chaque comprimé, générant un volume important de données.

**Contexte :**
- Chaque image est représentée par des milliers de pixels, chacun étant une variable.
- Le traitement en temps réel est crucial pour maintenir la cadence de production.
- Les ressources de calcul sont limitées sur les lignes de production.

**Défis :**
1. Réduire la dimensionnalité des données d'image tout en préservant les informations essentielles sur la qualité des comprimés.
2. Accélérer le traitement des images pour une analyse en temps réel.
3. Identifier les caractéristiques les plus importantes pour la détection des défauts.
4. Créer une représentation compacte des images pour le stockage à long terme.

**Solution mise en place :**
L'équipe de data science a implémenté un algorithme qui :

- Réduit la dimensionnalité des données d'image de manière linéaire.
- Identifie les directions de variance maximale dans les données.
- Permet de reconstruire les données originales avec une perte minimale d'information.
- Fournit des composantes orthogonales, facilitant l'interprétation des résultats.
- Est capable de traiter de grands ensembles de données efficacement.

**Résultats :**
- La dimensionnalité des images a été réduite de 95%, passant de 10 000 à 500 variables.
- Le temps de traitement par image a diminué de 80%.
- Les 10 premières composantes expliquent 85% de la variance totale des données.
- L'analyse des composantes principales a révélé des patterns de défauts auparavant non détectés.

**Question :**
Parmi les algorithmes suivants, lequel correspond le mieux à la solution décrite dans cette étude de cas ?

A) KMeans

B) DBSCAN

C) Clustering hiérarchique

D) Isolation Forest

E) PCA

F) Autoencodeurs






------------------------------------------------
## Étude de cas 4 : Segmentation des clients d'un supermarché
------------------------------------------------

Un grand supermarché cherche à optimiser ses stratégies marketing en segmentant sa clientèle. L'objectif est de créer des groupes de clients distincts pour personnaliser les offres et améliorer la satisfaction client.

**Contexte :**
- Le supermarché dispose de données sur les habitudes d'achat de 100 000 clients.
- Les données incluent la fréquence des achats, le montant moyen dépensé, et les catégories de produits achetés.
- L'entreprise souhaite créer un nombre fixe et prédéfini de segments clients.

**Défis :**
1. Créer des groupes de clients homogènes basés sur leurs comportements d'achat.
2. Traiter un grand volume de données de manière efficace.
3. Obtenir des centres de clusters facilement interprétables pour chaque segment.
4. Permettre une assignation rapide de nouveaux clients à un segment existant.

**Solution mise en place :**
L'équipe de data science a implémenté un algorithme qui :

- Divise les clients en un nombre prédéfini de clusters (5 segments).
- Minimise la variance intra-cluster tout en maximisant la variance inter-clusters.
- Fournit un centre de cluster (centroïde) représentatif pour chaque segment.
- Permet une assignation rapide de nouveaux clients basée sur la distance au centroïde le plus proche.
- Est capable de traiter efficacement de grands ensembles de données.

**Résultats :**
- 5 segments de clients distincts ont été identifiés : "Acheteurs fréquents à forte valeur", "Familles", "Jeunes professionnels", "Acheteurs occasionnels", et "Seniors économes".
- Chaque segment a un profil clair basé sur son centroïde, facilitant l'interprétation et la création de stratégies marketing ciblées.
- Le temps de traitement pour segmenter 100 000 clients était de seulement 2 minutes.
- L'assignation de nouveaux clients à un segment se fait en temps réel, permettant une personnalisation immédiate.

**Question pour les étudiants :**
Parmi les algorithmes suivants, lequel correspond le mieux à la solution décrite dans cette étude de cas ?

A) KMeans

B) DBSCAN

C) Clustering hiérarchique

D) Isolation Forest

E) PCA

F) Autoencodeurs



------------------------------------------------
## Étude de cas 5 : Détection de défauts dans la production de semi-conducteurs
------------------------------------------------
Une entreprise de fabrication de semi-conducteurs cherche à améliorer son contrôle qualité en détectant automatiquement les défauts de production. Les wafers de silicium sont photographiés à haute résolution à différentes étapes de la production.

**Contexte :**
- Des millions d'images de wafers sont générées chaque jour.
- Les défauts sont rares mais coûteux s'ils ne sont pas détectés à temps.
- Les images sont de très haute dimension (plusieurs millions de pixels chacune).
- Les types de défauts peuvent évoluer avec le temps en raison de changements dans le processus de fabrication.

**Défis :**
1. Réduire la dimensionnalité des images tout en préservant les informations cruciales sur les défauts.
2. Détecter des anomalies sans avoir d'exemples étiquetés de tous les types de défauts possibles.
3. S'adapter à de nouveaux types de défauts sans nécessiter un réentraînement complet du modèle.
4. Traiter un grand volume de données en temps réel.

**Solution mise en place :**
L'équipe de data science a implémenté un algorithme qui :

- Apprend une représentation compressée des images de wafers normaux.
- Reconstruit les images à partir de cette représentation compressée.
- Détecte les anomalies en comparant l'image originale à sa reconstruction.
- S'entraîne de manière non supervisée sur des images de wafers normaux.
- Peut s'adapter à de nouvelles variations en continuant son apprentissage.

**Résultats :**
- La dimensionnalité des images a été réduite de 99%, passant de 5 millions à 50 000 caractéristiques.
- Le taux de détection des défauts connus a atteint 98%.
- De nouveaux types de défauts, non vus pendant l'entraînement, ont été détectés avec succès.
- Le temps de traitement par image a été réduit à 100 ms, permettant une inspection en temps réel.

**Question pour les étudiants :**
Parmi les algorithmes suivants, lequel correspond le mieux à la solution décrite dans cette étude de cas ?

A) KMeans

B) DBSCAN

C) Clustering hiérarchique

D) Isolation Forest

E) PCA

F) Autoencodeurs




------------------------------------------------
## Étude de cas 6 : Analyse phylogénétique des espèces de primates
------------------------------------------------

Une équipe de biologistes étudie l'évolution des primates et cherche à construire un arbre phylogénétique basé sur des données génétiques. Ils ont collecté des séquences d'ADN de 50 espèces différentes de primates et veulent comprendre leurs relations évolutives.

**Contexte :**
- Les données consistent en des séquences d'ADN de longueur variable pour chaque espèce.
- Le nombre d'espèces est relativement petit (50), mais chaque séquence contient des milliers de paires de bases.
- Les relations entre les espèces sont supposées avoir une structure hiérarchique naturelle.
- L'objectif est de créer un arbre évolutif montrant les relations entre toutes les espèces.

**Défis :**
1. Construire une hiérarchie complète des relations entre les espèces.
2. Permettre une visualisation intuitive de l'arbre évolutif.
3. Ne pas avoir besoin de spécifier à l'avance le nombre de groupes.
4. Capturer les relations à différents niveaux de granularité.

**Solution mise en place :**
L'équipe de bio-informatique a implémenté un algorithme qui :

- Commence par considérer chaque espèce comme un cluster distinct.
- Fusionne progressivement les clusters les plus similaires.
- Utilise une mesure de distance basée sur la similarité des séquences d'ADN.
- Construit un dendrogramme montrant l'histoire complète des fusions.
- Permet de couper l'arbre à différents niveaux pour obtenir des groupements d'espèces.

**Résultats :**
- Un arbre phylogénétique complet a été construit, montrant les relations entre toutes les 50 espèces.
- L'arbre révèle clairement les sous-groupes majeurs de primates (ex: grands singes, singes du nouveau monde, lémuriens).
- Les biologistes peuvent explorer les relations à différents niveaux de détail en "coupant" l'arbre à différentes hauteurs.
- L'analyse a confirmé certaines hypothèses existantes sur l'évolution des primates et a révélé de nouvelles relations inattendues.

**Question pour les étudiants :**
Parmi les algorithmes suivants, lequel correspond le mieux à la solution décrite dans cette étude de cas ?

A) KMeans

B) DBSCAN

C) Clustering hiérarchique agglomératif

D) Isolation Forest

E) PCA

F) Autoencodeurs




