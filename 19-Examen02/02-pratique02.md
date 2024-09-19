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
## Étude de cas 1 - Détection d'anomalies dans un système de sécurité bancaire
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
# Étude de cas 2 - Analyse des zones de criminalité urbaine
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
