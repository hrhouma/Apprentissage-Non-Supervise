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

