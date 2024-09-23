# **Deuxième Partie : 2 questions (40 points)**

---

## **Partie 2.1 - Question de développement (10 points)**

- Comparez l'utilisation de l'Analyse en Composantes Principales (ACP) et des Autoencodeurs pour la réduction de dimensionnalité dans un contexte de votre choix (par exemple, l'analyse d'images médicales à haute résolution).

---

## **Partie 2.2 - Études de cas (30 points)**

Chaque étude de cas compte **6 points**.

### **Question générale pour toutes les études de cas :**

Pour chacune des 5 études de cas présentées, identifiez l'algorithme de machine learning qui correspond le mieux à la solution décrite.  
*Choisissez parmi les options suivantes :*

- A) KMeans  
- B) DBSCAN  
- C) PCA (Analyse en Composantes Principales)  
- D) Isolation Forest  
- E) Clustering hiérarchique agglomératif  
- F) Autoencodeurs  
- G) Régression logistique  
- H) Arbre de décision  
- I) Support Vector Machines (SVM)

*Justifiez brièvement votre choix pour chaque étude de cas en expliquant pourquoi l'algorithme sélectionné est le plus approprié dans le contexte donné.*





----------------------------------------------------------------------------------------
### **Étude de cas 1 : Réduction de dimensionnalité pour l'analyse d'images médicales (6 points)**
----------------------------------------------------------------------------------------

Un hôpital universitaire souhaite analyser un grand ensemble d'images médicales provenant de scanners IRM afin d'identifier les anomalies cérébrales. Chaque image est composée de millions de pixels, rendant le traitement des données coûteux en temps et en ressources.

**Contexte :**
- Chaque image contient plusieurs millions de pixels, ce qui représente des données de très haute dimension.
- Le traitement des données est limité par les ressources de calcul de l'hôpital.
- Il est crucial de conserver les informations les plus importantes pour détecter les anomalies, tout en simplifiant la représentation des images.
- Les relations entre les pixels semblent suivre des schémas réguliers et peuvent être approximées par des combinaisons linéaires de variables, ce qui suggère l'existence de directions principales expliquant la majeure partie de la variance dans les données.
- Les résultats doivent être visualisables et interprétables par les radiologues.

**Défis :**
1. Réduire la dimensionnalité des images tout en capturant les directions où la variance des données est maximale, afin de préserver l'essentiel des informations critiques.
2. Accélérer le traitement des images pour permettre une analyse rapide sans sacrifier la précision.
3. Faciliter la visualisation des anomalies détectées pour aider les radiologues à les interpréter.
4. Garantir que la reconstruction des images à partir des données réduites reste fidèle aux informations originales, en minimisant la perte d'information.

**Solution mise en place :**
L'équipe de data science a mis en œuvre un algorithme qui :

- Identifie les directions principales dans les données d'image, là où la variance est la plus élevée, et réduit la dimensionnalité en projetant les images sur ces axes principaux.
- Approxime chaque pixel comme une combinaison linéaire de ces directions principales, révélant les composantes les plus significatives, tout en réduisant le nombre de variables à traiter.
- Permet la reconstruction des images à partir de la version réduite, tout en minimisant la perte d'information critique.
- Facilite la visualisation des anomalies détectées dans cet espace réduit, permettant aux radiologues d'interpréter les résultats de manière efficace.

**Résultats :**
- La dimensionnalité des images a été réduite de 98%, passant de plusieurs millions de pixels à quelques centaines de variables tout en capturant les principales sources de variance.
- Le temps de traitement des images a diminué de 85%.
- La reconstruction des images à partir des données réduites a permis de préserver 95% des informations essentielles.
- Les anomalies détectées ont été visualisées et interprétées avec succès par les radiologues, qui ont pu se concentrer sur les zones critiques.



----------------------------------------------------------------------------------------
### **Étude de cas 2 : Analyse de données génomiques (6 points)**
----------------------------------------------------------------------------------------

Un laboratoire de recherche en génétique étudie l'expression des gènes dans différentes conditions environnementales. Les chercheurs disposent d'un vaste ensemble de données d'expression génique, avec des milliers de gènes mesurés pour chaque échantillon. L'objectif est de découvrir des motifs complexes et non linéaires dans ces données.

L'équipe de bio-informatique a mis en place une solution avec les caractéristiques suivantes :

1. Capture des relations entre les gènes, incluant des interactions du type $$y = e^x$$ ou $$y = x^2$$.
2. Réduit la dimensionnalité des données tout en préservant les informations essentielles.
3. Utilise des fonctions d'activation non linéaires dans son architecture.
4. Permet une reconstruction des données originales à partir de la représentation compressée.
5. Apprend automatiquement les caractéristiques les plus pertinentes sans supervision.

**Résultats obtenus :**
- Réduction de 95% de la dimensionnalité des données d'expression génique.
- Identification de motifs d'expression génique auparavant inconnus.
- Amélioration de 40% de la précision dans la prédiction des réponses cellulaires aux stimuli environnementaux.




----------------------------------------------------------------------------------------
### **Étude de cas 3 : Détection de défauts dans la production de semi-conducteurs (6 points)**
----------------------------------------------------------------------------------------

Une entreprise de fabrication de semi-conducteurs cherche à améliorer son contrôle qualité en détectant automatiquement les défauts de production. Les wafers de silicium sont photographiés à haute résolution à différentes étapes de la production.

**Contexte :**
- Des millions d'images de wafers sont générées chaque jour.
- Les défauts sont rares mais coûteux s'ils ne sont pas détectés à temps.
- Les images sont de très haute dimension (plusieurs millions de pixels chacune).
- Les variations entre les images normales semblent suivre des tendances régulières et proportionnelles.

**Défis :**
1. Réduire la dimensionnalité des images tout en capturant les directions de variance maximale.
2. Identifier les composantes principales qui expliquent la majorité de la variabilité dans les données.
3. Projeter les données dans un espace de dimension réduite pour faciliter l'analyse.
4. Traiter un grand volume de données en temps réel avec des ressources de calcul limitées.

**Solution mise en place :**
L'équipe de data science a implémenté un algorithme qui :

- Calcule la matrice de covariance des pixels des images.
- Identifie les vecteurs propres et les valeurs propres de cette matrice.
- Projette les images sur les vecteurs propres correspondant aux plus grandes valeurs propres.
- Réduit la dimensionnalité en ne conservant que les projections les plus significatives.
- Permet de reconstruire approximativement les images originales à partir des projections.

**Résultats :**
- La dimensionnalité des images a été réduite de 99%, passant de 5 millions à 50 000 caractéristiques.
- 95% de la variance totale des données est expliquée par les composantes retenues.
- Le temps de traitement par image a été réduit à 100 ms, permettant une inspection en temps réel.
- Les défauts sont détectés en comparant les projections des images aux projections typiques des images sans défaut.





----------------------------------------------------------------------------------------
### **Étude de cas 4 : Analyse des comportements clients dans un système bancaire (6 points)**
----------------------------------------------------------------------------------------

Une grande banque internationale cherche à mieux comprendre les comportements de ses clients à travers l'analyse de millions de transactions quotidiennes. L'objectif est d'identifier des groupes de clients aux comportements similaires pour personnaliser les services et améliorer l'expérience client.

**Contexte :**
- Les données comprennent des millions de transactions, chacune avec de nombreuses caractéristiques (montant, heure, localisation, type de compte, etc.).
- Les comportements des clients sont variés et évoluent constamment.
- La banque ne dispose pas d'étiquettes prédéfinies pour les différents types de comportements.

**Défis :**
1. Identifier des groupes de clients aux comportements similaires sans définition préalable de ces groupes.
2. Gérer un grand volume de données avec des ressources de calcul limitées.
3. S'adapter à l'évolution des comportements des clients sans nécessiter de réentraînement constant.
4. Traiter des données de grande dimension de manière efficace.

**Solution mise en place :**
L'équipe de data science a implémenté un algorithme qui :
- Regroupe les transactions en fonction de leurs similitudes dans un espace multidimensionnel.
- Fonctionne de manière non supervisée, sans nécessiter d'étiquettes prédéfinies.
- Est capable de gérer des données de grande dimension.
- Utilise une approche basée sur la densité pour identifier des groupes de formes variées.
- Peut s'adapter à de nouvelles données sans nécessiter un réentraînement complet.

**Résultats :**
- L'algorithme a identifié 8 groupes distincts de comportements clients.
- Il a révélé des patterns de transactions auparavant inconnus.
- Le temps de traitement a été réduit de 60% par rapport aux méthodes précédentes.
- La segmentation obtenue a permis d'améliorer la personnalisation des services bancaires.





----------------------------------------------------------------------------------------
###  **Étude de cas 5: Analyse des données de prospection géologique (6 points)**
----------------------------------------------------------------------------------------

Une équipe de géologues utilise des drones équipés de détecteurs spécialisés pour localiser des gisements minéraux potentiels dans une région montagneuse. Après plusieurs semaines de collecte de données, l'équipe fait face à un défi d'analyse.

**Contexte :**
- Les drones ont collecté des données sur des milliers de points d'intérêt.
- Chaque point est caractérisé par sa position GPS, l'intensité du signal minéral, la profondeur estimée, et d'autres mesures géologiques.
- L'objectif est d'identifier des groupes de points qui pourraient représenter des gisements minéraux importants.

**Défis :**
1. Traiter un grand volume de données multidimensionnelles.
2. Identifier des groupes de points sans connaître à l'avance le nombre de gisements potentiels.
3. Détecter des groupes de forme irrégulière, car les gisements peuvent suivre des caractéristiques géologiques.
4. Gérer le bruit dans les données dû aux formations rocheuses sans intérêt économique.

**Solution mise en place :**
L'équipe d'analyse a implémenté un algorithme qui :
- Groupe les points d'intérêt en fonction de leur proximité spatiale et de la similarité des signaux détectés.
- Ne nécessite pas de spécifier à l'avance le nombre de groupes.
- Peut identifier des clusters de forme arbitraire.
- Est capable de distinguer les zones denses (potentiels gisements) des points isolés (probablement sans intérêt).
- Identifie automatiquement les points considérés comme du bruit.

**Résultats :**
- L'algorithme a identifié 23 clusters potentiels de gisements minéraux.
- Il a révélé des formes de clusters suivant des caractéristiques géologiques comme des failles ou des plis.
- 15% des points ont été classés comme bruit, correspondant probablement à des formations rocheuses isolées sans intérêt économique.
- La visualisation des résultats a permis une meilleure compréhension de la distribution spatiale des gisements potentiels.




