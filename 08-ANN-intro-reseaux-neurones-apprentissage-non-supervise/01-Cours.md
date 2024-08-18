## **Table des Matières**

1. [Introduction aux Réseaux de Neurones dans l’Apprentissage Non-Supervisé](#introduction-aux-réseaux-de-neurones-dans-lapprentissage-non-supervisé)
   
2. [Rappel : Apprentissage Supervisé vs Non Supervisé et Concept de Division des Données](#rappel-apprentissage-supervisé-vs-non-supervisé-et-concept-de-division-des-données)
   
   - [Qu'est-ce que l'entraînement et le test ?](#quest-ce-que-lentraînement-et-le-test)
     
   - [Importance de la Division des Données](#importance-de-la-division-des-données)
     
3. [Type de Données pour les Algorithmes d'IA](#type-de-données-pour-les-algorithmes-dia)
   
   - [TensorFlow - Types de Données : Différence entre float32 et float64](#tensorflow---types-de-données--différence-entre-float32-et-float64)
     
   - [Comparaison : Numérique vs Chaîne de Caractères pour l'IA](#comparaison--numérique-vs-chaîne-de-caractères-pour-lia)
     
4. [Rappel du Concept des Variables Catégoriques](#rappel-du-concept-des-variables-catégoriques)
   
   - [Définition des Variables Catégoriques](#définition-des-variables-catégoriques)
     
   - [Importance en Traitement par un Algorithme d'IA](#importance-en-traitement-par-un-algorithme-dia)
     
   - [One-Hot Encoding : Explication et Utilisation](#one-hot-encoding--explication-et-utilisation)
     
5. [Rappel des Données de Validation](#rappel-des-données-de-validation)
   
   - [Qu'est-ce que les Données de Validation ?](#quest-ce-que-les-données-de-validation)
     
   - [Différence avec les Données de Test](#différence-avec-les-données-de-test)
     
6. [Normalisation en IA](#normalisation-en-ia)
   
   - [Qu'est-ce que la Normalisation en IA ?](#quest-ce-que-la-normalisation-en-ia)
     
   - [Processus et Importance de la Normalisation](#processus-et-importance-de-la-normalisation)
     
7. [Introduction au Modèle Séquentiel en Keras](#introduction-au-modèle-séquentiel-en-keras)
   
   - [Création et Compréhension des Modèles Séquentiels](#création-et-compréhension-des-modèles-séquentiels)
     
8. [Exploration des Couches dans un Réseau de Neurones](#exploration-des-couches-dans-un-réseau-de-neurones)
   
   - [La Couche Flatten : Fonctionnalités et Exemples](#la-couche-flatten--fonctionnalités-et-exemples)
     
   - [La Couche de Sortie : Fonctionnalités et Exemples](#la-couche-de-sortie--fonctionnalités-et-exemples)
     
   - [Couche Dense : Définition et Exemples](#couche-dense--définition-et-exemples)
     
9. [Fonction d'Activation](#fonction-dactivation)
   
   - [Qu'est-ce qu'une Fonction d'Activation ?](#quest-ce-quune-fonction-dactivation)
     
   - [Pourquoi Utiliser ReLU et Softmax pour la Sortie ?](#pourquoi-utiliser-relu-et-softmax-pour-la-sortie)
     
10. [Dessiner une Architecture de Réseau de Neurones](#dessiner-une-architecture-de-réseau-de-neurones)
    
    - [Choix des Neurones et des Couches](#choix-des-neurones-et-des-couches)
      
    - [Comment Choisir le Nombre de Neurones dans la Couche de Sortie ?](#comment-choisir-le-nombre-de-neurones-dans-la-couche-de-sortie)

---

## **Cours**

### Introduction aux Réseaux de Neurones dans l’Apprentissage Non-Supervisé **TO DO**  
   [Retour en haut](#table-des-matières)

### 2. Rappel : Apprentissage Supervisé vs Non Supervisé et Concept de Division des Données **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Qu'est-ce que l'entraînement et le test ? **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Importance de la Division des Données **TO DO**  
   [Retour en haut](#table-des-matières)

### 3. Type de Données pour les Algorithmes d'IA **TO DO**  
   [Retour en haut](#table-des-matières)

#### - TensorFlow - Types de Données : Différence entre float32 et float64 **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Comparaison : Numérique vs Chaîne de Caractères pour l'IA **TO DO**  
   [Retour en haut](#table-des-matières)

### 4. Rappel du Concept des Variables Catégoriques **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Définition des Variables Catégoriques **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Importance en Traitement par un Algorithme d'IA **TO DO**  
   [Retour en haut](#table-des-matières)

#### - One-Hot Encoding : Explication et Utilisation **TO DO**  
   [Retour en haut](#table-des-matières)

### 5. Rappel des Données de Validation **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Qu'est-ce que les Données de Validation ? **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Différence avec les Données de Test **TO DO**  
   [Retour en haut](#table-des-matières)

### 6. Normalisation en IA **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Qu'est-ce que la Normalisation en IA ? **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Processus et Importance de la Normalisation **TO DO**  
   [Retour en haut](#table-des-matières)

### 7. Introduction au Modèle Séquentiel en Keras **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Création et Compréhension des Modèles Séquentiels **TO DO**  
   [Retour en haut](#table-des-matières)

### 8. Exploration des Couches dans un Réseau de Neurones **TO DO**  
   [Retour en haut](#table-des-matières)

#### - La Couche Flatten : Fonctionnalités et Exemples **TO DO**  
   [Retour en haut](#table-des-matières)

#### - La Couche de Sortie : Fonctionnalités et Exemples **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Couche Dense : Définition et Exemples **TO DO**  
   [Retour en haut](#table-des-matières)

### 9. Fonction d'Activation **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Qu'est-ce qu'une Fonction d'Activation ? **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Pourquoi Utiliser ReLU et Softmax pour la Sortie ? **TO DO**  
   [Retour en haut](#table-des-matières)

### 10. Dessiner une Architecture de Réseau de Neurones **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Choix des Neurones et des Couches **TO DO**  
   [Retour en haut](#table-des-matières)

#### - Comment Choisir le Nombre de Neurones dans la Couche de Sortie ? **TO DO**  
   [Retour en haut](#table-des-matières)

