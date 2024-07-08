# Visualisation et Optimisation de DBSCAN

### Objectif du Projet

Le but de ce projet est de visualiser l'algorithme de clustering DBSCAN (Density-Based Spatial Clustering of Applications with Noise) et d'explorer l'influence des différents paramètres sur les résultats du clustering. L'application vise à trouver les meilleurs paramètres (`eps` et `min_samples`) qui donnent le meilleur score de silhouette, indiquant ainsi la qualité optimale du clustering.

### Description

Ce projet fournit une interface graphique interactive, développée avec Tkinter et Matplotlib, permettant de :
- Générer un ensemble de points de données simulés.
- Ajuster les paramètres `eps` (distance maximale entre deux échantillons pour qu'ils soient considérés comme voisins) et `min_samples` (nombre minimum d'échantillons dans un voisinage pour qu'un point soit considéré comme un noyau).
- Démarrer et visualiser le processus de clustering DBSCAN étape par étape.
- Mettre en pause et reprendre le clustering.
- Ajouter des points de données supplémentaires en cliquant sur la visualisation.
- Tester plusieurs combinaisons de paramètres et exporter les résultats.

### Fonctionnalités

- **Génération de Données :** Crée un ensemble de points de données basé sur des centres prédéfinis.
- **Contrôle des Paramètres :** Interface utilisateur pour ajuster le nombre de points, `eps` et `min_samples`.
- **Visualisation en Temps Réel :** Affiche les clusters formés à chaque étape du processus DBSCAN.
- **Pause et Reprise :** Permet de mettre en pause et de reprendre le processus de clustering.
- **Ajout de Points Manuels :** Permet d'ajouter des points de données supplémentaires en cliquant sur la visualisation.
- **Exportation des Résultats :** Exporte l'historique des paramètres testés et des scores de silhouette dans un fichier CSV.
- **Test de Paramètres Multiples :** Automatise le test de plusieurs combinaisons de paramètres pour trouver les meilleurs résultats basés sur le score de silhouette.
- **Classement des Résultats :** Classe les résultats de clustering par score de silhouette pour identifier les meilleures combinaisons de paramètres.

### Installation

1. **Prérequis :**
   - Python 3.6 ou plus récent
   - `virtualenv` pour créer des environnements virtuels

2. **Créer un Environnement Virtuel :**
   ```bash
   python -m venv dbscan-env
   ```

3. **Activer l'Environnement Virtuel :**
   - Sous Windows :
     ```bash
     dbscan-env\Scripts\activate
     ```
   - Sous macOS et Linux :
     ```bash
     source dbscan-env/bin/activate
     ```

4. **Installer les Dépendances :**
   ```bash
   pip install tkinter numpy matplotlib scikit-learn
   ```

5. **Exécuter le Script :**
   - Sauvegardez le script dans un fichier nommé `dbscan_visualizer.py`.
   - Exécutez le script avec la commande :
     ```bash
     python dbscan_visualizer.py
     ```

### Utilisation

1. **Paramétrage :** Ajustez les paramètres du clustering (nombre de points, `eps`, `min_samples`) via l'interface utilisateur.
2. **Démarrer le Clustering :** Cliquez sur "Start" pour lancer le processus de clustering.
3. **Pause/Reprise :** Utilisez le bouton "Pause" pour mettre en pause le processus, et "Resume" pour le reprendre.
4. **Ajout de Points :** Cliquez sur la visualisation pour ajouter de nouveaux points de données.
5. **Exportation :** Exportez l'historique des résultats dans un fichier CSV en utilisant le bouton approprié.
6. **Test Automatique :** Utilisez les boutons "Test Multiple Parameters" et "Run All Combinations" pour tester automatiquement différentes combinaisons de paramètres et identifier celles qui produisent le meilleur score de silhouette.

### Conclusion

Cette application permet de mieux comprendre l'algorithme DBSCAN et l'influence de ses paramètres sur le clustering. Elle offre une manière interactive et visuelle d'explorer les résultats, facilitant ainsi l'optimisation des paramètres de DBSCAN pour obtenir le meilleur score de silhouette possible, indiquant la qualité optimale du clustering.
