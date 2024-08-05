# Étude de Cas : Détection des Comportements Anormaux des Employés avec IsolationForest

# Contexte

Dans cette étude de cas, nous utilisons l'algorithme IsolationForest pour détecter les comportements anormaux parmi les employés d'une entreprise. L'objectif est d'identifier les employés qui présentent des comportements problématiques tels que des plaintes fréquentes, des absences élevées, des heures de travail atypiques et des scores de performance faibles.

# Étapes de l'analyse

1. **Importation des bibliothèques nécessaires**
   ```python
   import pandas as pd
   from sklearn.ensemble import IsolationForest
   import matplotlib.pyplot as plt
   ```

2. **Création du jeu de données fictif**
   Nous avons créé un jeu de données fictif pour illustrer notre cas. Ce jeu de données contient les informations suivantes pour 100 employés :
   - `employee_id` : Identifiant de l'employé.
   - `num_complaints` : Nombre de plaintes.
   - `num_absences` : Nombre d'absences de l'employé.
   - `work_hours` : Nombre d'heures travaillées par semaine.
   - `performance_score` : Score de performance de l'employé.
   ```python
   data = {
       'employee_id': range(1, 101),
       'num_complaints': [0, 1, 0, 2, 0, 20, 0, 0, 3, 0] * 10,
       'num_absences': [1, 0, 0, 0, 1, 20, 2, 1, 0, 0] * 10,
       'work_hours': [40, 38, 40, 35, 40, 42, 40, 39, 40, 40] * 10,
       'performance_score': [3, 4, 3, 2, 3, 1, 3, 4, 3, 3] * 10
   }
   df = pd.DataFrame(data)
   ```

3. **Affichage des premières lignes du DataFrame**
   ```python
   print(df.head())
   ```

4. **Définition des fonctionnalités pour le modèle**
   Nous sélectionnons les colonnes qui serviront de caractéristiques pour l'algorithme IsolationForest.
   ```python
   features = ['num_complaints', 'num_absences', 'work_hours', 'performance_score']
   ```

5. **Application de l'IsolationForest pour détecter les anomalies**
   Nous entraînons le modèle IsolationForest en supposant que 10% des employés présentent des comportements problématiques.
   ```python
   clf = IsolationForest(contamination=0.1)
   clf.fit(df[features])
   df['anomaly'] = clf.predict(df[features])
   ```

6. **Remplacement des valeurs d'anomalie par des étiquettes lisibles**
   ```python
   df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Problématique'})
   ```

7. **Comptage et affichage du nombre d'employés normaux et problématiques**
   ```python
   print(df['anomaly'].value_counts())
   ```

8. **Visualisation des résultats**
   Nous visualisons les employés normaux et problématiques en utilisant un scatter plot.
   ```python
   normal = df[df['anomaly'] == 'Normal']
   problematic = df[df['anomaly'] == 'Problématique']
   plt.scatter(normal['num_complaints'], normal['num_absences'], label='Normal', c='green')
   plt.scatter(problematic['num_complaints'], problematic['num_absences'], label='Problématique', c='red')
   plt.xlabel('Nombre de Plaintes')
   plt.ylabel('Nombre d\'Absences')
   plt.title('Détection des Employés Problématiques')
   plt.legend()
   plt.show()
   ```

9. **Affichage des employés problématiques**
   ```python
   print(df[df['anomaly'] == 'Problématique'])
   ```

### Conclusion
Cette étude de cas montre comment utiliser l'algorithme IsolationForest pour identifier les employés présentant des comportements anormaux. En visualisant et en analysant les résultats, l'entreprise peut prendre des mesures correctives pour résoudre les problèmes identifiés.
