# Exercice : Trouvez les erreurs dans ce code :


## 1. import pandas as pd
## 2. restaurants = pd.read_csv('path_to_restaurants_csv')
## 3. categories = pd.read_csv('path_to_categories_csv')
## 4. df_temp_1 = pd.merge(restaurants, categories, how='left', on='restaurant_id')
## 5. df_temp_1.reset_index(drop=True, inplace=True)
## 6. cnt = df_temp_1.groupby(['zone', 'categorie']).size().to_frame('count')
## 7.df_temp_2 = pd.merge(restaurants, cnt, how='left', on='zone')
## 8.df_temp_2.rename(columns={'count': 'zone_categories_intersection'}, inplace=True)
## 9. df_temp_3 = df_temp_2.drop_duplicates(['restaurant_id', 'nom', 'moyenne_etoiles', 'ville', 'zone', 'ferme']).groupby('restaurant_id').agg({'zone_categories_intersection': 'sum'}).reset_index()
## 10. features = pd.merge(features, df_temp_3, how='left', on='restaurant_id')
## 11. features.head()



```python

import pandas as pd
restaurants = pd.read_csv('path_to_restaurants_csv')
categories = pd.read_csv('path_to_categories_csv')
df_temp_1 = pd.merge(restaurants, categories, how='left', on='restaurant_id')
df_temp_1.reset_index(drop=True, inplace=True)
cnt = df_temp_1.groupby(['zone', 'categorie']).size().to_frame('count')
# Oupsssss erreuuuuurrrrr ici: Fusion incorrecte avec seulement 'zone'
# Cette fusion ne prend pas en compte la catégorie, ce qui entraîne une mauvaise association des comptes
df_temp_2 = pd.merge(restaurants, cnt, how='left', on='zone')
df_temp_2.rename(columns={'count': 'zone_categories_intersection'}, inplace=True)
# Oupsssss erreuuuuurrrrr ici: Agrégation avant de supprimer les duplicatas
# Il est incorrect de supprimer les duplicatas avant le groupby, cela peut entraîner des erreurs dans le comptage
df_temp_3 = df_temp_2.drop_duplicates(['restaurant_id', 'nom', 'moyenne_etoiles', 'ville', 'zone', 'ferme']).groupby('restaurant_id').agg({'zone_categories_intersection': 'sum'}).reset_index()
features = pd.merge(features, df_temp_3, how='left', on='restaurant_id')
features.head()

```


#********************************************************************************************************************************************
# 5) zone_categories_intersection
#********************************************************************************************************************************************
# Voici l'ancien code avec les erreurs soulignées:

import pandas as pd

# Charger les données à partir des fichiers CSV
# restaurants = pd.read_csv('path_to_restaurants_csv')
# categories = pd.read_csv('path_to_categories_csv')

# Fusionner les tables des restaurants et des catégories
df_temp_1 = pd.merge(restaurants, categories, how='left', on='restaurant_id')
df_temp_1.reset_index(drop=True, inplace=True)

# Compter le nombre de restaurants par zone et catégorie
cnt = df_temp_1.groupby(['zone', 'categorie']).size().to_frame('count')

# Oupsssss erreuuuuurrrrr ici: Fusion incorrecte avec seulement 'zone'
# Cette fusion ne prend pas en compte la catégorie, ce qui entraîne une mauvaise association des comptes
df_temp_2 = pd.merge(restaurants, cnt, how='left', on='zone')

df_temp_2.rename(columns={'count': 'zone_categories_intersection'}, inplace=True)

# Supprimer les duplicatas et regrouper par restaurant
# Oupsssss erreuuuuurrrrr ici: Agrégation avant de supprimer les duplicatas
# Il est incorrect de supprimer les duplicatas avant le groupby, cela peut entraîner des erreurs dans le comptage
df_temp_3 = df_temp_2.drop_duplicates(['restaurant_id', 'nom', 'moyenne_etoiles', 'ville', 'zone', 'ferme']).groupby('restaurant_id').agg({'zone_categories_intersection': 'sum'}).reset_index()

# Ajouter la colonne 'zone_categories_intersection' au DataFrame des features
features = pd.merge(features, df_temp_3, how='left', on='restaurant_id')

features.head()
```

### Explications détaillées des erreurs

1. **Erreur de fusion incorrecte** :
   ```python
   # Oupsssss erreuuuuurrrrr ici: Fusion incorrecte avec seulement 'zone'
   df_temp_2 = pd.merge(restaurants, cnt, how='left', on='zone')
   ```
   - **Erreur** : La fusion n'inclut pas `categorie`, ce qui entraîne une mauvaise association des comptes. Cela signifie que le comptage des restaurants ne correspondra pas correctement à chaque catégorie dans une zone donnée.
   - **Correction** : Inclure à la fois `zone` et `categorie` pour la fusion afin de s'assurer que chaque restaurant est correctement associé à ses comptes de catégorie et de zone.
   ```python
   df_temp_2 = pd.merge(df_temp_1, cnt, how='left', on=['zone', 'categorie'])
   ```

2. **Erreur de duplication incorrecte** :
   ```python
   # Oupsssss erreuuuuurrrrr ici: Agrégation avant de supprimer les duplicatas
   df_temp_3 = df_temp_2.drop_duplicates(['restaurant_id', 'nom', 'moyenne_etoiles', 'ville', 'zone', 'ferme']).groupby('restaurant_id').agg({'zone_categories_intersection': 'sum'}).reset_index()
   ```
   - **Erreur** : Supprimer les duplicatas après le regroupement peut entraîner une perte de données ou des doublons incorrects. Cela pourrait causer des erreurs où plusieurs lignes pour le même restaurant pourraient être agrégées de manière incorrecte.
   - **Correction** : Supprimer les duplicatas avant de regrouper et d'agréger les données. Cela garantit que chaque restaurant est unique avant de calculer les totaux.
   ```python
   df_temp_3 = df_temp_2.drop_duplicates(subset=['restaurant_id', 'nom', 'moyenne_etoiles', 'ville', 'zone', 'ferme'])
   df_temp_3 = df_temp_3.groupby('restaurant_id').agg({'zone_categories_intersection': 'sum'}).reset_index()
   ```

### Commentaires pour les débutants

- **Charger les données** : Utilisez `pd.read_csv('path_to_file')` pour charger les fichiers CSV contenant les données des restaurants et des catégories. Assurez-vous que les chemins des fichiers sont corrects.
- **Fusionner les tables** : Utilisez `pd.merge` pour fusionner les DataFrames sur une colonne clé commune, ici `restaurant_id`. Cela permet de combiner les informations des deux fichiers en un seul DataFrame.
- **Compter les occurrences** : Utilisez `groupby` pour regrouper les données par `zone` et `categorie`, puis `size()` pour compter le nombre de restaurants dans chaque groupe. `reset_index()` transforme le résultat en DataFrame.
- **Fusionner les comptes** : Fusionnez les comptes obtenus avec le DataFrame initial pour ajouter une nouvelle colonne indiquant le nombre de restaurants partageant la même zone et catégorie.
- **Supprimer les doublons** : Utilisez `drop_duplicates` pour supprimer les lignes dupliquées, en conservant uniquement une ligne par restaurant. Cela garantit que chaque restaurant est unique dans le DataFrame.
- **Grouper et sommer** : Utilisez `groupby` pour regrouper les données par `restaurant_id`, puis `agg` pour sommer les valeurs de `zone_categories_intersection`. Cela permet d'obtenir le nombre total de restaurants de la même zone et catégorie pour chaque restaurant.
- **Ajouter la colonne** : Fusionnez le DataFrame résultant avec le DataFrame des features existantes pour ajouter les nouvelles informations. Utilisez `pd.merge` pour cette fusion.
- **Afficher les résultats** : Utilisez `head()` pour afficher les premières lignes du DataFrame final et vérifier les résultats.

En suivant ces étapes et en corrigeant les erreurs, le code devrait maintenant fonctionner correctement et fournir les résultats attendus.






