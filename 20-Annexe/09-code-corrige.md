# 1 - Initialiser les variables temporaires pour s'assurer qu'elles sont réinitialisées à chaque exécution

```python
import pandas as pd

# Charger les données à partir des fichiers CSV
# restaurants = pd.read_csv('path_to_restaurants_csv')
# categories = pd.read_csv('path_to_categories_csv')

features = restaurants[['restaurant_id', 'moyenne_etoiles', 'ville', 'zone']].copy()
df_temp_1 = None
cnt = None
df_temp_2 = None
df_temp_3 = None
```


# 2 - Fusionner les tables des restaurants et des catégories en utilisant 'restaurant_id' comme clé commune
# Cela permet de combiner les informations des restaurants avec leurs catégories respectives

```python
df_temp_1 = pd.merge(restaurants, categories, how='left', on='restaurant_id')
df_temp_1.reset_index(drop=True, inplace=True)  # Réinitialiser les index du DataFrame après la fusion pour garder un ordre cohérent
```

# Compter le nombre de restaurants par zone et catégorie
# Utiliser groupby pour regrouper par 'zone' et 'categorie', puis compter les occurrences
# Cela nous donne le nombre de restaurants dans chaque combinaison de zone et de catégorie

```python
cnt = df_temp_1.groupby(['zone', 'categorie']).size().to_frame('count').reset_index()
```

# Fusionner les comptes obtenus avec les données des restaurants en incluant 'categorie'
# Cela ajoute le nombre de restaurants de la même zone et catégorie à chaque ligne du DataFrame

```python
df_temp_2 = pd.merge(df_temp_1, cnt, how='left', on=['zone', 'categorie'])
df_temp_2.rename(columns={'count': 'zone_categories_intersection'}, inplace=True)  # Renommer la colonne 'count' en 'zone_categories_intersection'
```

# Supprimer les doublons pour obtenir un DataFrame unique pour chaque restaurant
## Utiliser drop_duplicates pour enlever les lignes dupliquées basées sur les colonnes spécifiées
## Cela garantit que chaque restaurant apparaît une seule fois dans le DataFrame

```python
df_temp_3 = df_temp_2.drop_duplicates(subset=['restaurant_id', 'nom', 'moyenne_etoiles', 'ville', 'zone', 'ferme'])
```

# Grouper par 'restaurant_id' et sommer les valeurs de 'zone_categories_intersection'
# Cela permet d'obtenir le nombre total de restaurants de la même zone et catégorie pour chaque restaurant

```python
df_temp_3 = df_temp_3.groupby('restaurant_id').agg({'zone_categories_intersection': 'sum'}).reset_index()
```

# Ajouter la colonne 'zone_categories_intersection' au DataFrame 'features' en fusionnant les DataFrames
# Cela combine les nouvelles informations avec le DataFrame des features existantes

```python
features = pd.merge(features, df_temp_3, how='left', on='restaurant_id')
# Afficher les premières lignes du DataFrame 'features' pour vérifier le résultat
features.head()
```
