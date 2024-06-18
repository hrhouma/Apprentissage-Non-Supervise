- Pour exécuter ce script sur Amazon SageMaker, vous pouvez suivre ces étapes. 
- Assurez-vous d'avoir un compte AWS configuré avec les permissions nécessaires pour utiliser SageMaker.

1. **Créez un notebook Jupyter sur Amazon SageMaker** :

    - Connectez-vous à votre console AWS.
    - Accédez à SageMaker > Notebooks > Notebook instances.
    - Créez une nouvelle instance de notebook et lancez-la.
    - Ouvrez le Jupyter Notebook une fois que l'instance est en cours d'exécution.

2. **Téléchargez vos données** :

    - Téléchargez le fichier CSV `happiness_report.csv` dans votre notebook SageMaker.
    - Vous pouvez utiliser l'interface utilisateur Jupyter pour télécharger directement le fichier dans l'environnement de votre notebook.

3. **Installez les bibliothèques nécessaires** (si elles ne sont pas déjà installées) :

    ```python
    !pip install pandas seaborn matplotlib scikit-learn plotly
    ```

4. **Utilisez le script suivant pour analyser et segmenter le Rapport Mondial sur le Bonheur** :

```python
# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot

# Charger les données depuis le fichier CSV
data = pd.read_csv('happiness_report.csv')

# Analyse exploratoire des données
print(data.info())
print(data.isnull().sum())
print(data.describe())

# Visualisation des données
sns.pairplot(data[['Score', 'GDP per capita', 'Social support', 'Healthy life expectancy']])
plt.show()

# Histogrammes
columns = ['Score', 'GDP per capita', 'Social support', 'Healthy life expectancy',
           'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
plt.figure(figsize=(20, 50))
for i in range(len(columns)):
    plt.subplot(8, 2, i+1)
    sns.histplot(data[columns[i]], kde=True)
    plt.title(columns[i])
plt.tight_layout()
plt.show()

# Matrice de corrélation
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matrice de corrélation des variables numériques')
plt.show()

# Préparation des données pour le clustering
data_for_clustering = data.drop(['Overall rank', 'Score'], axis=1)
data_numeric = data_for_clustering.select_dtypes(include=[np.number])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_numeric)

# Détermination du nombre optimal de clusters avec la méthode du coude
scores = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    scores.append(kmeans.inertia_)

plt.plot(range(1, 10), scores, 'bx-')
plt.title('Finding the right number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (scores)')
plt.show()

# Application de la méthode K-means avec le nombre optimal de clusters (par exemple, 3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

# Centroides des clusters
cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=data_numeric.columns)
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data=cluster_centers, columns=data_numeric.columns)
print(cluster_centers)

# Labels associés à chaque point de données
labels = kmeans.labels_
happy_df_cluster = pd.concat([data, pd.DataFrame({'cluster': labels})], axis=1)
print(happy_df_cluster.head())

# Visualisation des clusters
for i in data_numeric.columns:
    plt.figure(figsize=(35, 10))
    for j in range(3):  # Supposons qu'il y ait 3 clusters
        plt.subplot(1, 3, j + 1)
        cluster = happy_df_cluster[happy_df_cluster['cluster'] == j]
        cluster[i].hist(bins=20)
        plt.title('{}    \nCluster {} '.format(i, j))
    plt.show()

# Visualisation des clusters avec Plotly
fig = px.scatter(happy_df_cluster, x='GDP per capita', y='Perceptions of corruption', size='Score', color='cluster', hover_name='Country or region')
fig.update_layout(title='Clusters based on Economy, Corruption and Happiness')
fig.show()

# Visualisation géographique des clusters
data_geo = dict(type='choropleth',
                locations=happy_df_cluster["Country or region"],
                locationmode='country names',
                colorscale='RdYlGn',
                z=happy_df_cluster['cluster'],
                text=happy_df_cluster["Country or region"],
                colorbar={'title': 'Clusters'})

layout_geo = dict(title='Geographical Visualization of Clusters',
                  geo=dict(showframe=True, projection={'type': 'azimuthal equal area'}))

choromap3 = go.Figure(data=[data_geo], layout=layout_geo)
iplot(choromap3)
```

### Explication des étapes :

1. **Importation des bibliothèques** :
    - Importez toutes les bibliothèques nécessaires pour le traitement des données, la visualisation et le clustering.

2. **Chargement des données** :
    - Utilisez `pd.read_csv` pour charger les données depuis le fichier CSV.

3. **Analyse exploratoire des données** :
    - Affichez des informations générales sur les données, vérifiez les valeurs nulles, et obtenez des statistiques descriptives.

4. **Visualisation des données** :
    - Utilisez `seaborn` et `matplotlib` pour visualiser les relations entre les variables et tracer des histogrammes des différentes colonnes.

5. **Matrice de corrélation** :
    - Calculez et visualisez la matrice de corrélation pour comprendre les relations entre les variables.

6. **Préparation des données pour le clustering** :
    - Sélectionnez les colonnes numériques, standardisez les données et préparez-les pour le clustering.

7. **Méthode du coude** :
    - Déterminez le nombre optimal de clusters en utilisant la méthode du coude et tracez les scores d'inertie.

8. **Application de l'algorithme K-means** :
    - Appliquez l'algorithme K-means avec le nombre optimal de clusters, obtenez les centroides des clusters et les labels des clusters pour chaque point de données.

9. **Visualisation des clusters** :
    - Utilisez `seaborn` et `matplotlib` pour tracer des histogrammes des différentes colonnes selon les clusters.
    - Utilisez `plotly` pour des visualisations interactives des clusters.

10. **Visualisation géographique des clusters** :
    - Utilisez `plotly` pour créer une carte choroplèthe visualisant les clusters géographiquement.

### Note

- Assurez-vous que le fichier CSV est correctement téléchargé dans le répertoire de travail de votre notebook SageMaker.
- Vous pouvez ajuster le nombre de clusters dans le script selon les résultats de la méthode du coude.
- Les visualisations interactives de `plotly` nécessitent que vous exécutiez ce script dans un environnement prenant en charge le rendu de `plotly`.

Avec ce script, vous pouvez analyser, segmenter et visualiser les données du Rapport Mondial sur le Bonheur en utilisant l'algorithme K-means sur Amazon SageMaker.
