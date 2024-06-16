### 📊 KMeans Bank Customer Segmentation

Bienvenue dans le projet KMeans Bank Customer Segmentation ! Ce projet utilise l'Analyse en Composantes Principales (ACP) pour faciliter la segmentation des clients bancaires. Ce README explique pourquoi et comment l'ACP est utilisée dans ce projet, et à quoi cela ressemblerait sans l'ACP.

## 🚀 Objectif de l'ACP

L'Analyse en Composantes Principales (ACP) est utilisée dans ce projet pour simplifier les données complexes et améliorer la visualisation des clusters de clients bancaires. Les données initiales contiennent de nombreuses variables, ce qui peut rendre la visualisation et l'interprétation des résultats de clustering difficile. L'ACP nous aide à réduire ces données à deux dimensions principales tout en conservant le maximum d'information possible.

## Pourquoi utiliser l'ACP ?

1. **Réduction de la dimensionnalité** : Les données de clients bancaires contiennent plusieurs variables (solde, fréquence des achats, limites de crédit, etc.). L'ACP transforme ces nombreuses variables en quelques nouvelles variables (composantes principales), réduisant ainsi la complexité des données.

2. **Facilitation de la visualisation** : En réduisant les données à deux dimensions, l'ACP permet de visualiser les clusters de clients de manière simple et claire sur un graphique 2D.

3. **Conservation de l'information** : L'ACP sélectionne les composantes principales qui capturent la majorité de la variance des données originales, assurant ainsi que l'information essentielle est préservée.

### Contexte du Projet

Dans ce projet, nous avons un ensemble de données de clients bancaires comprenant diverses variables comme le solde, la fréquence des achats, les paiements, etc. Voici pourquoi l'ACP est essentielle dans ce contexte :

### Exemple de Données

Voici un extrait des données de clients bancaires :

| CUST_ID | BALANCE  | BALANCE_FREQUENCY | PURCHASES | ONEOFF_PURCHASES | INSTALLMENTS_PURCHASES | CASH_ADVANCE | PURCHASES_FREQUENCY | ONEOFF_PURCHASES_FREQUENCY | PURCHASES_INSTALLMENTS_FREQUENCY | CASH_ADVANCE_FREQUENCY | CASH_ADVANCE_TRX | PURCHASES_TRX | CREDIT_LIMIT | PAYMENTS   | MINIMUM_PAYMENTS | PRC_FULL_PAYMENT | TENURE |
|---------|----------|-------------------|-----------|------------------|-----------------------|--------------|---------------------|---------------------------|----------------------------------|-----------------------|------------------|----------------|--------------|------------|------------------|------------------|--------|
| C10001  | 40.90    | 0.818             | 95.4      | 0                | 95.4                  | 0            | 0.167               | 0                         | 0.083                           | 0                     | 0                | 2              | 1000         | 201.80     | 139.51           | 0                | 12     |
| C10002  | 3202.47  | 0.909             | 0         | 0                | 0                     | 6442.95      | 0                   | 0                         | 0                               | 0.25                  | 4                | 0              | 7000         | 4103.03    | 1072.34          | 0.222            | 12     |
| C10003  | 2495.15  | 1                 | 773.17    | 773.17           | 0                     | 0            | 1                   | 1                         | 0                               | 0                     | 0                | 12             | 7500         | 622.07     | 627.28           | 0                | 12     |
| ...     | ...      | ...               | ...       | ...              | ...                   | ...          | ...                 | ...                       | ...                             | ...                   | ...              | ...            | ...          | ...        | ...              | ...              | ...    |

### Sans ACP

Si nous appliquons directement l'algorithme KMeans sur ces données sans ACP, nous devons traiter toutes les variables en même temps. Cela peut causer plusieurs problèmes :

1. **Complexité** : Avec de nombreuses variables, le modèle devient complexe et difficile à interpréter.
2. **Visualisation** : Il est pratiquement impossible de visualiser les clusters dans un espace à haute dimension (plus de 3 dimensions).
3. **Performances** : Plus de variables peuvent ralentir l'algorithme et potentiellement réduire sa performance.

### Avec ACP

L'ACP simplifie les choses en réduisant la dimensionnalité des données :

1. **Standardisation des données** :
   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   data_scaled = scaler.fit_transform(data)
   ```

2. **Application de l'ACP** :
   ```python
   from sklearn.decomposition import PCA

   pca = PCA(n_components=2)
   principal_comp = pca.fit_transform(data_scaled)
   pca_df = pd.DataFrame(data=principal_comp, columns=['pca1', 'pca2'])
   ```

### Résultat avec ACP

Après avoir appliqué l'ACP, les données sont réduites à deux dimensions principales :

| pca1  | pca2  | cluster |
|-------|-------|---------|
| 1.23  | -0.34 | 0       |
| -0.98 | 1.05  | 1       |
| ...   | ...   | ...     |

Nous pouvons maintenant visualiser facilement les clusters en utilisant un graphique de dispersion 2D :

```python
import plotly.express as px
from plotly.offline import plot

fig = px.scatter(pca_df, x='pca1', y='pca2', color=data['cluster'], title='Customer Segmentation with PCA')
plot(fig, filename='cluster_plot.html', auto_open=True)
```

### Détail de l'ACP dans ce projet

1. **Chargement et Prétraitement des Données** :
   - Les données des clients bancaires sont chargées depuis un fichier CSV et les valeurs manquantes sont remplies avec la moyenne de la colonne concernée.
   - Les données sont normalisées pour s'assurer que chaque variable ait une moyenne de 0 et une variance de 1.

   ```python
   import pandas as pd

   data = pd.read_csv("bank_customers.csv")
   data.fillna(data.mean(), inplace=True)
   scaler = StandardScaler()
   data_scaled = scaler.fit_transform(data.drop(columns=['CUST_ID']))
   ```

2. **Application de l'ACP** :
   - L'ACP est appliquée pour réduire les données à 2 dimensions principales. Ces nouvelles variables (pca1 et pca2) capturent la majorité de la variance des données originales, ce qui signifie qu'elles représentent les motifs les plus importants dans les données.

   ```python
   from sklearn.decomposition import PCA

   pca = PCA(n_components=2)
   principal_comp = pca.fit_transform(data_scaled)
   pca_df = pd.DataFrame(data=principal_comp, columns=['pca1', 'pca2'])
   ```

3. **Clustering avec KMeans** :
   - Les données réduites (composantes principales) sont ensuite utilisées pour le clustering avec l'algorithme KMeans.

   ```python
   from sklearn.cluster import KMeans

   kmeans = KMeans(n_clusters=4, random_state=42)
   kmeans.fit(principal_comp)
   pca_df['cluster'] = kmeans.labels_
   ```

4. **Visualisation** :
   - Les résultats des clusters sont visualisés en utilisant un graphique de dispersion 2D des composantes principales. Chaque point du graphique représente un client, et les couleurs indiquent les différents clusters.

   ```python
   import plotly.express as px
   from plotly.offline import plot

   fig = px.scatter(pca_df, x='pca1', y='pca2', color='cluster', title='Customer Segmentation with PCA')
   plot(fig, filename='cluster_plot.html', auto_open=True)
   ```

### Conclusion

L'ACP est un outil essentiel dans ce projet pour simplifier les données de clients bancaires et améliorer la visualisation des clusters. En réduisant la dimensionnalité des données, l'ACP permet de représenter visuellement les résultats du clustering de manière claire et compréhensible, tout en conservant l'information essentielle. Sans l'ACP, la visualisation et l'interprétation des clusters seraient beaucoup plus difficiles, rendant l'analyse moins efficace et plus complexe.
