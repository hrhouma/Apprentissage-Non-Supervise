# Algorithme de Clustering K-Means : Comprendre son Fonctionnement

# 1 - Introduction
- Dans le monde du commerce de détail, de nombreux propriétaires d'entreprises trouvent difficile de reconnaître les besoins de leurs clients. 
- Les entreprises axées sur les données telles que Netflix, Walmart et Target réussissent exceptionnellement bien parce qu'elles possèdent une armée d'analystes de données certifiés qui utilisent les bons outils pour créer des stratégies de marketing personnalisées.
- Nous comprenons que tous les clients ne se ressemblent pas et n'ont pas les mêmes goûts.
- Cela mène au défi de commercialiser le bon produit au bon client. Une offre ou un produit qui pourrait séduire un segment particulier de clients peut ne pas être utile pour d'autres segments.
- Ainsi, vous pouvez appliquer l'algorithme de clustering K-means pour segmenter votre audience clientèle en groupes ayant des traits et des préférences similaires basés sur diverses métriques (telles que leurs activités, leurs goûts et dégoûts sur les réseaux sociaux et leur historique d'achats).
- En fonction de ces segments de clients identifiés, vous pouvez créer des stratégies de marketing personnalisées et accroître les affaires de votre organisation.

# 2 - Introduction à l'Apprentissage Automatique
- L'apprentissage automatique est l'une des technologies les plus récentes et passionnantes. Vous l'utilisez probablement des dizaines de fois par jour sans même le savoir. 
- L'apprentissage automatique est une forme d'intelligence artificielle qui donne aux ordinateurs la capacité d'apprendre sans être explicitement programmés.
- Il fonctionne sur des modèles d'apprentissage supervisés et non supervisés.
- Contrairement au modèle d'apprentissage supervisé, le modèle non supervisé de l'apprentissage automatique n'a pas de groupes prédéfinis sous lesquels vous pouvez distribuer vos données.
- Vous pouvez trouver ces groupements à travers le clustering.

Je vais expliquer cela plus en détail à travers les exemples suivants.

### Besoin du Clustering avec Exemples
Le clustering permet de regrouper intelligemment les données en fonction de leurs similitudes, facilitant ainsi des analyses ciblées et efficaces.

### Qu'est-ce que le Clustering?
Le clustering est un processus d'organisation des données en groupes où chaque groupe, ou cluster, contient des éléments ayant des caractéristiques communes.

### Types de Clustering
- **Clustering Exclusif** : Chaque donnée appartient à un seul et unique cluster.
- **Clustering Chevauchant** : Les données peuvent appartenir à plusieurs clusters.
- **Clustering Hiérarchique** : Les clusters sont formés en plusieurs étapes, créant une structure arborescente.

### Clustering K-Means
Le clustering K-means est une méthode puissante et populaire dans l'apprentissage non supervisé. Il segmente efficacement les grandes quantités de données en groupes homogènes, ce qui est essentiel pour de nombreuses applications pratiques.

### Mise en Pratique : Implémentation du Clustering K-Means sur un Jeu de Données Cinématographique avec Python
Nous allons appliquer l'algorithme K-means sur un jeu de données de films pour identifier des groupes basés sur leur rentabilité et popularité auprès des spectateurs. Voici un exemple simplifié en Python :

```python
from sklearn.cluster import KMeans
import numpy as np

# Exemple de données : popularité (nombre de likes), budget (en millions)
X = np.array([[100, 20], [150, 30], [130, 25], [170, 45], [80, 20]])

# Application du K-Means
kmeans = KMeans(n_clusters=2)  # Définition de 2 clusters
kmeans.fit(X)

# Centres des clusters et étiquettes
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print("Centroïdes :")
print(cent

roids)
print("Étiquettes :")
print(labels)
```

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/751858c6-f13e-40b7-8991-0441e46ad0d2)


Comme vous pouvez le voir sur cette image (ci-haut), les points de données sont représentés par des points bleus. Ces points de données ne portent pas d'étiquettes qui permettraient de les différencier. Vous ne savez donc rien de précis sur ces données. La question est donc : pouvez-vous identifier une structure quelconque dans ces données ? Ce problème peut être résolu en utilisant la technique de clustering. Le clustering divisera cet ensemble complet de données sous différentes étiquettes (ici appelées clusters) regroupant des points de données similaires dans un seul cluster, comme illustré dans le graphique ci-dessous. Cette méthode est utilisée comme une technique très puissante pour l'analyse descriptive exploratoire.

**Clustering** : Ici (en bas), la technique de clustering a divisé l'ensemble des points de données en deux clusters. Les points de données à l'intérieur d'un cluster sont similaires entre eux mais différents des autres clusters. Par exemple, si vous disposez de données sur les symptômes des patients, vous pouvez maintenant identifier le nom d'une maladie spécifique basée sur ces symptômes.
![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/0272af43-7074-46af-b3c2-be5e7b8dcf0c)

Poursuivons la compréhension du clustering avec l'exemple de Google News.

Ce que fait Google News, c'est qu'il regroupe chaque jour des centaines et des milliers de nouvelles qui apparaissent sur le web en histoires cohérentes. Comment cela fonctionne-t-il ?


## Exemple de Clustering avec Google News

Lorsque vous visitez le site news.google.com, vous êtes confronté à un flux dense d'informations provenant de diverses sources. Pour faciliter la navigation et l'accès à l'information pertinente, Google News utilise des techniques de clustering pour regrouper les nouvelles par thèmes ou sujets similaires. Voici comment cela fonctionne :

### Organisation des Nouvelles en Clusters
Sur l'image que vous avez partagée, nous pouvons voir trois catégories principales où les nouvelles sont regroupées : Canada, Divertissements, et Sports. Chaque catégorie représente un cluster de nouvelles qui partagent des thèmes ou sujets communs.

1. **Canada** : Ce cluster inclut des nouvelles sur des sujets variés concernant le Canada, comme la politique, les conditions météorologiques, et des questions sociales.
   
2. **Divertissements** : Ce cluster rassemble des articles sur des célébrités, des événements culturels, et d'autres nouvelles liées au divertissement.
   
3. **Sports** : Ici, les articles sont principalement axés sur des événements sportifs, des analyses de matches, et des nouvelles concernant les athlètes.

### Mécanisme de Regroupement
- **Extraction de Caractéristiques** : Google extrait des mots-clés et des entités (comme des noms de personnes ou de lieux) des titres et des contenus des articles.
- **Application de l'Algorithme de Clustering** : Les articles sont ensuite analysés pour détecter des similarités dans les caractéristiques extraites. Les articles similaires sont regroupés ensemble.
- **Présentation des Clusters** : Chaque groupe d'articles est présenté sous la forme d'une histoire agrégée qui met en avant un article principal et offre des liens vers d'autres articles relatifs au même sujet.

### Avantages du Clustering
- **Amélioration de l'Expérience Utilisateur** : Les utilisateurs peuvent rapidement trouver des nouvelles qui les intéressent, car les articles similaires sont regroupés ensemble.
- **Découverte Facilitée** : Cette méthode expose les lecteurs à une gamme plus large d'articles sur un même sujet, augmentant ainsi la profondeur de l'information accessible.
- **Efficient pour Suivre des Événements en Évolution** : Les utilisateurs peuvent suivre l'évolution d'un sujet ou d'une histoire grâce à la mise à jour régulière des clusters avec de nouveaux articles.

- Ce système de clustering est un exemple classique de l'utilisation de l'apprentissage non supervisé pour organiser de grands ensembles de données (ici, des articles de nouvelles), ce qui permet de structurer l'information de manière intuitive et accessible.


![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/6dd5db3b-cfcc-439a-b37c-70e06e2d8777)


Une autre application fascinante du clustering se trouve dans le domaine de la génomique. 
![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/ac1dd003-1a46-4445-8f60-090e6190d8cc)

La génomique est l'étude de l'ADN. Comme vous pouvez le voir sur l'image, différentes couleurs telles que le rouge, le vert et le gris illustrent le degré auquel un individu possède ou non un gène spécifique. Ainsi, vous pouvez utiliser un algorithme de clustering sur les données ADN d'un groupe de personnes pour créer différents clusters. Cela peut fournir des insights très précieux sur la santé de certains gènes.
Par exemple, les personnes ayant un génotype Duffy-négatif ont tendance à présenter une plus grande résistance au paludisme et se trouvent généralement dans les régions africaines. Vous pouvez donc établir une relation entre le génotype, l'habitat naturel et découvrir leur réaction à certaines maladies.
En somme, le clustering partitionne l'ensemble de données selon les similarités en différents groupes qui peuvent servir de base pour des analyses plus approfondies. Le résultat est que les objets d'un même groupe seront similaires entre eux, mais différents des objets d'un autre groupe.


## Conclusion
Ce cours offre une introduction complète au monde passionnant du Machine Learning non supervisé et montre comment des techniques telles que le clustering K-means peuvent révolutionner les stratégies de marketing. En segmentant les clients en groupes homogènes, les entreprises améliorent non seulement leur compréhension des besoins des clients mais aussi optimisent l'efficacité de leurs efforts marketing.

Nous espérons que ce cours vous aidera à mieux comprendre le potentiel des analyses de données dans l'élaboration des stratégies de marketing et vous encouragera à explorer ces technologies innovantes. 🌟

Ce README vise à vous guider à travers les concepts fondamentaux du clustering et son application pratique pour améliorer l'engagement et la satisfaction client dans un contexte de marketing basé sur les données. 🚀
