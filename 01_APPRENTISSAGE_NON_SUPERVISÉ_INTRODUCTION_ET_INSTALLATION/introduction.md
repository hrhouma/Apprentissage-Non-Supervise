# Aperçu de l'Apprentissage Automatique Non Supervisé

# Introduction

Il y a seulement quelques décennies, l'idée d'enseigner à une machine et de s'attendre à des réponses intelligentes semblait être un rêve lointain. Mais aujourd'hui, grâce à l'essor du machine learning, ce rêve est devenu une réalité. Nous pourrions même dire que les machines sont devenues, à certains égards, plus intelligentes que nous. Cet article explore en profondeur l'univers fascinant de l'apprentissage automatique non supervisé.

# 1 - Contenus

1. **Un Aperçu de l’Apprentissage Automatique**
   - Définition et historique
   - Importance dans le monde moderne

2. **Qu'est-ce que l'Apprentissage Non Supervisé?**
   - Définition
   - Comment cela fonctionne-t-il ?

3. **Importance de l'Apprentissage Non Supervisé**
   - Pourquoi est-il crucial dans les analyses de données modernes?

4. **Types d'Apprentissage Non Supervisé**
   - Clustering
   - Réduction de dimensionnalité

5. **Applications de l'Apprentissage Non Supervisé**
   - Exemples pratiques dans divers secteurs

6. **Comparaison : Apprentissage Supervisé vs Non Supervisé**
   - Différences clés
   - Avantages et utilisations spécifiques

7. **Inconvénients de l’Apprentissage Non Supervisé**
   - Limitations et défis

------------
# 2 - Un aperçu de l'apprentissage automatique

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/bd7bd49a-9f28-42e4-b3d1-034543850794)

L'apprentissage automatique, ou *machine learning*, est un domaine fascinant de l'intelligence artificielle qui vise à doter les machines de la capacité d'apprendre à partir des données sans être explicitement programmées pour chaque tâche. Imaginez que vous enseignez à un enfant comment reconnaître des fruits. Vous lui montrez plusieurs exemples de pommes et de bananes, et au fil du temps, il apprend à les distinguer, même s'il n'a jamais vu certains de ces fruits auparavant. L'apprentissage automatique fonctionne de manière similaire : nous alimentons le système avec des données (les exemples de fruits), il apprend des modèles à partir de ces données, et finalement, il peut faire des prédictions ou prendre des décisions basées sur de nouvelles données qu'il n'a jamais vues.

Le processus commence par la collecte et le nettoyage des données, qui est crucial car la qualité des données affecte directement la performance de l'apprentissage. Ensuite, des algorithmes sont développés et "entraînés" à reconnaître des modèles essentiels à partir des données. Si l'algorithme fonctionne bien et répond aux attentes, alors il est considéré comme réussi. Si non, il faut revoir le modèle, ajuster ou reformuler les données, et essayer à nouveau.

Dans le monde du machine learning, il existe trois types principaux d'apprentissage :

1. **Apprentissage supervisé** : C'est comme un test ouvert où chaque question (donnée d'entrée) a une réponse correcte (étiquette). Cela signifie que chaque exemple dans les données d'entraînement comporte une étiquette, ou un résultat attendu, et l'objectif est de créer un modèle capable de prédire l'étiquette pour de nouvelles entrées. Cela est souvent utilisé pour des tâches comme la classification (par exemple, est-ce un email est un spam ou non) ou la régression (par exemple, prédire le prix d'une maison).

2. **Apprentissage non supervisé** : Ici, les données ne sont pas étiquetées, et le but est de découvrir des structures cachées ou des groupements dans les données. Imaginez que vous disposez d'un grand nombre de journaux de discussion sans aucune indication sur les sujets discutés. L'apprentissage non supervisé peut aider à organiser ces journaux en groupes thématiques sans aucune supervision préalable.

3. **Apprentissage par renforcement** : Dans ce type d'apprentissage, le modèle interagit avec un environnement (par exemple, un jeu ou un simulateur) et reçoit des récompenses ou des pénalités basées sur ses actions. Cela s'apparente à former un animal avec des récompenses : le modèle apprend à effectuer des actions qui maximisent les récompenses au fil du temps, améliorant ainsi son comportement pour atteindre un objectif spécifique.

L'apprentissage automatique ouvre une multitude d'applications et de possibilités, mais il présente aussi des défis, notamment en termes de gestion des données, de conception d'algorithme, et d'évaluation des modèles. Le parcours pour maîtriser ces outils peut être complexe, mais il est essentiel pour exploiter pleinement le potentiel de l'intelligence artificielle.

Dans la suite, nous explorerons en détail l'apprentissage non supervisé, ses applications et ses algorithmes, pour mieux comprendre où et comment cette technologie peut être utilisée efficacement.

------------

# 3 - Qu'est-ce que l'apprentissage non supervisé ?
![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/e3015052-0e92-48f3-ab1f-de7a7f30f201)


L'apprentissage non supervisé est souvent comparé à l'auto-apprentissage, où l'algorithme découvre des structures et des motifs inconnus dans des ensembles de données dépourvus d'étiquettes explicites. Cette forme d'apprentissage est cruciale pour modéliser les distributions de probabilité sous-jacentes, détecter des anomalies, ou découvrir des groupements intéressants dans les données. Pour simplifier, imaginez un étudiant qui possède tous les manuels nécessaires pour étudier mais qui n'a pas de professeur pour le guider. Cet étudiant doit apprendre par lui-même pour réussir ses examens, une démarche similaire à celle que nous utilisons pour enseigner à nos machines via l'apprentissage non supervisé.

## Exemple d'apprentissage non supervisé

![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/426e0612-e83f-4fc9-899a-6d04c58fb45c)

Voici une version corrigée de l'exemple pour l'illustrer correctement avec le soccer et l'apprentissage non supervisé :

Prenons un exemple quotidien pour illustrer ce concept. Imaginez que vous assistiez pour la première fois à un match de soccer, sans aucune connaissance préalable du jeu, et que vous soyez invité par des amis à voir un match entre Dortmund et Real Madrid. 
Au début, vous êtes perdu, ne connaissant rien au soccer. Mais au fur et à mesure que le match avance, vous commencez à observer et à tirer des conclusions :

- Vous remarquez que l'équipe de Real Madrid porte des maillots blancs et Dortmund des jaunes.
- Vous identifiez des joueurs qui défendent leur zone de but (les gardiens) et d'autres qui essaient de marquer (les attaquants).
- Vous comprenez que si la balle passe entre les poteaux du but, cela compte comme un point (un but).

Au fur et à mesure que vous observez ces éléments, vous apprenez les règles du jeu. Ce processus d'apprentissage, où vous tirez des leçons de votre environnement sans instruction directe, est analogue à l'apprentissage non supervisé. Vous n'aviez pas de guide explicite, mais vous avez utilisé les données disponibles (le match en cours et les réactions de vos amis) pour comprendre le soccer par vous-même.

## Pourquoi l'apprentissage non supervisé est-il important ?

L'apprentissage non supervisé est essentiel pour plusieurs raisons :

- **Découverte de modèles cachés** : Les algorithmes non supervisés explorent des ensembles de données non étiquetés pour révéler des structures ou des associations que nous n'aurions pas pu identifier autrement.
- **Catégorisation et association** : Ces modèles peuvent aider à regrouper des éléments similaires ou à découvrir des relations entre différentes données, ce qui est utile dans de nombreux domaines tels que le marketing, la génétique, etc.
- **Détection d'anomalies** : Ils sont également utilisés pour identifier les données qui dévient des normes établies, ce qui est crucial pour la détection de fraude, la surveillance de la santé, etc.
- **Gestion des données non étiquetées** : Comme la plupart des données disponibles ne sont pas étiquetées, l'apprentissage non supervisé facilite leur analyse sans nécessiter un effort considérable pour étiqueter les données au préalable.

## Types d'apprentissage non supervisé

L'apprentissage non supervisé peut être principalement divisé en deux catégories :

1. **Clustering (Regroupement)** : Il s'agit de regrouper un ensemble de données de manière à ce que les données dans le même groupe (appelé cluster) soient plus similaires entre elles qu'avec celles d'autres groupes. Les techniques populaires incluent :
   - **Clustering hiérarchique** : Il crée des clusters en regroupant progressivement les données selon leur similitude.
   - **K-Means** : Il segmente les données en K groupes en minimisant la distance entre les points de données et le centroïde de leur cluster.
   - **K-NN (K-nearest neighbors)** : Bien qu'il soit souvent utilisé en apprentissage supervisé, il peut être adapté pour des usages non supervisés en classant les points basés sur la proximité avec leurs voisins.

2. **Association** : Cette technique identifie des règles ou des modèles intéressants entre les variables dans de grands ensembles de données. Par exemple, dans un contexte de vente au détail, cela pourrait signifier identifier les produits souvent achetés ensemble.

En résumé, l'apprentissage non supervisé est un pilier fondamental du machine learning, permettant aux machines de tirer des leçons des données sans guidage explicite, ouvrant la voie à des découvertes et innovations significatives.
------------

# 4- Le Clustering dans l'apprentissage non supervisé
![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/3050f653-d014-4780-8f0a-0942e0c84100)


Le clustering, ou regroupement, est une technique fondamentale dans l'apprentissage non supervisé où l'objectif est de découvrir des groupes naturels ou des structures cachées dans les données. Les données peuvent être regroupées en fonction de divers attributs comme la forme, la taille, la couleur, etc., permettant ainsi de révéler des informations qui ne sont pas immédiatement évidentes. C'est un peu comme organiser un tiroir encombré : une fois que vous commencez à regrouper des objets similaires ensemble, des patterns commencent à émerger et tout devient plus clair.

#### Types de Clustering

1. **Clustering hiérarchique** :
   Ce type de clustering crée des clusters en regroupant les données étape par étape, sur la base de leur similitude. Imaginez construire un arbre généalogique pour les données où chaque branche représente un cluster. Chaque groupe commence avec un seul élément, et les groupes se combinent avec d'autres qui leur sont similaires jusqu'à ce que tout soit regroupé en une hiérarchie de clusters. Cette méthode est particulièrement utile lorsque la relation entre les groupes est importante et que vous souhaitez observer cette structure de regroupement à différents niveaux de granularité.

2. **Clustering K-Means** :
![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/6e614512-3b44-4b39-87c5-663202671dbb)

   Le K-Means est une méthode plus directive et quantitative. Elle fonctionne en assignant initialement des centres de clusters aléatoires, puis en optimisant ces centres en minimisant la distance entre les points de données et leur centre de cluster le plus proche. Le processus est répété jusqu'à ce que les centres des clusters ne bougent plus beaucoup, ce qui indique que les clusters sont aussi homogènes que possible. Chaque cluster est défini par son centroïde, qui est essentiellement le "cœur" du cluster, et chaque point est assigné au cluster dont le centroïde est le plus proche. Cela crée des partitions claires dans les données, ce qui facilite leur interprétation.

4. **Clustering K-NN (K-nearest neighbors)** :
   Bien que traditionnellement utilisé pour l'apprentissage supervisé, K-NN peut également être adapté pour des usages non supervisés. Contrairement aux autres algorithmes qui apprennent explicitement à partir des données, K-NN est un "apprenant paresseux" qui classe simplement les nouveaux points en fonction de leurs voisins les plus proches. Pour chaque nouveau point, l'algorithme examine les K points les plus proches qu'il a mémorisés et les classe en fonction de leur majorité. Ce processus est intuitif et simple, mais il peut être lent avec de grands ensembles de données car il nécessite de comparer chaque nouveau point à tous les points stockés.

### Pourquoi le clustering est-il utile ?

Le clustering est extrêmement utile dans de nombreux domaines :
- **Marketing** : Comprendre les différents segments de clients pour cibler les campagnes publicitaires de manière plus efficace.
- **Biologie** : Regrouper des organismes ou des gènes avec des fonctions similaires pour aider à cartographier le génome.
- **Bibliothèque de documents** : Organiser des articles ou des livres similaires pour faciliter la recherche et l'accès.
- **Réseaux sociaux** : Détecter des communautés ou des groupes dans les réseaux sociaux pour analyser les patterns d'interaction ou la diffusion d'informations.

En somme, le clustering aide à mettre de l'ordre dans le chaos des données non étiquetées, révélant des structures naturelles et facilitant des insights plus profonds sur les données traitées.
------------
# 5- L'association dans l'apprentissage non supervisé

L'association est une technique puissante dans l'apprentissage non supervisé, qui vise à découvrir des liens ou des dépendances entre différents éléments dans un ensemble de données. Ce processus ressemble à trouver des motifs de comportement ou des habitudes dans un ensemble de données, où la présence d'un élément peut impliquer la présence d'un autre. Cela est particulièrement utile dans des domaines comme le marketing, la gestion de stocks, et l'analyse des transactions, où comprendre ces relations peut mener à des décisions plus éclairées et profitable.
![image](https://github.com/hrhouma/Apprentissage-Non-Supervise/assets/10111526/7a03166e-5ccf-443b-a721-58c82d554c6e)

#### Algorithme Apriori

Un des algorithmes les plus connus pour l'exploration de règles d'association est l'**Algorithme Apriori**. Cet algorithme procède par une exploration en largeur pour calculer la fréquence à laquelle des éléments apparaissent ensemble dans des transactions. Par exemple, en analysant les achats dans une épicerie, Apriori peut identifier que les clients qui achètent du pain sont également susceptibles d'acheter du lait et des œufs. Cela permet aux magasins de planifier des stratégies de marketing ciblées ou de réorganiser leurs produits pour augmenter les ventes. L'algorithme génère des règles qui prédisent la probabilité d'occurrence d'un élément basée sur la présence d'autres éléments, utilisant des métriques telles que la 'support' (la fréquence relative des éléments) et la 'confidence' (la probabilité conditionnelle).

#### Algorithme de croissance FP (FP-Growth)

Un autre algorithme efficace est l'**Algorithme de croissance FP**. Contrairement à l'Apriori, qui peut nécessiter de multiples parcours des données pour calculer la fréquence des ensembles d'éléments, l'algorithme FP-Growth utilise une structure d'arbre pour compresser le dataset, puis extrait les ensembles d'éléments fréquents directement de cette structure. Cela commence par la construction d'un "arbre FP" avec l'élément le plus fréquent à la racine, suivi par les autres éléments en ordre de fréquence décroissante. Cette méthode est généralement plus rapide que l'Apriori car elle réduit le nombre d'itérations nécessaires pour atteindre les seuils de support et ne requiert pas la génération de sous-ensembles candidats qui sont coûteux en calcul.

### Pourquoi l'association est-elle utile ?

Les techniques d'association sont cruciales pour :

- **Analyse de marché** : Découvrir les associations entre produits peut aider à optimiser les promotions croisées et les placements de produits.
- **Sécurité informatique** : Identifier les motifs dans les comportements d'attaques peut aider à prédire et prévenir les incidents de sécurité.
- **Recherche médicale** : Dans le domaine médical, ces techniques peuvent révéler des associations entre des symptômes et des diagnostics, aidant ainsi à prédire les maladies.
- **Recommandations en ligne** : Les sites de e-commerce utilisent les règles d'association pour recommander des produits aux utilisateurs, basés sur les articles fréquemment achetés ensemble.

En somme, comprendre et appliquer les techniques d'association permet aux organisations de déceler des modèles cachés dans les données, offrant ainsi des opportunités pour améliorer l'engagement des clients, optimiser les opérations, et accroître l'efficacité des prises de décisions.

------------

# 6 - Applications de l'apprentissage non supervisé

L'apprentissage non supervisé offre des solutions innovantes à de nombreux problèmes complexes du monde réel en exploitant les données de manière créative et efficace. Voici quelques-unes des façons dont il apporte sa contribution :

1. **Segmentation de la clientèle** : L'apprentissage non supervisé aide à regrouper les clients selon des caractéristiques similaires sans connaissance préalable des groupes. Cela permet aux entreprises de cibler leurs communications marketing de manière plus précise et d'offrir des services personnalisés.

2. **Détection de défauts** : En identifiant les anomalies ou les comportements inhabituels dans les ensembles de données, l'apprentissage non supervisé peut signaler des erreurs ou des défauts qui pourraient autrement passer inaperçus, améliorant ainsi la qualité et la fiabilité des produits.

3. **Cartographie des dépendances** : Il peut révéler des relations complexes entre les variables, aidant à comprendre comment les différentes caractéristiques d'un ensemble de données sont interconnectées.

4. **Nettoyage de données** : En éliminant les fonctionnalités redondantes ou non pertinentes, l'apprentissage non supervisé contribue à simplifier les modèles de machine learning, ce qui améliore leur efficacité et leur performance.

#### Exemples concrets d'utilisation

- **Airbnb** : Utilise l'apprentissage non supervisé pour comprendre les préférences des utilisateurs à partir de leurs recherches et interactions. Cette connaissance permet à Airbnb de recommander des logements et des expériences qui correspondent aux préférences des utilisateurs, améliorant ainsi la satisfaction et l'engagement des clients.

- **Amazon** : Amazon emploie des techniques d'apprentissage non supervisé pour analyser les habitudes d'achat des clients et recommander des produits qui sont souvent achetés ensemble. Cela augmente non seulement les ventes incitatives mais améliore également l'expérience d'achat en rendant les suggestions plus pertinentes.

- **Détection de fraude par carte de crédit** : Les algorithmes d'apprentissage non supervisé analysent les modèles de transactions pour identifier les comportements inhabituels. Si une transaction ne correspond pas au modèle habituel d'un utilisateur, elle peut être marquée pour une enquête plus approfondie, réduisant ainsi les risques de fraude et protégeant à la fois les consommateurs et les institutions financières.

Ces applications montrent comment l'apprentissage non supervisé peut non seulement simplifier les processus mais aussi ouvrir de nouvelles voies pour la personnalisation des services et l'amélioration de la sécurité. En dévoilant les structures cachées et les relations dans les données, il permet aux entreprises de prendre des décisions plus informées et d'offrir des expériences utilisateur enrichies.

------------

# 7 - Apprentissage supervisé vs apprentissage non supervisé

L'apprentissage supervisé et non supervisé sont deux approches fondamentales en machine learning, chacune avec ses propres méthodes, applications et défis. Voici un tableau comparatif qui résume les principales différences entre ces deux types d'apprentissage :

| **Paramètre**               | **Apprentissage supervisé**                           | **Apprentissage non supervisé**                 |
|-----------------------------|------------------------------------------------------|-------------------------------------------------|
| **Base de données**         | Ensemble de données étiqueté                          | Ensemble de données sans étiquette              |
| **Méthode d'apprentissage** | Apprentissage guidé                                   | L'algorithme apprend tout seul                  |
| **Complexité**              | Méthode plus simple                                   | Complexe informatiquement                       |
| **Précision**               | Plus précise                                          | Moins précis                                    |

### Inconvénients de l’apprentissage non supervisé

Malgré son utilité dans divers scénarios d'application, l'apprentissage non supervisé présente plusieurs inconvénients significatifs :

1. **Manque de supervision** : Sans données étiquetées, il est difficile de guider l'algorithme sur ce qui est correct ou incorrect, ce qui peut conduire à des interprétations erronées ou des groupements non pertinents.

2. **Précision inférieure** : Comme l'algorithme doit faire des suppositions basées uniquement sur les structures inhérentes aux données, les résultats peuvent être moins précis comparés à l'apprentissage supervisé où les modèles sont entraînés avec des données étiquetées spécifiques.

3. **Correspondance des résultats à des classes de sortie** : Il peut être difficile de mapper les résultats de l'apprentissage non supervisé aux catégories ou étiquettes existantes, car l'algorithme peut découvrir des groupements qui ne correspondent pas aux divisions préétablies.

4. **Interprétation des données** : L'utilisateur doit souvent intervenir pour interpréter et valider les groupements ou les motifs découverts, ce qui nécessite une compréhension approfondie à la fois des données et des objectifs de l'analyse.

### Résumé de l'article

- Ce travail a exploré le vaste domaine du machine learning, en mettant un accent particulier sur l'apprentissage non supervisé. 
- Nous avons commencé par définir ce qu'est le machine learning et ses différents types, avant de plonger dans les détails de l'apprentissage non supervisé, ses applications, et ses avantages pour comprendre des données complexes sans étiquettes pré-définies.
- Les discussions ont inclus des explications sur le clustering et l'association mining, deux techniques clés de l'apprentissage non supervisé, ainsi que des exemples concrets d'application comme Airbnb et Amazon.
- Nous avons également comparé les approches supervisées et non supervisées, et examiné les défis spécifiques associés à l'apprentissage non supervisé.
- Ce parcours a permis de révéler comment, malgré certains défis, l'apprentissage non supervisé reste un outil puissant pour découvrir des insights cachés dans les données non étiquetées.

------------









## Conclusion

L'apprentissage non supervisé offre des possibilités incroyables, mais aussi des défis uniques. En comprendre les subtilités nous permet de mieux exploiter son potentiel et d'anticiper ses impacts futurs.
