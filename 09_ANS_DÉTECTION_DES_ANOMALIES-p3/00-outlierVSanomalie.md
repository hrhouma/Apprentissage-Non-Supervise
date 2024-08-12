----
# 1 (DÉFINITION1) - Outlier vs anomalie
----

- Un **outlier** est une donnée qui se trouve à l'extérieur de la majorité des autres points de données, souvent loin de la moyenne ou de la médiane.
- Une **anomalie**, c'est un outlier qui a une signification particulière, généralement indiquant un comportement ou un événement inhabituel qui nécessite une attention particulière.

![image](https://github.com/user-attachments/assets/89243cab-5410-429c-9312-3d96ff57df02)

- Par exemple, dans la détection de fraude, une anomalie pourrait être une transaction financière inhabituelle qui pourrait indiquer une activité frauduleuse.

# En résumé : Tous les outliers ne sont pas des anomalies, mais toutes les anomalies sont des outliers.



----
# 2 (DÉFINITION2) - *anomalie* et un *outlier*
----

La différence entre une *anomalie* et un *outlier* (ou valeur aberrante) réside principalement dans le contexte d'utilisation et l'interprétation des données.

## Différences entre anomalie et outlier

- **Outlier (Valeur aberrante)** :
  - Un *outlier* est un point de données qui se situe à une distance significative de la majorité des autres points de données dans un ensemble donné. Il est souvent utilisé dans le contexte de la modélisation statistique pour indiquer que le modèle ne décrit pas correctement les données. Les *outliers* peuvent être des valeurs extrêmes mais ne sont pas nécessairement des erreurs ou des anomalies[1][2].
  
- **Anomalie** :
  - Une *anomalie* est une observation qui dévie de manière significative des attentes basées sur le comportement normal des données. Dans le contexte de la détection d'anomalies, on recherche des comportements inhabituels qui peuvent être alarmants ou significatifs, comme une fraude ou une défaillance de système. Les anomalies peuvent inclure des *outliers*, mais elles peuvent aussi être des comportements qui ne sont pas simplement des valeurs extrêmes mais qui sont significatifs dans un contexte donné[1][2].

## Imane Khelif : Outlier ou Anomalie ?

En ce qui concerne la boxeuse Imane Khelif, elle a été au centre de controverses liées à des tests de genre et à des allégations concernant son éligibilité à concourir dans la catégorie féminine. Ces controverses ont été qualifiées d'« anomalies » par certains, en raison de la nature inattendue et controversée des résultats des tests de genre effectués par l'Association Internationale de Boxe (IBA)[3][6].

Dans ce contexte, le terme *anomalie* est utilisé pour décrire une situation qui dévie des attentes ou des normes établies, plutôt qu'un simple *outlier* statistique. Les tests sanguins ont révélé des « anomalies » qui ont conduit à des discussions sur son genre, ce qui est différent de la simple identification d'une valeur aberrante dans un ensemble de données[6].

Citations:
[1] https://stats.stackexchange.com/questions/189664/outlier-vs-anomaly-in-machine-learning
[2] https://community.deeplearning.ai/t/difference-between-outlier-and-anomaly/281064
[3] https://www.ladepeche.fr/2024/08/06/jo-2024-hyperandrogenie-les-tests-sanguins-montrent-que-ces-deux-boxeuses-sont-des-hommes-la-declaration-choc-de-la-federation-internationale-12126793.php
[4] https://en.wikipedia.org/wiki/Imane_Khelif
[5] https://apnews.com/article/olympics-2024-imane-khelif-boxing-paris-2eb07d442ffb29a61e09911884dcdaa9



----
# 3 (DÉFINITION3) - *anomalie* et un *outlier*
----

- La différence entre une anomalie et un outlier réside principalement dans le *contexte* et la signification des deux termes.

1. **Outlier (Valeur Aberrante) :**
   - Un outlier est une observation ou une donnée qui se situe à une distance considérable par rapport aux autres points de données dans un ensemble donné. 
   - Il peut s'agir d'une donnée extrêmement élevée ou basse par rapport à la distribution générale des données.
   - Les outliers peuvent être le résultat de diverses raisons telles que des erreurs de mesure, des variations naturelles dans les données, ou des événements rares.
   - Dans certains cas, un outlier peut être légitime et représenter une partie normale de la variabilité des données, même s'il est extrême.

2. **Anomalie :**
   - Une anomalie, en revanche, est une donnée qui ne correspond pas au modèle attendu ou à la norme dans un contexte spécifique. 
   - Les anomalies sont souvent considérées comme des indicateurs d'un problème ou d'un événement inhabituel, comme une fraude, une défaillance système, ou un comportement inattendu.
   - Toutes les anomalies peuvent être des outliers, mais toutes les outliers ne sont pas nécessairement des anomalies. Par exemple, dans un ensemble de données, un outlier peut être une valeur légitime et ne pas être considéré comme une anomalie.

**En résumé** : Un outlier est simplement une valeur qui se distingue par son écart par rapport aux autres, tandis qu'une anomalie est une valeur qui signale quelque chose d'inhabituel ou de problématique dans un *contexte* particulier.



----
# 4 (DÉFINITION3) - *Exemples*
----

## 4.1 - Exemple d'une classe d'éléves : 

- Imagine que tu as une classe d'élèves. La plupart des enfants apprennent à lire à peu près au même âge, disons entre 5 et 7 ans. Mais il y a toujours des exceptions. Par exemple, si un enfant apprend à lire à 3 ans, c'est un **outlier** — il est en dehors de la norme, mais cela ne signifie pas forcément qu'il y a quelque chose d'inhabituel. Peut-être qu'il a juste eu beaucoup de stimulation précoce.
- Maintenant, pense à un enfant autiste qui, à 5 ans, connaît déjà tous les systèmes planétaires et les lois de la physique. C'est plus qu'un outlier; c'est une **anomalie** parce que son comportement est si exceptionnel qu'il attire l'attention et pourrait indiquer quelque chose d'unique qui mérite d'être étudié plus en profondeur.

## 4.2 - Exemple de Cristiano Ronaldo et Lionel Messi

Pour mieux comprendre la distinction entre *outliers* et *anomalies* dans le contexte du football, examinons les exemples de Cristiano Ronaldo, Pelé, Neymar, Lionel Messi, et Ronaldo Rozário.

#### Outliers : Cristiano Ronaldo, Pelé, Neymar

- **Cristiano Ronaldo, Pelé, Neymar** : Ces joueurs peuvent être considérés comme des *outliers* dans le monde du football en raison de leurs performances exceptionnelles qui les placent bien au-dessus de la norme. Par exemple, Neymar a surpassé le record de buts de Pelé pour le Brésil, ce qui le distingue comme un joueur exceptionnel même parmi les meilleurs[3]. Cristiano Ronaldo et Pelé sont souvent cités parmi les plus grands joueurs de tous les temps en raison de leurs statistiques impressionnantes et de leur impact sur le jeu[7]. Ces joueurs se démarquent par leurs réalisations et leur talent, les plaçant en dehors de la distribution normale des performances des joueurs de football.

#### Anomalies : Lionel Messi et Ronaldo Rozário

- **Lionel Messi et Ronaldo Rozário** : Dans ce contexte, une *anomalie* pourrait se référer à des joueurs qui présentent des caractéristiques ou des talents qui ne sont pas seulement exceptionnels, mais qui défient les attentes ou les normes établies.
- Lionel Messi, par exemple, est souvent considéré comme une anomalie qui viens d'une autre planète en raison de sa capacité unique à dribbler et à créer des occasions de but d'une manière qui semble défier la logique du jeu[4].
- Ronaldo Rozário, bien que moins connu, pourrait être considéré comme une anomalie s'il présente des compétences ou des caractéristiques qui ne correspondent pas aux attentes habituelles pour un joueur de son profil.
- En résumé, les *outliers* sont des joueurs dont les performances sont exceptionnellement élevées par rapport à la norme, tandis que les *anomalies* sont des joueurs qui présentent des caractéristiques ou des talents qui défient les attentes ou les normes établies dans le sport.




----
# 5  Exercice - Exemples de l'actualité
----

# 5.1 - Outlier ou Anomalie ?

![image](https://github.com/user-attachments/assets/23e66374-d409-468a-9df1-b2b93e842edc)

- source : https://ca.news.yahoo.com/olympic-shooters-become-viral-sensations-as-they-secure-silver-medals-in-paris-did-turkey-send-a-hitman-to-main-character-energy-145517201.html?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAB7JxtnQY1fv5xhoh1w5P5WGxH0GiZ6Rv2mZ0C_hZOxEImFIkQAXXD5bOsZMqOBM12aOgpgnvAv0hHaQr8S4T7-7dGle6vpKMAP-H34bGZcMXdGDaLdMUQVyEogKUIYWUFr7a-iJ7tUPs7xZqvKYJW0m3QDZm41Q-FYwuPImU2mr

# 5.2 - Outlier ou Anomalie ?

![image](https://github.com/user-attachments/assets/76b2a214-37ad-46a0-be82-1d9d4eb5bb4b)

- source : https://www.transfermarkt.com/lionel-messi-vs-cristiano-ronaldo-who-had-highest-market-value-across-20-year-careers-/view/news/433256




----------
# Résumé :
----------



**En résumé** : 
- **Outlier** : Si on parle de ses performances en tant qu'individu exceptionnel qui dépasse la norme générale des athlètes.
- **Anomalie** : Si on se réfère à elle comme un cas rare et exceptionnel dans un contexte particulier, où sa performance pourrait être perçue comme inattendue ou rare.

Je répète : un **outlier** est quelqu'un ou quelque chose qui est en dehors de la norme, tandis qu'une **anomalie** est un outlier qui est tellement exceptionnel qu'il mérite une attention spéciale.


