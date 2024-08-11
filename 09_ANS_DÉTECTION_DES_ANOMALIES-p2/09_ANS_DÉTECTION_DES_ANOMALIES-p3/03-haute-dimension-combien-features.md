# Combien de features sont nécessaires pour dire qu'on est en haute dimension ?

- Le terme "haute dimension" est relatif et peut varier selon le contexte et l'application. Cependant, en général, on parle de "haute dimension" dans les cas suivants :

### Définition de "Haute Dimension"

1. **Règle Empirique :**
   - Si le nombre de features (dimensions) est supérieur à 10, on commence souvent à parler de données de haute dimension. 
   - Lorsque le nombre de dimensions atteint ou dépasse 100, les données sont presque toujours considérées comme de haute dimension.

2. **Ratio Nombres de Données/Features :**
   - Même si le nombre de features n'est pas extrêmement élevé, si le nombre de données est comparable au nombre de features (par exemple, autant de samples que de features ou moins), cela peut créer des défis similaires à ceux des données de haute dimension.

3. **Problèmes Spécifiques à la Haute Dimension :**
   - **Curse of Dimensionality :** La distance entre les points de données devient moins significative à mesure que le nombre de dimensions augmente, rendant les modèles de clustering et d'anomalie moins efficaces.
   - **Overfitting :** Les modèles ont tendance à surajuster les données d'entraînement en raison du grand nombre de dimensions par rapport au nombre de samples.
   - **Complexité Computationnelle :** Les calculs deviennent de plus en plus coûteux à mesure que le nombre de dimensions augmente.

### Implications Pratiques

- **Moins de 10 Features :** Généralement considéré comme faible à moyenne dimension. Les techniques classiques d'analyse de données fonctionnent bien.
- **10 à 100 Features :** Transition vers des données de haute dimension. Certaines méthodes peuvent commencer à éprouver des difficultés, et les techniques de réduction de dimensionnalité (comme PCA) peuvent être utiles.
- **Plus de 100 Features :** Typiquement considéré comme haute dimension. Les défis de la haute dimensionnalité sont significatifs, et des méthodes spécifiques pour traiter ces problèmes deviennent nécessaires.

### Exemple de Contexte d'Application

- **Traitement d'Image :** Chaque pixel peut être considéré comme une dimension. Les images de haute résolution (par exemple, 64x64 pixels ou plus) ont souvent des milliers de dimensions.
- **Text Mining :** Les techniques de représentation de texte comme TF-IDF ou Word Embeddings peuvent facilement produire des vecteurs avec des centaines voire des milliers de dimensions.

En résumé, bien que le seuil exact puisse varier, on considère généralement que des données avec plus de 10-15 dimensions commencent à entrer dans le domaine de la haute dimensionnalité, et les défis associés augmentent significativement au-delà de 100 dimensions.
