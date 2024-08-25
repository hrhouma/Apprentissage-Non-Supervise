### Évaluation Formative : Comprendre le Choix du Nombre de Neurones et des Hyperparamètres dans un Autoencodeur

Cette évaluation formative est conçue pour évaluer votre compréhension du choix du nombre de neurones et des hyperparamètres dans les autoencodeurs, en particulier dans le contexte des deux implémentations présentées pour le jeu de données MNIST.

---

#### **Question 1 :**
Expliquez pourquoi le modèle dense dans le code 1 commence avec 400 neurones dans la première couche après l'aplatissement des images, même si la dimension d'entrée aplatie est de 784. Quels facteurs peuvent influencer ce choix, et comment cela pourrait-il affecter la performance du modèle ?

#### **Question 2 :**
Le code 1 utilise une réduction progressive du nombre de neurones (400 → 200 → 100 → 50 → 25). Quels sont les avantages de cette stratégie de réduction graduelle du nombre de neurones à chaque couche ? Comment cela aide-t-il à éviter le surapprentissage (overfitting) tout en permettant une compression efficace des données ?

#### **Question 3 :**
Dans le code 2 (Autoencodeur Convolutionnel), les couches de convolution et de pooling ne suivent pas une stratégie de réduction linéaire comme dans le code 1. Expliquez comment le choix du nombre de filtres (par exemple, 16, 8) dans les couches de convolution peut être comparé à la réduction du nombre de neurones dans un modèle dense. Quelles sont les considérations principales pour décider du nombre de filtres dans un autoencodeur convolutionnel ?

#### **Question 4 :**
Imaginez que vous deviez ajuster le nombre de neurones dans un autoencodeur dense pour un jeu de données plus complexe que MNIST, comme des images de haute résolution. Comment décideriez-vous du nombre de neurones pour chaque couche ? Quels tests ou validations mettriez-vous en place pour optimiser cette décision ?

#### **Question 5 :**
Le choix du nombre de neurones dans un autoencodeur est souvent empirique. Proposez une méthode systématique pour tester et valider différents choix de neurones dans les couches denses d'un autoencodeur. Comment utiliseriez-vous les résultats pour affiner l'architecture du modèle ?

---

#### **Question 6 :**
Qu'est-ce qu'un hyperparamètre dans le contexte d'un autoencodeur ? Donnez des exemples d'hyperparamètres présents dans les codes proposés et expliquez leur rôle spécifique dans l'entraînement du modèle.

#### **Question 7 :**
Le code 1 utilise l'optimiseur `SGD` avec un taux d'apprentissage (`learning rate`) de 1.5, tandis que le code 2 utilise l'optimiseur `adam`. Expliquez la différence entre ces deux optimisateurs et discutez de l'impact potentiel du taux d'apprentissage sur les performances d'un autoencodeur. Comment choisiriez-vous un optimiseur approprié pour un autoencodeur ?

#### **Question 8 :**
Le nombre d'époques (`epochs`) est un autre hyperparamètre crucial. Le code 1 utilise 20 époques, tandis que le code 2 en utilise 50. Quels sont les effets potentiels d'un nombre trop élevé ou trop faible d'époques sur l'entraînement d'un autoencodeur ? Comment déterminer le nombre d'époques optimal pour un autoencodeur ?

#### **Question 9 :**
La taille des lots (`batch size`) est également un hyperparamètre important. Bien qu'elle ne soit pas spécifiée explicitement dans le code 1, le code 2 utilise une taille de lot de 256. Expliquez l'importance de ce paramètre et comment il peut affecter l'entraînement du modèle. Comment choisiriez-vous la taille de lot appropriée pour un autoencodeur ?

#### **Question 10 :**
Le choix de la fonction de perte est un hyperparamètre essentiel pour l'entraînement d'un autoencodeur. Dans les deux codes, la fonction de perte utilisée est `binary_crossentropy`. Pourquoi cette fonction de perte est-elle appropriée pour un autoencodeur, et dans quels cas pourrait-on envisager d'utiliser une autre fonction de perte ?

#### **Question 11 :**
En plus des hyperparamètres standards (optimiseur, taux d'apprentissage, époques, taille de lot), discutez de l'importance du choix des couches et de leur nombre de neurones comme hyperparamètre. Comment testeriez-vous et ajusteriez-vous ces choix pour améliorer les performances de votre autoencodeur ?

#### **Question 12 :**
Proposez une approche systématique pour ajuster et optimiser les hyperparamètres dans un autoencodeur. Comment valideriez-vous vos choix d'hyperparamètres pour garantir que votre modèle n'est pas en surapprentissage (overfitting) ou sous-apprentissage (underfitting) ?

#### **Question 13 :**
Expliquez l'importance de la régularisation (comme le dropout ou L2) dans la conception d'un autoencodeur. Quels sont les hyperparamètres associés à la régularisation, et comment influencent-ils la capacité du modèle à généraliser sur de nouvelles données ?

---

Ces questions couvrent à la fois le choix des neurones et des hyperparamètres dans les autoencodeurs, vous permettant d'explorer en profondeur les aspects techniques et pratiques de la conception et de l'optimisation de ces modèles. Répondez de manière détaillée, en vous appuyant sur des exemples concrets, des principes théoriques et des expériences personnelles.
