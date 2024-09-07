# RÉFÉRENCE: 
- https://setosa.io/ev/image-kernels/
  
### Introduction aux Noyaux d'Image

Un noyau d'image est une petite matrice utilisée pour appliquer des effets tels que le flou, l'accentuation, le contour ou l'embossage sur des images. Ils sont également utilisés en apprentissage automatique pour l'extraction de caractéristiques, une technique permettant de déterminer les parties les plus importantes d'une image. Dans ce contexte, le processus est généralement appelé "convolution" (voir : réseaux de neurones convolutionnels).

### Exemple Pratique avec un Noyau 3x3

#### Description de l'Image d'Entrée

Nous allons commencer par examiner une image en noir et blanc. La matrice à gauche contient des nombres, entre 0 et 255, qui correspondent chacun à la luminosité d'un pixel dans une image d'un visage. L'image large et granuleuse a été agrandie pour la rendre plus facile à voir.

#### Application d'un Noyau d'Accentuation 3x3

Nous allons appliquer le noyau d'accentuation suivant à l'image du visage :

```
[
  [ 0, -1,  0],
  [-1,  5, -1],
  [ 0, -1,  0]
]
```

#### Calcul de la Convolution

Pour chaque bloc de pixels 3x3 dans l'image de gauche, nous multiplions chaque pixel par l'entrée correspondante du noyau, puis nous faisons la somme. Cette somme devient un nouveau pixel dans l'image de droite.

### Gestion des Bords de l'Image

Une subtilité de ce processus est ce qu'il faut faire le long des bords de l'image. Par exemple, le coin supérieur gauche de l'image d'entrée n'a que trois voisins. Une façon de corriger cela est d'étendre les valeurs des bords dans l'image originale tout en conservant la même taille de notre nouvelle image. Dans cette démonstration, nous avons plutôt ignoré ces valeurs en les rendant noires.

### Zone Interactive

Voici un espace interactif où vous pouvez sélectionner différentes matrices de noyaux et voir comment elles affectent l'image originale ou créer votre propre noyau. Vous pouvez également télécharger votre propre image ou utiliser la vidéo en direct si votre navigateur le prend en charge.

### Conclusion

Le noyau d'accentuation met en évidence les différences de valeurs de pixels adjacents. Cela rend l'image plus vive.

Pour en savoir plus, consultez l'excellente documentation de Gimp sur l'utilisation des noyaux d'image. Vous pouvez également appliquer vos propres filtres personnalisés dans Photoshop en allant dans Filtre -> Autre -> Personnalisé...

Pour plus d'explications, visitez la page d'accueil du projet Explained Visually.


