# La standardisation et la normalisation
La standardisation et la normalisation sont des techniques utilisées pour transformer des données, mais elles ont des objectifs et des méthodes différents.

1. **Standardisation** :
   - **Objectif** : Centrer les données autour de la moyenne avec une échelle de déviation standard.
   - **Méthode** : Soustraire la moyenne et diviser par l'écart-type.
   - **Exemple** : Si les notes des élèves varient de 50 à 100, après standardisation, elles auront une moyenne de 0 et un écart-type de 1.

2. **Normalisation** :
   - **Objectif** : Redimensionner les données pour qu'elles se situent dans une plage spécifique (souvent 0 à 1).
   - **Méthode** : Soustraire la valeur minimale et diviser par la différence entre la valeur maximale et la valeur minimale.
   - **Exemple** : Si les notes des élèves varient de 50 à 100, après normalisation, elles varieront de 0 à 1.

### Exemple pratique

- **Données initiales** : [50, 60, 70, 80, 90, 100]
- **Standardisation** :
  - Moyenne = 75, Écart-type = 17.08
  - [ (50-75)/17.08, (60-75)/17.08, ... ] = [-1.46, -0.88, -0.29, 0.29, 0.88, 1.46]
- **Normalisation** :
  - Min = 50, Max = 100
  - [ (50-50)/(100-50), (60-50)/(100-50), ... ] = [0, 0.2, 0.4, 0.6, 0.8, 1]

En résumé, la standardisation ajuste les données en fonction de leur moyenne et écart-type, tandis que la normalisation ajuste les données pour qu'elles s'inscrivent dans une plage donnée.
