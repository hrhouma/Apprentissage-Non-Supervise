# Partie 1 : Configuration du projet

## Description
Cette partie consiste à configurer le chemin pour pouvoir importer des modules situés dans des répertoires supérieurs. Cette configuration permet une meilleure organisation des fichiers du projet en les répartissant dans plusieurs dossiers. Nous ajoutons le chemin parent au système pour garantir que les modules ou fichiers de niveau supérieur soient accessibles.

## Code

```python
import sys
sys.path.append("..")
```

## Justification
La ligne de code `sys.path.append("..")` est essentielle lorsque vous travaillez dans une structure de projet où des modules ou des fichiers nécessaires sont stockés dans des répertoires parents. En ajoutant le chemin du répertoire parent au système Python, vous pouvez importer et utiliser des modules situés dans des répertoires de niveau supérieur sans avoir à spécifier leur chemin complet.

Cela facilite l'organisation du projet, en particulier si vous avez une structure complexe avec plusieurs répertoires. C'est également utile pour éviter les erreurs d'importation de modules lorsque vous travaillez avec plusieurs fichiers répartis dans différents dossiers.

---
# Annexe : code 
---

L'instruction suivante :

```python
import sys
sys.path.append("..")
```

sert à manipuler la variable `sys.path`, qui est une liste des répertoires où Python cherche les modules lorsqu'on utilise l'instruction `import`. Voyons cela en détail.

### 1. `import sys`
Cette ligne importe le module `sys`, qui permet d'interagir avec certains aspects du système d'exécution Python. Ce module contient diverses fonctions et variables liées à l'environnement et aux processus du système.

### 2. `sys.path`
`sys.path` est une liste de chaînes de caractères qui représente les chemins que Python parcourt pour chercher des modules à importer. Lorsque vous faites un `import nom_du_module`, Python cherche dans chaque répertoire listé dans `sys.path`. Cette liste contient généralement :
- Le répertoire courant (celui d'où le script est exécuté).
- Les répertoires des bibliothèques standards de Python.
- Les répertoires des bibliothèques tierces installées (par exemple, celles installées via `pip`).

### 3. `sys.path.append("..")`
Cette ligne ajoute le répertoire parent (`".."`) à la liste `sys.path`. En d'autres termes, Python cherchera aussi les modules dans le répertoire parent du script en cours d'exécution. Voici les étapes que cela implique :
- Le `..` correspond au répertoire parent du répertoire courant.
- `sys.path.append("..")` modifie la liste des chemins de recherche en y ajoutant le chemin vers le répertoire parent.

Cela est utile si vous voulez importer des modules se trouvant dans un répertoire parent par rapport à celui où votre script est exécuté.

[Retour à la Table des matières](../Tables-des-matieres.md)

