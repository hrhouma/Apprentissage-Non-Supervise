
# 1 - Création des environnements virtuels
### Méthode 1 : Utiliser `venv` avec PowerShell (Windows)

```powershell
# Naviguez vers le répertoire de votre projet
cd C:\Users\Haythem\Desktop\fastapi\fastapi-the-complete-course\

# Créez un environnement virtuel
python3 -m venv .monenv

# Activez l'environnement virtuel
cd .\.monenv\Scripts\
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\Activate.ps1

# Pour désactiver l'environnement virtuel
deactivate
```

### Méthode 2 : Utiliser `conda` (Windows/Linux)

```bash
# Créez un environnement virtuel
conda create --name monenv python=3.x

# Activez l'environnement virtuel
conda activate monenv

# Pour désactiver l'environnement virtuel
conda deactivate
```

### Méthode 3 : Utiliser `venv` avec Bash (Linux)

```bash
# Naviguez vers le répertoire de votre projet
cd /chemin/vers/votre/projet

# Créez un environnement virtuel
python3 -m venv kafka_env

# Activez l'environnement virtuel
source kafka_env/bin/activate

# Pour désactiver l'environnement virtuel
deactivate
```
---
# 2 - Résumé
### Méthode 1 : Utiliser `venv` avec PowerShell (Windows)

```powershell
cd C:\Users\Haythem\Desktop\fastapi\fastapi-the-complete-course\
python3 -m venv .monenv
cd .\.monenv\Scripts\
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\Activate.ps1
deactivate
```

### Méthode 2 : Utiliser `conda` (Windows/Linux)

```bash
conda create --name monenv python=3.x
conda activate monenv
conda deactivate
```

### Méthode 3 : Utiliser `venv` avec Bash (Linux)

```bash
cd /chemin/vers/votre/projet
python3 -m venv kafka_env
source kafka_env/bin/activate
deactivate
```
---

# 3 - Autres commandes

```bash
# Changer de répertoire
cd C:\Users\Haythem\Desktop\codesPython

# Créer un nouvel environnement Conda
conda create -n test-env

# Lister tous les environnements Conda
conda env list

# Activer un environnement Conda spécifique
conda activate tkinter-env

# Désactiver l'environnement Conda actuellement actif
conda deactivate

# Supprimer un environnement Conda
conda remove --name test-env --all

# Initialiser Conda (si nécessaire)
conda init

# Exemple de gestion d'environnement

# Changer de répertoire
cd C:\Users\Haythem\Desktop\codesPython

# Créer un environnement appelé haythem-env
conda create -n haythem-env

# Lister tous les environnements Conda
conda env list

# Activer l'environnement tkinter-env
conda activate tkinter-env

# Désactiver l'environnement Conda actif
conda deactivate

# Supprimer l'environnement haythem-env
conda remove --name haythem-env --all
```

---

# 4 - Encore d'autres commandes utiles


```bash
# Changer de répertoire
cd C:\Users\Haythem\Desktop\codesPython

# Créer un nouvel environnement Conda
conda create -n test-env

# Lister tous les environnements Conda
conda env list

# Activer un environnement Conda spécifique
conda activate tkinter-env

# Désactiver l'environnement Conda actuellement actif
conda deactivate

# Supprimer un environnement Conda
conda remove --name test-env --all

# Initialiser Conda (si nécessaire)
conda init

# Exemple de gestion d'environnement

# Changer de répertoire
cd C:\Users\Haythem\Desktop\codesPython

# Créer un environnement appelé haythem-env
conda create -n haythem-env

# Lister tous les environnements Conda
conda env list

# Activer l'environnement tkinter-env
conda activate tkinter-env

# Désactiver l'environnement Conda actif
conda deactivate

# Supprimer l'environnement haythem-env
conda remove --name haythem-env --all

# Installer un package spécifique
conda install numpy

# Mettre à jour un package spécifique
conda update numpy

# Mettre à jour Conda
conda update conda

# Lister tous les packages installés dans l'environnement actif
conda list

# Lister les packages installés dans un environnement spécifique
conda list -n haythem-env

# Rechercher un package dans les canaux Conda
conda search numpy

# Exporter les packages installés dans un fichier
conda env export > environment.yml

# Créer un environnement à partir d'un fichier d'exportation
conda env create -f environment.yml

# Nettoyer les fichiers inutilisés
conda clean --all

# Désactiver tous les environnements actifs
conda deactivate

# Revenir à une version antérieure d'un package
conda install numpy=1.18

# Créer un environnement avec une version spécifique de Python
conda create -n py38-env python=3.8

# Cloner un environnement existant
conda create --name clone-env --clone haythem-env

# Afficher les informations sur Conda
conda info

# Afficher les informations sur un environnement spécifique
conda info --envs

# Afficher les informations détaillées sur un package
conda list numpy

# Lister les environnements Conda disponibles
conda env list

# Créer un environnement avec des packages spécifiques
conda create -n data-env numpy pandas matplotlib

# Supprimer tous les packages inutilisés et les anciennes versions
conda clean --all

# Désactiver l'environnement Conda actif
conda deactivate

# Activer un environnement Conda spécifique
conda activate haythem-env

# Désactiver l'environnement Conda actif
conda deactivate
```

---

# 5 - 100 commandes


- Liste de 100 commandes supplémentaires utiles pour la gestion d'environnements virtuels, la manipulation de packages Python, et d'autres tâches courantes en développement :

### Gestion des environnements virtuels

1. `python -m venv env`
2. `python -m venv myenv`
3. `source env/bin/activate`
4. `source myenv/bin/activate`
5. `env\Scripts\activate`
6. `myenv\Scripts\activate`
7. `deactivate`
8. `conda create -n myenv`
9. `conda activate myenv`
10. `conda deactivate`
11. `conda env list`
12. `conda remove --name myenv --all`
13. `virtualenv env`
14. `pipenv --python 3.x`
15. `pipenv shell`
16. `pipenv install`
17. `pipenv install requests`
18. `pipenv uninstall requests`
19. `pipenv lock`
20. `pipenv graph`

### Gestion des packages avec pip

21. `pip install requests`
22. `pip install numpy`
23. `pip install pandas`
24. `pip install flask`
25. `pip install django`
26. `pip install -r requirements.txt`
27. `pip freeze`
28. `pip freeze > requirements.txt`
29. `pip uninstall requests`
30. `pip list`
31. `pip show requests`
32. `pip search requests`
33. `pip install --upgrade pip`
34. `pip install --upgrade requests`
35. `pip check`

### Gestion des packages avec conda

36. `conda install numpy`
37. `conda install pandas`
38. `conda install flask`
39. `conda install django`
40. `conda install -c conda-forge requests`
41. `conda update conda`
42. `conda update numpy`
43. `conda list`
44. `conda list -n myenv`
45. `conda remove numpy`
46. `conda info`
47. `conda info --envs`
48. `conda clean --all`
49. `conda env export > environment.yml`
50. `conda env create -f environment.yml`

### Gestion de versions de Python

51. `pyenv install 3.x.x`
52. `pyenv uninstall 3.x.x`
53. `pyenv versions`
54. `pyenv global 3.x.x`
55. `pyenv local 3.x.x`
56. `pyenv shell 3.x.x`
57. `pyenv virtualenv 3.x.x myenv`
58. `pyenv virtualenvs`
59. `pyenv activate myenv`
60. `pyenv deactivate`

### Gestion des projets avec Git

61. `git init`
62. `git clone <repo-url>`
63. `git add .`
64. `git commit -m "Initial commit"`
65. `git push origin main`
66. `git pull origin main`
67. `git status`
68. `git log`
69. `git branch`
70. `git checkout -b new-branch`
71. `git merge new-branch`
72. `git remote -v`
73. `git fetch`
74. `git rebase main`
75. `git stash`
76. `git stash pop`

### Commandes diverses

77. `python script.py`
78. `python -m unittest`
79. `python -m pytest`
80. `python -m http.server`
81. `python -m cProfile script.py`
82. `jupyter notebook`
83. `jupyter lab`
84. `django-admin startproject mysite`
85. `flask run`
86. `mkdocs serve`
87. `mkdocs build`
88. `aws configure`
89. `aws s3 ls`
90. `aws ec2 describe-instances`
91. `docker build -t myimage .`
92. `docker run -d -p 80:80 myimage`
93. `docker ps`
94. `docker stop <container-id>`
95. `docker rm <container-id>`
96. `docker images`
97. `docker rmi <image-id>`
98. `kubectl get pods`
99. `kubectl describe pod <pod-name>`
100. `kubectl logs <pod-name>`


