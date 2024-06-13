# 1 - Activation de l'Environnement Virtuel pour le Projet dans WINDOWS

- Ce guide vous montre comment activer l'environnement virtuel Python pour le développement de ce projet sous Windows. 
- Les instructions sont fournies pour l'utilisation dans la ligne de commande Windows ainsi que dans Visual Studio Code.

## Prérequis

- Assurez-vous d'avoir Python et Visual Studio Code installés sur votre machine. 
- Si ce n'est pas le cas, installez-les à partir de leurs sites officiels ou à partie de Windows Store (Python) :

- Python: [python.org](https://www.python.org/downloads/)
- Visual Studio Code: [code.visualstudio.com](https://code.visualstudio.com/)

## Activation dans Windows

### 1- Installez ANACONDA :  https://www.anaconda.com/anaconda-navigator 
### 2 - Allez dans Environnements
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/b42ec3ef-611a-4533-8c1f-9e6e1b782d1f)
### 3 - Créez un nouvel environnement tkinter-env (choisir votre version de python pour cet environnement python 3.9 par exemple)
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/b3139f27-4e4c-4082-a993-06a9abd4c9e5)
### 4 - Lancez l'environnement (méthode 1 via le navigateur anaconda) : 
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/9743bde2-69ee-4af2-908f-bb65a3e0a1f4)
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/f42f6253-14cc-47b4-a9c0-22ee806b8b28) 

## 4 - Lancez l'environnement (méthode 2 via la ligne de commande) : 
## 4-1. Assurez vous d'avoir conda ajoutée dans vos variables d'environnements (localisez anaconda3 d'abord, dans mon cas : C:\ProgramData\anaconda3)
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/480af092-c1c7-4bad-9d9e-f61e46b8f133)
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/0751f0f3-c5d1-4e12-85b7-ee37d070c031)
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/015cd000-c45c-46de-916d-8606a2d61bdc)
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/3ea82f71-9da4-46b7-9553-2d141b892705)

## 4-2. Excécutez les commandes suivantes : 
```powershell
cd C:\Users\Haythem\Desktop\codesPython
conda activate tkinter-env
deactivate ou conda deactivate
```
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/a15dd895-0350-46d0-8f36-84307af33217)


## 4-3. Autres commandes : 


- Par exemple, pour supprimer un environnement Conda, vous devez utiliser la commande `conda remove --name nom_env --all`. 
- Voici comment vous pouvez procéder étape par étape, en incluant votre demande initiale :

1. **Créer un environnement Conda** (dans votre cas, vous avez mentionné la création d'un environnement appelé `test-env`) :
   ```bash
   conda create -n test-env
   ```

2. **Lister tous les environnements Conda** pour vérifier que votre environnement a bien été créé :
   ```bash
   conda env list
   ```

3. **Activer un environnement Conda** spécifique (dans votre cas `tkinter-env`), pour travailler avec :
   ```bash
   conda activate tkinter-env
   ```

4. **Désactiver l'environnement Conda actuellement actif** :
   ```bash
   conda deactivate
   ```

5. **Supprimer un environnement Conda**. Si vous souhaitez supprimer l'environnement `test-env` que vous avez créé précédemment, utilisez :
   ```bash
   conda remove --name test-env --all
   ```
6. **Autre commande utile pour initier un environnement**.
   ```bash
   conda init
   ```

- Assurez-vous d'être dans le bon dossier ou de spécifier le chemin complet si nécessaire. 
- Si vous travaillez souvent avec des scripts ou des commandes dans des dossiers spécifiques, vous pouvez vous déplacer dans le dossier approprié en utilisant `cd` (Change Directory) dans votre terminal ou invite de commande. Dans notre cas pour aller sur le bureau dans le dossier codesPython :
```bash
cd C:\Users\Haythem\Desktop\codesPython
```

Après vous être déplacé dans le dossier, vous pouvez exécuter les commandes Conda comme décrit ci-dessus.

```powershell
conda create -n haythem-env
```
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/ecc998fc-e9a2-4426-a44e-3cbff22a58e5)
```powershell
conda env list
```
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/ca4d83c8-bbdc-4808-abbd-3e00b4eb280b)
```powershell
conda activate tkinter-env
```
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/f26cef9f-8e00-4819-8445-94ae3bc149b8)
```powershell
 conda deactivate
```
 ![image](https://github.com/hrhouma/YOLO-2/assets/10111526/f33eeb76-2799-4e68-a4c1-643a7310972b)
 ```powershell
conda remove --name haythem-env --all
```
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/253eede6-fcaf-41c8-98a3-29b28d6da8d7)



# 2 - Activation de l'Environnement Virtuel pour le Projet dans WINDOWS

Pour créer un environnement virtuel dans Visual Studio Code (VSCode) pour un projet Python sur Windows et activer cet environnement, suivez les étapes détaillées ci-dessous. Ce tutoriel vous guidera depuis la préparation initiale jusqu'à l'activation de l'environnement virtuel.

## Prérequis
1. Assurez-vous d'avoir installé Python sur votre ordinateur. Vous pouvez le télécharger depuis le [site officiel de Python](https://www.python.org/downloads/).
2. Installez Visual Studio Code si ce n'est pas déjà fait. Téléchargez-le depuis le [site officiel de Visual Studio Code](https://code.visualstudio.com/).

## Étape 1: Configurer VSCode pour l'utilisation de Python
1. Ouvrez VSCode.
2. Installez l'extension Python pour Visual Studio Code depuis le Marketplace. Allez dans l'onglet Extensions (icône des blocs sur la barre latérale), cherchez "Python", puis cliquez sur "Installer".

## Étape 2: Création et Activation d'un Environnement Virtuel
### Création de l'environnement virtuel
1. Ouvrez un terminal dans VSCode en cliquant sur `Terminal > Nouveau terminal` dans la barre de menu.
2. Naviguez jusqu'au dossier où vous souhaitez créer le projet. Par exemple:
   ```powershell
   cd C:\Users\Haythem\Desktop\fastapi\fastapi-the-complete-course\
   ```
3. Créez un environnement virtuel en exécutant:
   ```powershell
   python3 -m venv .monenv
   ```
   ![image](https://github.com/hrhouma/YOLO-2/assets/10111526/2ca51975-248c-4a4f-99c3-0bde024f5408)


### Activation de l'environnement virtuel
4. Changez le répertoire pour entrer dans le dossier de l'environnement virtuel, puis dans le dossier `Scripts` :
   ```powershell
   cd .monenv\Scripts\
   ```
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/4db2c5b5-9784-4afe-8c9f-8eddd86e0a2e)
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/ae9e6576-31c0-454f-b3ea-9a3cf00e836b)
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/da664853-26d3-4669-8791-947c1da64f14)


6. Pour éviter les problèmes de politique d'exécution des scripts, ajustez la politique d'exécution:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   ![image](https://github.com/hrhouma/YOLO-2/assets/10111526/74d03bfa-d010-4bb4-82d8-24847fabf3e9)

7. Activez l'environnement virtuel:
   ```powershell
   .\Activate.ps1
   ```
   Vous devriez voir le nom de votre environnement virtuel (`(.monenv)`) devant le prompt dans le terminal, indiquant que l'environnement est activé.

![image](https://github.com/hrhouma/YOLO-2/assets/10111526/ad8e9249-6ebe-4924-b5ed-b96a1bc32221)

### Exécution d'un script Python
7. **Retournez au répertoire principal** de votre projet et exécutez votre script Python:
   ```powershell
   cd ..
   python interface-tkinter.py
   ```
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/ca62e85e-16ba-482a-927a-264d86883df5)
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/7e339d16-2e0d-4a35-83b2-fc619832ac5c)
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/68bc1f11-b2c7-4197-aa1e-511f8d938ddf)
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/0b6b7bd2-17d1-458c-b93e-b5f3d1cee859)


### Désactivation de l'environnement virtuel
8. Pour désactiver l'environnement virtuel, tapez:
   ```powershell
   deactivate
   ```
![image](https://github.com/hrhouma/YOLO-2/assets/10111526/889af472-c38b-4377-89fa-96c5a0badbf4)

En suivant ces étapes, vous aurez correctement configuré et utilisé un environnement virtuel dans Visual Studio Code sur Windows, permettant un développement Python isolé et propre.

## Résumé des commandes
```powershell
cd C:\Users\Haythem\Desktop\fastapi\fastapi-the-complete-course\
python3 -m venv .monenv
cd .\.monenv\
cd .\Scripts\
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\Activate.ps1
Exécutez le code en cliquant sur le bouton run en haut ou en exécutant python interface-tkinter.py
deactivate
```

# Références utiles :
- https://stackoverflow.com/questions/53995171/anaconda-conda-error-argument-command-invalid-choice-when-trying-to-update-pa
- https://code.visualstudio.com/docs/python/environments
- https://www.geeksforgeeks.org/create-virtual-environment-using-venv-python/
- https://medium.com/@dipan.saha/managing-git-repositories-with-vscode-setting-up-a-virtual-environment-62980b9e8106
