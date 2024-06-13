# Installation des prérequis pour le support GPU dans TensorFlow 2.1

Pour utiliser TensorFlow 2.1 avec le support GPU, suivez les étapes ci-dessous pour configurer votre environnement.

## Prérequis

1. **Installation des pilotes GPU** :
   - Assurez-vous d'installer les derniers pilotes GPU disponibles pour votre carte graphique.

2. **Installation de CUDA 10.1** :
   - Téléchargez et installez CUDA 10.1 depuis le site de NVIDIA.
   - Si lors de l'installation de CUDA, un message indique que "vous installez une version antérieure du pilote", vous pouvez opter pour une installation personnalisée et désélectionner certains composants. Notez bien que TensorFlow ne requiert pas les logiciels additionnels tels que GeForce Experience, PhysX, le pilote d'affichage et l'intégration de Visual Studio.

3. **Installation de cuDNN** :
   - Téléchargez cuDNN v7.6.4.38 pour CUDA 10.1. Ceci nécessite une inscription au NVIDIA Developer Program.
   - Décompressez le fichier dans un emplacement approprié et ajoutez les répertoires `bin`, `incluide` et `lib\x64` à votre variable d'environnement `PATH`.


4. **Création d'un environnement Anaconda** :
   - Ouvrez Anaconda Prompt et créez un nouvel environnement avec Python 3.6 :
     ```
     conda create -n tf_gpu python=3.6
     ```
   - Activez l'environnement créé :
     ```
     conda activate tf_gpu
     ```

5. **Installation de TensorFlow GPU** :
   - Dans l'environnement actif, installez TensorFlow version GPU :
     ```
     conda install tensorflow-gpu
     ```

## Tests de vérification

Pour vérifier si TensorFlow a été correctement installé et si le GPU est utilisable, exécutez les commandes suivantes dans votre Anaconda Prompt (assurez-vous que l'environnement `tf_gpu` est activé) :

1. **Vérifier la version de TensorFlow** :
   ```
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

2. **Vérifier la disponibilité du GPU** :
   ```
   python -c "import tensorflow as tf; print('GPU available:' if tf.test.is_gpu_available() else 'GPU not available')"
   ```

3. **Nombre de GPUs disponibles** :
   ```
   python -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))"
   ```

4. **Exécution d'une opération matricielle sur le GPU** :
   - Copiez et exécutez ce script pour vérifier si le GPU est utilisé pour les calculs :
     ```python
     import tensorflow as tf
     with tf.device('/GPU:0'):
         a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
         b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
         c = tf.matmul(a, b)
     print(c)
     ```

5. **Gestion de la croissance de la mémoire GPU** :
   - Ce script configure la croissance de la mémoire afin d'éviter la consommation de toute la mémoire GPU disponible immédiatement :
     ```python
     import tensorflow as tf
     gpus = tf.config.list_physical_devices('GPU')
     if gpus:
         try:
             for gpu in gpus:
                 tf.config.experimental.set_memory_growth(gpu, True)
             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
             print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
         except RuntimeError as e:
             print(e)
     ```

## Support et documentation

Pour toute question ou problème lors de l'installation, vous pouvez consulter la documentation officielle de TensorFlow ou le site de NVIDIA pour plus d'informations sur l'utilisation des GPUs.
