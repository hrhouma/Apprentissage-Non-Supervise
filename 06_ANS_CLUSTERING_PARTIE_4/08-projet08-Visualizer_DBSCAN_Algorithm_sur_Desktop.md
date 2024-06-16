## DBSCAN Visualizer

- Ce projet utilise Tkinter pour créer une interface graphique permettant de visualiser l'algorithme DBSCAN en temps réel. Les points sont générés, normalisés, puis l'algorithme DBSCAN est appliqué par étapes pour montrer l'évolution des clusters.

### Prérequis

- Python 3.6 ou version ultérieure

### Étapes de Configuration et d'Exécution

1. **Création du Dossier Projet**

   Créez un nouveau dossier pour votre projet et naviguez à l'intérieur :

   ```bash
   mkdir dbscan-visualizer
   cd dbscan-visualizer
   ```

2. **Création de l'Environnement Virtuel**

   Créez un environnement virtuel pour le projet :

   ```bash
   python -m venv dbscan-env
   ```

3. **Activation de l'Environnement Virtuel**

   Activez l'environnement virtuel :

   - Sur Windows :

     ```bash
     dbscan-env\Scripts\activate
     ```

   - Sur macOS et Linux :

     ```bash
     source dbscan-env/bin/activate
     ```

4. **Création du Fichier `requirements.txt`**

   Créez un fichier `requirements.txt` pour lister les dépendances nécessaires :

   ```bash
   touch requirements.txt
   ```

   Ajoutez les lignes suivantes au fichier `requirements.txt` :

   ```
   numpy
   matplotlib
   scikit-learn
   ```

5. **Installation des Dépendances**

   Installez les dépendances à partir du fichier `requirements.txt` :

   ```bash
   pip install -r requirements.txt
   ```

6. **Création du Script Python**

   Créez un fichier Python nommé `dbscan_visualizer.py` :

   ```bash
   touch dbscan_visualizer.py
   ```

   Ajoutez le code suivant à `dbscan_visualizer.py` :

   ```python
   import tkinter as tk
   from tkinter import ttk
   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
   from matplotlib.animation import FuncAnimation
   from sklearn.datasets import make_blobs
   from sklearn.preprocessing import StandardScaler
   from sklearn.cluster import DBSCAN
   from sklearn import metrics
   import time

   class DBSCANVisualizer(tk.Tk):
       def __init__(self):
           super().__init__()
           self.title("DBSCAN Visualization")
           self.geometry("800x600")

           self.n_points = 750  # Use the given dataset size
           self.eps = 0.3
           self.min_samples = 10
           self.colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, 10)]

           self.X, self.labels_true = self.generate_points()
           self.labels = np.full(self.n_points, -1)  # Initialize labels with -1 (noise)
           self.cluster_step = 0

           self.is_paused = False
           self.start_time = None

           self.create_widgets()
           self.update_plot()

       def create_widgets(self):
           control_frame = ttk.Frame(self)
           control_frame.pack(side=tk.TOP, fill=tk.X)

           ttk.Label(control_frame, text="Number of Points:").pack(side=tk.LEFT, padx=5)
           self.n_points_var = tk.IntVar(value=self.n_points)
           ttk.Entry(control_frame, textvariable=self.n_points_var, width=5).pack(side=tk.LEFT, padx=5)

           ttk.Label(control_frame, text="Eps:").pack(side=tk.LEFT, padx=5)
           self.eps_var = tk.DoubleVar(value=self.eps)
           ttk.Entry(control_frame, textvariable=self.eps_var, width=5).pack(side=tk.LEFT, padx=5)

           ttk.Label(control_frame, text="Min Samples:").pack(side=tk.LEFT, padx=5)
           self.min_samples_var = tk.IntVar(value=self.min_samples)
           ttk.Entry(control_frame, textvariable=self.min_samples_var, width=5).pack(side=tk.LEFT, padx=5)

           ttk.Button(control_frame, text="Start", command=self.start_clustering).pack(side=tk.LEFT, padx=5)
           self.pause_button = ttk.Button(control_frame, text="Pause", command=self.pause_resume)
           self.pause_button.pack(side=tk.LEFT, padx=5)

           self.figure, self.ax = plt.subplots()
           self.canvas = FigureCanvasTkAgg(self.figure, master=self)
           self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

           self.status_label = ttk.Label(self, text="Click 'Start' to begin clustering")
           self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

           self.canvas.mpl_connect('button_press_event', self.on_click)

       def generate_points(self):
           centers = [[1, 1], [-1, -1], [1, -1]]
           X, labels_true = make_blobs(n_samples=self.n_points, centers=centers, cluster_std=0.4, random_state=0)
           X = StandardScaler().fit_transform(X)
           return X, labels_true

       def update_plot(self):
           self.ax.clear()
           unique_labels = set(self.labels)
           for k, col in zip(unique_labels, self.colors):
               if k == -1:
                   # Black used for noise.
                   col = [0, 0, 0, 1]

               class_member_mask = (self.labels == k)

               xy = self.X[class_member_mask]
               self.ax.plot(
                   xy[:, 0],
                   xy[:, 1],
                   "o",
                   markerfacecolor=tuple(col),
                   markeredgecolor="k",
                   markersize=14 if k != -1 else 6,
               )
           self.canvas.draw()

       def start_clustering(self):
           self.n_points = self.n_points_var.get()
           self.eps = self.eps_var.get()
           self.min_samples = self.min_samples_var.get()
           self.X, self.labels_true = self.generate_points()
           self.labels = np.full(self.n_points, -1)  # Reset labels
           self.is_paused = False
           self.cluster_step = 0
           self.status_label.config(text="Clustering in progress...")
           self.start_time = time.time()

           self.db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
           self.current_index = 0
           self.core_samples_mask = np.zeros_like(self.labels, dtype=bool)
           self.animation = FuncAnimation(self.figure, self.dbscan_step, interval=100, repeat=False)  # Faster interval
           self.canvas.draw()

       def dbscan_step(self, frame):
           if self.is_paused:
               return
           if self.current_index < len(self.X):
               self.db.fit(self.X[:self.current_index + 1])
               self.labels[:self.current_index + 1] = self.db.labels_
               self.core_samples_mask = np.zeros_like(self.labels, dtype=bool)
               self.core_samples_mask[:self.current_index + 1][self.db.core_sample_indices_] = True
               self.update_plot()
               self.current_index += 10  # Process points in batches of 10 for speed

               elapsed_time = time.time() - self.start_time
               estimated_total_time = (elapsed_time / (self.current_index + 1)) * self.n_points
               remaining_time = estimated_total_time - elapsed_time
               self.status_label.config(text=f"Clustering in progress... Estimated time remaining: {int(remaining_time)} seconds")
           else:
               self.animation.event_source.stop()
               self.status_label.config(text="Clustering complete")

       def on_click(self, event):
           if event.inaxes != self.ax:
               return
           new_point = np.array([[event.xdata, event.ydata]])
           self.X = np.vstack([self.X, new_point])
           self.labels = np.append(self.labels, -1)  # Initialize new point as noise
           self.n_points += 1
           self.update_plot()

       def pause_resume(self):
           if self.is_paused:
               self.is_paused = False
               self.pause_button.config(text="Pause")
           else:
               self.is_paused = True
               self.pause_button.config(text="Resume")

   if __name__ == "__main__":
       app = DBSCANVisualizer()
       app.mainloop()
   ```

7. **Exécution du Script**

   Pour exécuter le script, utilisez la commande suivante :

   ```bash
   python dbscan_visualizer.py
   ```

### Utilisation de l'Application

- **Nombre de Points** : Spécifiez le nombre de points à générer.
- **Eps** : Spécifiez le rayon pour considérer les voisins.
- **Min Samples** : Spécifiez le nombre minimum de points pour former un cluster.
- **Start** : Cliquez pour démarrer le clustering.
- **Pause** : Cliquez pour mettre en pause ou reprendre le clustering.

Pendant l'exécution, l'animation montrera comment les clusters se forment et évoluent. Le temps restant estimé sera affiché dans la barre de statut en bas de la fenêtre.

### Notes

- Si vous ajoutez des points en cliquant sur le graphique, ils seront ajoutés au prochain cycle de clustering et traités comme des points de bruit jusqu'à ce qu'ils soient intégrés dans un cluster.

Avec ce guide, vous devriez être en mesure de configurer l'environnement, d'installer les dépendances nécessaires, et d'exécuter le script pour visualiser le processus de clustering DBSCAN.
