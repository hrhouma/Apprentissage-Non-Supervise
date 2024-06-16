# 📊 KMeans Visualizer

Bienvenue dans le projet KMeans Visualizer ! Ce projet vous guidera à travers la création d'un visualiseur KMeans en Python, étape par étape.

## 🚀 Étape 1 : Création du dossier de travail

1. Créez un dossier sur votre bureau nommé `kmeans_visualizer`.

2. Ouvrez une fenêtre de terminal (ou PowerShell) et naviguez vers le dossier créé :
   ```sh
   cd Desktop/kmeans_visualizer
   ```

## 🛠️ Étape 2 : Installation de l'environnement virtuel

1. Créez un environnement virtuel Python :
   ```sh
   python -m venv kmeans_visualizer-env
   ```

2. Activez l'environnement virtuel :
   - Sur Windows :
     ```sh
     kmeans_visualizer-env\Scripts\activate
     ```
   - Sur macOS/Linux :
     ```sh
     source kmeans_visualizer-env/bin/activate
     ```

3. Installez les dépendances nécessaires :
   ```sh
   pip install -r requirements.txt
   ```

## 📝 Étape 3 : Exécution des scripts

### Version 1 : `kmeans_visualizer-v1.py`

Ce script implémente une version de base de l'algorithme KMeans pour visualiser les clusters de données.

**Commandes :**
```sh
python kmeans_visualizer-v1.py
```

**Description :**
- Génère des points aléatoires.
- Initialise les centroids.
- Affiche les points et les centroids sur un graphique.

### Version 2 : `kmeans_visualizer-v2.py`

Ce script améliore la version 1 en ajoutant des fonctionnalités de visualisation dynamique et une interaction utilisateur.

**Nouvelles fonctionnalités ajoutées :**
- Animation de l'algorithme KMeans avec `FuncAnimation`.
- Ajout de points en cliquant sur la zone de tracé.

**Commandes :**
```sh
python kmeans_visualizer-v2.py
```

### Version 3 : `kmeans_visualizer-v3.py`

Ce script améliore la version 2 en ajoutant une fonctionnalité de pause et de reprise pour l'animation.

**Nouvelles fonctionnalités ajoutées :**
- Bouton de pause/reprise pour l'animation.
- Prolongation de l'animation à 50 itérations.

**Commandes :**
```sh
python kmeans_visualizer-v3.py
```

## 🧩 Résumé des commandes

1. Créer et activer l'environnement virtuel :
   ```sh
   python -m venv kmeans_visualizer-env
   kmeans_visualizer-env\Scripts\activate
   ```

2. Installer les dépendances :
   ```sh
   pip install -r requirements.txt
   ```

## 📜 `requirements.txt`

Voici le contenu du fichier `requirements.txt` :

```
numpy
pandas
matplotlib
scikit-learn
tk
```

3. Exécuter les scripts :
   ```sh
   python -m venv kmeans_visualizer-env
   kmeans_visualizer-env\Scripts\activate
   pip install -r requirements.txt
   # Copier coller le contenu de kmeans_visualizer-v2.py
   python kmeans_visualizer-v1.py
   # CTL+C pour sortir ==> Vos interprétations ??
   # Copier coller le contenu de kmeans_visualizer-v2.py
   python kmeans_visualizer-v2.py
   # CTL+C pour sortir ==> Vos interprétations ?? Ajout par rapport à la version précédente ?
   # Copier coller le contenu de kmeans_visualizer-v2.py
   python kmeans_visualizer-v3.py
   # CTL+C pour sortir ==> Vos interprétations ?? Ajout par rapport à la version précédente ?
   ```

Assurez-vous de suivre chaque étape attentivement et d'exécuter les commandes dans l'ordre. Bon développement ! 🚀


### Contenu des fichiers `kmeans_visualizer-v1.py`, `kmeans_visualizer-v2.py`, `kmeans_visualizer-v3.py`

#### `kmeans_visualizer-v1.py`

```python
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

class KMeansVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("K-Means Visualization")
        self.geometry("800x600")

        self.n_points = 200
        self.n_clusters = 4
        self.colors = ["#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#e74c3c"]

        self.points = self.generate_points(self.n_points)
        self.centroids = self.initialize_centroids(self.n_clusters)
        self.point_colors = ['grey'] * self.n_points  # Initialize point colors

        self.create_widgets()
        self.update_plot()

    def create_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Number of Points:").pack(side=tk.LEFT, padx=5)
        self.n_points_var = tk.IntVar(value=self.n_points)
        ttk.Entry(control_frame, textvariable=self.n_points_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Number of Clusters:").pack(side=tk.LEFT, padx=5)
        self.n_clusters_var = tk.IntVar(value=self.n_clusters)
        ttk.Entry(control_frame, textvariable=self.n_clusters_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Start", command=self.start_clustering).pack(side=tk.LEFT, padx=5)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def generate_points(self, n):
        return np.random.rand(n, 2)

    def initialize_centroids(self, k):
        indices = random.sample(range(len(self.points)), k)
        return self.points[indices]

    def update_plot(self):
        self.ax.clear()
        self.ax.scatter(self.points[:, 0], self.points[:, 1], c=self.point_colors, s=30)
        self.ax.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=100, marker='X')
        self.canvas.draw()

    def start_clustering(self):
        self.n_points = self.n_points_var.get()
        self.n_clusters = self.n_clusters_var.get()
        self.points = self.generate_points(self.n_points)
        self.centroids = self.initialize_centroids(self.n_clusters)
        self.point_colors = ['grey'] * self.n_points  # Reset point colors
        self.kmeans()

    def kmeans(self):
        for _ in range(10):  # Iterate a fixed number of times for simplicity
            clusters = self.assign_clusters()
            new_centroids = self.update_centroids(clusters)
            self.centroids = new_centroids
            self.update_plot()
            self.update()
            self.after(500)

    def assign_clusters(self):
        distances = np.sqrt(((self.points - self.centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_indices = np.argmin(distances, axis=0)
        self.point_colors = [self.colors[i] for i in cluster_indices]
        return cluster_indices

    def update_centroids(self, clusters):
        new_centroids = np.array([self.points[clusters == k].mean(axis=0) for k in range(self.n_clusters)])
        return new_centroids

if __name__ == "__main__":
    app = KMeansVisualizer()
    app.mainloop()
```

#### `kmeans_visualizer-v2.py`

```python
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import random

class KMeansVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("K-Means Visualization")
        self.geometry("800x600")

        self.n_points = 200
        self.n_clusters = 4
        self.colors = ["#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#e74c3c"]

        self.points = self.generate_points(self.n_points)
        self.centroids = self.initialize_centroids(self.n_clusters)
        self.point_colors = ['grey'] * self.n_points  # Initialize point colors

        self.create_widgets()
        self.update_plot()

    def create_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Number of Points:").pack(side=tk.LEFT, padx=5)
        self.n_points_var = tk.IntVar(value=self.n_points)
        ttk.Entry(control_frame, textvariable=self.n_points_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Number of Clusters:").pack(side=tk.LEFT, padx=5)
        self.n_clusters_var = tk.IntVar(value=self.n_clusters)
        ttk.Entry(control_frame, textvariable=self.n_clusters_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Start", command=self.start_clustering).pack(side=tk.LEFT, padx=5)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect('button_press_event', self.on_click)

    def generate_points(self, n):
        return np.random.rand(n, 2)

    def initialize_centroids(self

, k):
        indices = random.sample(range(len(self.points)), k)
        return self.points[indices]

    def update_plot(self):
        self.ax.clear()
        self.ax.scatter(self.points[:, 0], self.points[:, 1], c=self.point_colors, s=30)
        self.ax.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=100, marker='X')
        self.canvas.draw()

    def start_clustering(self):
        self.n_points = self.n_points_var.get()
        self.n_clusters = self.n_clusters_var.get()
        self.points = self.generate_points(self.n_points)
        self.centroids = self.initialize_centroids(self.n_clusters)
        self.point_colors = ['grey'] * self.n_points  # Reset point colors
        self.iteration = 0
        self.animation = FuncAnimation(self.figure, self.kmeans_step, interval=500, repeat=False)
        self.canvas.draw()

    def kmeans_step(self, frame):
        if self.iteration < 10:  # Iterate a fixed number of times for simplicity
            clusters = self.assign_clusters()
            new_centroids = self.update_centroids(clusters)
            self.centroids = new_centroids
            self.update_plot()
            self.iteration += 1
        else:
            self.animation.event_source.stop()

    def assign_clusters(self):
        distances = np.sqrt(((self.points - self.centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_indices = np.argmin(distances, axis=0)
        self.point_colors = [self.colors[i] for i in cluster_indices]
        return cluster_indices

    def update_centroids(self, clusters):
        new_centroids = np.array([self.points[clusters == k].mean(axis=0) for k in range(self.n_clusters)])
        return new_centroids

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        self.points = np.vstack([self.points, [event.xdata, event.ydata]])
        self.point_colors.append('grey')
        self.update_plot()

if __name__ == "__main__":
    app = KMeansVisualizer()
    app.mainloop()
```

#### `kmeans_visualizer-v3.py`

```python
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import random

class KMeansVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("K-Means Visualization")
        self.geometry("800x600")

        self.n_points = 200
        self.n_clusters = 4
        self.colors = ["#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#e74c3c"]

        self.points = self.generate_points(self.n_points)
        self.centroids = self.initialize_centroids(self.n_clusters)
        self.point_colors = ['grey'] * self.n_points  # Initialize point colors

        self.is_paused = False

        self.create_widgets()
        self.update_plot()

    def create_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Number of Points:").pack(side=tk.LEFT, padx=5)
        self.n_points_var = tk.IntVar(value=self.n_points)
        ttk.Entry(control_frame, textvariable=self.n_points_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Number of Clusters:").pack(side=tk.LEFT, padx=5)
        self.n_clusters_var = tk.IntVar(value=self.n_clusters)
        ttk.Entry(control_frame, textvariable=self.n_clusters_var, width=5).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Start", command=self.start_clustering).pack(side=tk.LEFT, padx=5)
        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.pause_resume)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect('button_press_event', self.on_click)

    def generate_points(self, n):
        return np.random.rand(n, 2)

    def initialize_centroids(self, k):
        indices = random.sample(range(len(self.points)), k)
        return self.points[indices]

    def update_plot(self):
        self.ax.clear()
        self.ax.scatter(self.points[:, 0], self.points[:, 1], c=self.point_colors, s=30)
        self.ax.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=100, marker='X')
        self.canvas.draw()

    def start_clustering(self):
        self.n_points = self.n_points_var.get()
        self.n_clusters = self.n_clusters_var.get()
        self.points = self.generate_points(self.n_points)
        self.centroids = self.initialize_centroids(self.n_clusters)
        self.point_colors = ['grey'] * self.n_points  # Reset point colors
        self.iteration = 0
        self.is_paused = False
        self.animation = FuncAnimation(self.figure, self.kmeans_step, interval=500, repeat=False)
        self.canvas.draw()

    def kmeans_step(self, frame):
        if self.is_paused:
            return
        if self.iteration < 50:  # Prolong the animation to 50 iterations
            clusters = self.assign_clusters()
            new_centroids = self.update_centroids(clusters)
            self.centroids = new_centroids
            self.update_plot()
            self.iteration += 1
        else:
            self.animation.event_source.stop()

    def assign_clusters(self):
        distances = np.sqrt(((self.points - self.centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_indices = np.argmin(distances, axis=0)
        self.point_colors = [self.colors[i] for i in cluster_indices]
        return cluster_indices

    def update_centroids(self, clusters):
        new_centroids = np.array([self.points[clusters == k].mean(axis=0) for k in range(self.n_clusters)])
        return new_centroids

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        new_point = np.array([[event.xdata, event.ydata]])
        self.points = np.vstack([self.points, new_point])
        self.point_colors.append('grey')
        self.update_plot()

    def pause_resume(self):
        if self.is_paused:
            self.is_paused = False
            self.pause_button.config(text="Pause")
        else:
            self.is_paused = True
            self.pause_button.config(text="Resume")

if __name__ == "__main__":
    app = KMeansVisualizer()
    app.mainloop()
```
