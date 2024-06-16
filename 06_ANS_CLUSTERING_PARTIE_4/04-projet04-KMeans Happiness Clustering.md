### 📊 KMeans Happiness Clustering

Bienvenue dans le projet KMeans Happiness Clustering ! Ce projet vous guidera à travers la création d'un visualiseur KMeans pour le rapport mondial sur le bonheur en Python, étape par étape.

## 🚀 Étape 1 : Création du dossier de travail

1. Créez un dossier sur votre bureau nommé `kmeans_happiness_clustering`.

2. Ouvrez une fenêtre de terminal (ou PowerShell) et naviguez vers le dossier créé :
   ```sh
   cd Desktop/kmeans_happiness_clustering
   ```

## 🛠️ Étape 2 : Installation de l'environnement virtuel

1. Créez un environnement virtuel Python :
   ```sh
   python -m venv kmeans_happiness_clustering-env
   ```

2. Activez l'environnement virtuel :
   - Sur Windows :
     ```sh
     kmeans_happiness_clustering-env\Scripts\activate
     ```
   - Sur macOS/Linux :
     ```sh
     source kmeans_happiness_clustering-env/bin/activate
     ```

3. Installez les dépendances nécessaires :
   ```sh
   pip install -r requirements.txt
   ```

## 📝 Étape 3 : Exécution des scripts

### Version 1 : `kmeans_happiness_clustering-v1.py`

**Commandes :**
```sh
python kmeans_happiness_clustering-v1.py
```

**Description :**
- Ce script implémente une version de base de l'algorithme KMeans pour visualiser les clusters de données du rapport mondial sur le bonheur.
- Charge les données d'un fichier CSV.
- Applique l'algorithme KMeans pour regrouper les données en 4 clusters.
- Affiche les résultats des clusters dans une interface graphique Tkinter.
- Génère un graphique de dispersion interactif avec Plotly.
- Génère une carte géographique des clusters avec Plotly.

### Version 2 : `kmeans_happiness_clustering-v2.py`

**Commandes :**
```sh
python kmeans_happiness_clustering-v2.py
```

**Nouvelles fonctionnalités ajoutées :**
- Ce script améliore la version 1 en ajoutant des fonctionnalités de sélection du nombre de clusters et une meilleure interaction utilisateur.
- Permet à l'utilisateur de sélectionner le nombre de clusters via une combobox.
- Maintient les fonctionnalités de base de la version 1.

## 🧩 Résumé des commandes

### Créer et activer l'environnement virtuel :
```sh
python -m venv kmeans_happiness_clustering-env
kmeans_happiness_clustering-env\Scripts\activate
```

### Installer les dépendances :
```sh
pip install -r requirements.txt
```

### Exécuter les scripts :
```sh
python kmeans_happiness_clustering-v1.py
python kmeans_happiness_clustering-v2.py
```

## 📜 `requirements.txt`

Voici le contenu du fichier `requirements.txt` :

```
tk
pandas
numpy
scikit-learn
plotly
```

Assurez-vous de suivre chaque étape attentivement et d'exécuter les commandes dans l'ordre. Bon développement ! 🚀

### Contenu des fichiers `kmeans_happiness_clustering-v1.py` et `kmeans_happiness_clustering-v2.py`

#### `kmeans_happiness_clustering-v1.py`

```python
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import plot
import webbrowser

class HappinessClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("World Happiness Report Clustering")
        self.create_widgets()

    def create_widgets(self):
        # Load Button
        self.load_button = ttk.Button(self.root, text="Load Data", command=self.load_data)
        self.load_button.pack(pady=10)

        # Cluster Button
        self.cluster_button = ttk.Button(self.root, text="Cluster Data", command=self.cluster_data)
        self.cluster_button.pack(pady=10)

        # Plot Button
        self.plot_button = ttk.Button(self.root, text="Plot Data", command=self.plot_data)
        self.plot_button.pack(pady=10)

        # Map Button
        self.map_button = ttk.Button(self.root, text="Show Map", command=self.show_map)
        self.map_button.pack(pady=10)

        # Text box to display status and results
        self.text = tk.Text(self.root, wrap='word', width=80, height=20)
        self.text.pack(pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data = pd.read_csv(file_path)
            self.text.insert(tk.END, f"Data Loaded: {file_path}\n")
            self.text.insert(tk.END, self.data.head().to_string() + "\n")

    def cluster_data(self):
        if hasattr(self, 'data'):
            self.data_for_clustering = self.data.drop(['Overall rank', 'Score'], axis=1)
            self.data_numeric = self.data_for_clustering.select_dtypes(include=[np.number])
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.data_numeric)
            self.kmeans = KMeans(n_clusters=4, random_state=42)
            self.kmeans.fit(self.scaled_data)
            self.data['cluster'] = self.kmeans.labels_
            self.text.insert(tk.END, "Data clustered into 4 clusters.\n")
        else:
            self.text.insert(tk.END, "No data loaded.\n")

    def plot_data(self):
        if hasattr(self, 'data'):
            fig = px.scatter(self.data, x='GDP per capita', y='Score', color='cluster',
                             hover_name='Country or region', size='Overall rank', 
                             title='Happiness Score vs GDP per capita')
            plot(fig, filename='cluster_plot.html', auto_open=True)
            self.text.insert(tk.END, "Cluster plot created and opened in browser.\n")
        else:
            self.text.insert(tk.END, "No data loaded.\n")

    def show_map(self):
        if hasattr(self, 'data'):
            data = dict(type='choropleth',
                        locations=self.data["Country or region"],
                        locationmode='country names',
                        colorscale='RdYlGn',
                        z=self.data['cluster'],
                        text=self.data["Country or region"],
                        colorbar={'title': 'Clusters'})

            layout = dict(title='Geographical Visualization of Clusters',
                          geo=dict(showframe=True, projection={'type': 'azimuthal equal area'}))

            choromap3 = go.Figure(data=[data], layout=layout)
            plot(choromap3, filename='geo_cluster_map.html', auto_open=False)
            webbrowser.open('geo_cluster_map.html')
            self.text.insert(tk.END, "Geographical cluster map created and opened in browser.\n")
        else:
            self.text.insert(tk.END, "No data loaded.\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = HappinessClusteringApp(root)
    root.mainloop()
```

#### `kmeans_happiness_clustering-v2.py`

```python
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import plot
import webbrowser

class HappinessClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("World Happiness Report Clustering")
        self.create_widgets()

    def create_widgets(self):
        # Load Button
        self.load_button = ttk.Button(self.root, text="Load Data", command=self.load_data)
        self.load_button.pack(pady=10)

        # Cluster Combobox
        self.cluster_label = ttk.Label(self.root, text="Select number of clusters:")
        self.cluster_label.pack(pady=5)
        
        self.cluster_combobox = ttk.Combobox(self.root, values=[2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.cluster_combobox.current(2)  # Default value
        self.cluster_combobox.pack(pady=5)

        # Cluster Button
        self.cluster_button = ttk.Button(self.root, text="Cluster Data", command=self.cluster_data)
        self.cluster_button.pack(pady=10)

        # Plot Button
        self.plot_button = ttk.Button(self.root, text="Plot Data", command=self.plot_data)
        self.plot_button.pack(pady=10)

        # Map Button
        self.map_button = ttk.Button(self.root, text="Show Map", command=self.show_map)
        self.map_button.pack(pady=10)

        # Text box to display status and results
        self.text = tk.Text(self.root, wrap='word', width=80, height=20)
        self.text.pack(pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data = pd.read_csv(file_path)
            self.text.insert(tk.END, f"Data Loaded: {file_path}\n")
            self.text.insert(tk.END, self.data.head().to_string() + "\n")

    def cluster_data(self):
        if hasattr(self, 'data'):
           

 num_clusters = int(self.cluster_combobox.get())
            self.data_for_clustering = self.data.drop(['Overall rank', 'Score'], axis=1)
            self.data_numeric = self.data_for_clustering.select_dtypes(include=[np.number])
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.data_numeric)
            self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            self.kmeans.fit(self.scaled_data)
            self.data['cluster'] = self.kmeans.labels_
            self.text.insert(tk.END, f"Data clustered into {num_clusters} clusters.\n")
        else:
            self.text.insert(tk.END, "No data loaded.\n")

    def plot_data(self):
        if hasattr(self, 'data'):
            fig = px.scatter(self.data, x='GDP per capita', y='Score', color='cluster',
                             hover_name='Country or region', size='Overall rank', 
                             title='Happiness Score vs GDP per capita')
            plot(fig, filename='cluster_plot.html', auto_open=True)
            self.text.insert(tk.END, "Cluster plot created and opened in browser.\n")
        else:
            self.text.insert(tk.END, "No data loaded.\n")

    def show_map(self):
        if hasattr(self, 'data'):
            data = dict(type='choropleth',
                        locations=self.data["Country or region"],
                        locationmode='country names',
                        colorscale='RdYlGn',
                        z=self.data['cluster'],
                        text=self.data["Country or region"],
                        colorbar={'title': 'Clusters'})

            layout = dict(title='Geographical Visualization of Clusters',
                          geo=dict(showframe=True, projection={'type': 'azimuthal equal area'}))

            choromap3 = go.Figure(data=[data], layout=layout)
            plot(choromap3, filename='geo_cluster_map.html', auto_open=False)
            webbrowser.open('geo_cluster_map.html')
            self.text.insert(tk.END, "Geographical cluster map created and opened in browser.\n")
        else:
            self.text.insert(tk.END, "No data loaded.\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = HappinessClusteringApp(root)
    root.mainloop()
```

Assurez-vous de suivre chaque étape attentivement et d'exécuter les commandes dans l'ordre. Bon développement ! 🚀
