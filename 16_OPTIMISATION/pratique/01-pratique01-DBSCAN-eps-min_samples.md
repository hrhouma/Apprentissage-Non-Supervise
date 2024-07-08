# OPTIMISATION DBSCAN-eps-min_samples
----
# Partie 1 - Tester le programme en local en utilisant dbscan_gui.py ou DBSCAN.ipynb dans Colab
# Lien des fichiers : 
https://drive.google.com/drive/folders/1Nylr3kw389kdNaBizsmq0uSbHeI0K4Ys?usp=sharing
## Prérequis

- Python 3.x installé sur votre système.
- pip (gestionnaire de paquets pour Python).

## Étapes d'Installation

### 1. Créer un Environnement Virtuel

1. Ouvrez une invite de commande (cmd, terminal, etc.).
2. Naviguez jusqu'au répertoire où vous souhaitez créer l'environnement virtuel.
3. Exécutez la commande suivante pour créer un environnement virtuel nommé `env` :

    ```bash
    python -m venv env
    ```

### 2. Activer l'Environnement Virtuel

- Sur Windows :

    ```bash
    .\env\Scripts\activate
    ```

- Sur macOS/Linux :

    ```bash
    source env/bin/activate
    ```

### 3. Installer les Librairies Nécessaires

Une fois l'environnement virtuel activé, installez les librairies requises avec la commande suivante :

```bash
pip install tkinter scikit-learn matplotlib seaborn pandas numpy
```

## Tester le Programme

1. Téléchargez le script Python ci-dessous et enregistrez-le dans un fichier nommé `dbscan_gui.py`.

2. Exécutez le programme en utilisant la commande suivante :

    ```bash
    python dbscan_gui.py
    ```

3. Une fenêtre Tkinter s'ouvrira avec les options pour charger les données et ajuster les paramètres DBSCAN.

## Code du Programme

```python
import tkinter as tk
from tkinter import ttk, filedialog
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import pandas as pd
import numpy as np

# Fonction pour charger les données
def load_data():
    file_path = filedialog.askopenfilename()
    if file_path:
        data = pd.read_csv(file_path)
        return data
    return None

# Fonction pour appliquer DBSCAN et afficher les résultats
def apply_dbscan(data, eps, min_samples):
    features = ['Calories', 'Protein (g)', 'Fat', 'Sugars', 'Vitamins and Minerals']
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_
    data['Cluster'] = labels

    if len(set(labels)) > 1:
        score = silhouette_score(X_scaled, labels)
        sample_silhouette_values = silhouette_samples(X_scaled, labels)
        score_label.config(text=f'Silhouette Score: {score:.3f}')
    else:
        score = None
        sample_silhouette_values = None
        score_label.config(text='Silhouette Score: Not applicable (only one cluster)')

    # Effacer les anciens graphiques
    for ax in fig.get_axes():
        ax.clear()

    fig.suptitle(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})', fontsize=16)

    sns.scatterplot(ax=axes[0], x='Calories', y='Sugars', hue='Cluster', data=data, palette='viridis', legend='full')
    axes[0].set_title('Clusters')
    axes[0].set_xlabel('Calories')
    axes[0].set_ylabel('Sugars')

    if score is not None:
        y_lower = 10
        axes[1].set_xlim([-0.1, 1])
        axes[1].set_ylim([0, len(X_scaled) + (len(set(labels)) + 1) * 10])
        for i in set(labels):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = plt.cm.nipy_spectral(float(i) / len(set(labels)))
            axes[1].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
            axes[1].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        axes[1].set_title('Silhouette Plot')
        axes[1].set_xlabel('Silhouette Coefficient Values')
        axes[1].set_ylabel('Cluster')
        axes[1].axvline(x=score, color="red", linestyle="--")
        axes[1].set_yticks([])
        axes[1].set_xticks(np.arange(-0.1, 1.1, 0.2))
    else:
        axes[1].text(0.5, 0.5, 'Silhouette Score not applicable', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)

    plt.tight_layout()
    canvas.draw()

# Fonction de rappel pour mettre à jour les graphiques
def update_plot(event=None):
    eps = eps_slider.get()
    min_samples = min_samples_slider.get()
    if data is not None:
        apply_dbscan(data, eps, min_samples)

# Création de l'interface Tkinter
root = tk.Tk()
root.title("DBSCAN Parameter Tuning")

# Bouton pour charger les données
def upload_data():
    global data
    data = load_data()
    if data is not None:
        update_plot()

load_button = ttk.Button(root, text="Upload Data", command=upload_data)
load_button.grid(row=0, column=0, columnspan=4, pady=10)

# Canvas pour afficher les graphiques
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=1, column=0, columnspan=4)

# Label et slider pour epsilon
eps_label = ttk.Label(root, text="Epsilon:")
eps_label.grid(column=0, row=2, padx=10, pady=5)
eps_slider = tk.Scale(root, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL)
eps_slider.grid(column=1, row=2, padx=10, pady=5)
eps_slider.set(0.5)

# Label et slider pour min_samples
min_samples_label = ttk.Label(root, text="Min Samples:")
min_samples_label.grid(column=2, row=2, padx=10, pady=5)
min_samples_slider = tk.Scale(root, from_=1, to=20, orient=tk.HORIZONTAL)
min_samples_slider.grid(column=3, row=2, padx=10, pady=5)
min_samples_slider.set(5)

# Label pour afficher le score de silhouette
score_label = ttk.Label(root, text="Silhouette Score: N/A")
score_label.grid(column=0, row=3, columnspan=4, pady=5)

# Bouton pour mettre à jour le graphique
update_button = ttk.Button(root, text="Update Plot", command=update_plot)
update_button.grid(column=0, row=4, columnspan=4, pady=10)

# Initialiser les données à None
data = None

# Lancer l'application Tkinter
root.mainloop()
```

## Notes

- Assurez-vous que le fichier CSV que vous chargez contient les colonnes : 'Calories', 'Protein (g)', 'Fat', 'Sugars', 'Vitamins and Minerals'.
- Vous pouvez ajuster les paramètres `eps` et `min_samples` à l'aide des curseurs dans l'interface pour voir l'impact sur le clustering et les scores de silhouette.

