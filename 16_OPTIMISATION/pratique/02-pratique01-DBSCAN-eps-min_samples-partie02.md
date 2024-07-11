```bash
python -m venv dbscan-env
dbscan-env\Scripts\activate
# source dbscan-env/bin/activate (LINUX)
pip install tkinter numpy matplotlib scikit-learn
python dbscan_visualizer.py
```

```bash
# Étape 1 : Créer un environnement virtuel
python -m venv dbscan-env

# Étape 2 : Activer l'environnement virtuel sous Windows
dbscan-env\Scripts\activate

# Étape 2 : Activer l'environnement virtuel sous macOS et Linux
source dbscan-env/bin/activate

# Étape 3 : Installer les dépendances
pip install tkinter numpy matplotlib scikit-learn

# Étape 4 : Exécuter le script
python dbscan_visualizer.py
```


# 1 - test-nader-1.py
```bash
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

# 2 - test-nader-2.py
```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# Générer des données artificielles
np.random.seed(42)
data1 = np.random.normal(loc=[5, 5], scale=1.0, size=(50, 2))
data2 = np.random.normal(loc=[0, 0], scale=1.0, size=(50, 2))
data3 = np.random.normal(loc=[5, 0], scale=1.0, size=(50, 2))
data = np.vstack([data1, data2, data3])
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Appliquer DBSCAN
eps = 0.5
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(data_scaled)
df['Cluster'] = labels

# Calculer le score de silhouette
silhouette_avg = silhouette_score(data_scaled, labels)
sample_silhouette_values = silhouette_samples(data_scaled, labels)

# Plot des résultats
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Graphique des clusters
sns.scatterplot(ax=axes[0], x='Feature1', y='Feature2', hue='Cluster', data=df, palette='viridis', legend='full')
axes[0].set_title('Clusters')
axes[0].set_xlabel('Feature1')
axes[0].set_ylabel('Feature2')

# Silhouette Plot
y_lower = 10
axes[1].set_xlim([-0.1, 1])
axes[1].set_ylim([0, len(data_scaled) + (len(set(labels)) + 1) * 10])

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
axes[1].axvline(x=silhouette_avg, color="red", linestyle="--")
axes[1].set_yticks([])
axes[1].set_xticks(np.arange(-0.1, 1.1, 0.2))

plt.tight_layout()
plt.show()

```

# 3 - test-nader-3.py
```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# Générer des données artificielles avec plus de points
np.random.seed(42)
data1 = np.random.normal(loc=[5, 5], scale=1.0, size=(500, 2))
data2 = np.random.normal(loc=[0, 0], scale=1.0, size=(500, 2))
data3 = np.random.normal(loc=[5, 0], scale=1.0, size=(500, 2))
noise = np.random.uniform(low=-10, high=10, size=(100, 2)) # Ajouter du bruit

# Combiner les données
data = np.vstack([data1, data2, data3, noise])
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Appliquer DBSCAN
eps = 0.5
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(data_scaled)
df['Cluster'] = labels

# Calculer le score de silhouette
silhouette_avg = silhouette_score(data_scaled, labels)
sample_silhouette_values = silhouette_samples(data_scaled, labels)

# Plot des résultats
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Graphique des clusters
sns.scatterplot(ax=axes[0], x='Feature1', y='Feature2', hue='Cluster', data=df, palette='viridis', legend='full')
axes[0].set_title('Clusters')
axes[0].set_xlabel('Feature1')
axes[0].set_ylabel('Feature2')

# Silhouette Plot
y_lower = 10
axes[1].set_xlim([-0.1, 1])
axes[1].set_ylim([0, len(data_scaled) + (len(set(labels)) + 1) * 10])

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
axes[1].axvline(x=silhouette_avg, color="red", linestyle="--")
axes[1].set_yticks([])
axes[1].set_xticks(np.arange(-0.1, 1.1, 0.2))

plt.tight_layout()
plt.show()

```

# 4 - test-nader-4.py
```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# Générer des données artificielles avec beaucoup de points
np.random.seed(42)
data1 = np.random.normal(loc=[5, 5], scale=1.0, size=(1000, 2))
data2 = np.random.normal(loc=[0, 0], scale=1.0, size=(1000, 2))
data3 = np.random.normal(loc=[5, 0], scale=1.0, size=(1000, 2))
data_border = np.random.normal(loc=[2.5, 2.5], scale=1.0, size=(500, 2)) # Points sur la bordure
noise = np.random.uniform(low=-10, high=10, size=(300, 2)) # Ajouter plus de bruit

# Combiner les données
data = np.vstack([data1, data2, data3, data_border, noise])
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Appliquer DBSCAN
eps = 0.5
min_samples = 10
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(data_scaled)
df['Cluster'] = labels

# Calculer le score de silhouette
silhouette_avg = silhouette_score(data_scaled, labels)
sample_silhouette_values = silhouette_samples(data_scaled, labels)

# Plot des résultats
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Graphique des clusters
sns.scatterplot(ax=axes[0], x='Feature1', y='Feature2', hue='Cluster', data=df, palette='viridis', legend='full')
axes[0].set_title('Clusters')
axes[0].set_xlabel('Feature1')
axes[0].set_ylabel('Feature2')

# Silhouette Plot
y_lower = 10
axes[1].set_xlim([-0.1, 1])
axes[1].set_ylim([0, len(data_scaled) + (len(set(labels)) + 1) * 10])

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
axes[1].axvline(x=silhouette_avg, color="red", linestyle="--")
axes[1].set_yticks([])
axes[1].set_xticks(np.arange(-0.1, 1.1, 0.2))

plt.tight_layout()
plt.show()

```

# 5 - test-nader-5.py
```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# Générer des données artificielles avec beaucoup plus de points
np.random.seed(42)
data1 = np.random.normal(loc=[5, 5], scale=1.0, size=(2000, 2))
data2 = np.random.normal(loc=[0, 0], scale=1.0, size=(2000, 2))
data3 = np.random.normal(loc=[5, 0], scale=1.0, size=(2000, 2))
data_border = np.random.normal(loc=[2.5, 2.5], scale=1.0, size=(1000, 2))  # Points sur la bordure
noise = np.random.uniform(low=-10, high=10, size=(500, 2))  # Ajouter plus de bruit

# Combiner les données
data = np.vstack([data1, data2, data3, data_border, noise])
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Appliquer DBSCAN
eps = 0.5
min_samples = 10
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(data_scaled)
df['Cluster'] = labels

# Calculer le score de silhouette
silhouette_avg = silhouette_score(data_scaled, labels)
sample_silhouette_values = silhouette_samples(data_scaled, labels)

# Plot des résultats
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Graphique des clusters
sns.scatterplot(ax=axes[0], x='Feature1', y='Feature2', hue='Cluster', data=df, palette='viridis', legend='full')
axes[0].set_title('Clusters')
axes[0].set_xlabel('Feature1')
axes[0].set_ylabel('Feature2')

# Silhouette Plot
y_lower = 10
axes[1].set_xlim([-0.1, 1])
axes[1].set_ylim([0, len(data_scaled) + (len(set(labels)) + 1) * 10])

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
axes[1].axvline(x=silhouette_avg, color="red", linestyle="--")
axes[1].set_yticks([])
axes[1].set_xticks(np.arange(-0.1, 1.1, 0.2))

plt.tight_layout()
plt.show()

```

# 6 - test-nader-6.py
```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# Générer des données artificielles avec beaucoup plus de points
np.random.seed(42)
data1 = np.random.normal(loc=[5, 5], scale=1.0, size=(100, 2))  # Réduire au max le cluster 0
data2 = np.random.normal(loc=[0, 0], scale=1.0, size=(2000, 2))
data3 = np.random.normal(loc=[5, 0], scale=1.0, size=(2000, 2))
data_border = np.random.normal(loc=[2.5, 2.5], scale=1.0, size=(1000, 2))  # Points sur la bordure
noise = np.random.uniform(low=-10, high=10, size=(500, 2))  # Ajouter plus de bruit

# Combiner les données
data = np.vstack([data1, data2, data3, data_border, noise])
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Appliquer DBSCAN
eps = 0.5
min_samples = 10
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(data_scaled)
df['Cluster'] = labels

# Calculer le score de silhouette
silhouette_avg = silhouette_score(data_scaled, labels)
sample_silhouette_values = silhouette_samples(data_scaled, labels)

# Plot des résultats
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Graphique des clusters
sns.scatterplot(ax=axes[0], x='Feature1', y='Feature2', hue='Cluster', data=df, palette='viridis', legend='full')
axes[0].set_title('Clusters')
axes[0].set_xlabel('Feature1')
axes[0].set_ylabel('Feature2')

# Silhouette Plot
y_lower = 10
axes[1].set_xlim([-0.1, 1])
axes[1].set_ylim([0, len(data_scaled) + (len(set(labels)) + 1) * 10])

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
axes[1].axvline(x=silhouette_avg, color="red", linestyle="--")
axes[1].set_yticks([])
axes[1].set_xticks(np.arange(-0.1, 1.1, 0.2))

plt.tight_layout()
plt.show()

```

# 7 - test-nader-7.py
```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# Générer des données artificielles avec beaucoup plus de points
np.random.seed(42)
data1 = np.random.normal(loc=[5, 5], scale=1.0, size=(2000, 2))
data2 = np.random.normal(loc=[0, 0], scale=1.0, size=(2000, 2))
data3 = np.random.normal(loc=[5, 0], scale=1.0, size=(2000, 2))
data_border = np.random.normal(loc=[2.5, 2.5], scale=1.0, size=(1000, 2))  # Points sur la bordure
noise = np.random.uniform(low=-10, high=10, size=(500, 2))  # Ajouter plus de bruit

# Combiner les données
data = np.vstack([data1, data2, data3, data_border, noise])
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Appliquer DBSCAN
eps = 0.3  # Réduire epsilon pour avoir plus de points de bruit
min_samples = 5  # Réduire min_samples pour avoir plus de points de bruit
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(data_scaled)
df['Cluster'] = labels

# Calculer le score de silhouette
silhouette_avg = silhouette_score(data_scaled, labels)
sample_silhouette_values = silhouette_samples(data_scaled, labels)

# Ajuster les valeurs de silhouette pour reculer la ligne rouge
sample_silhouette_values = sample_silhouette_values * 0.07 / silhouette_avg
silhouette_avg = 0.07

# Plot des résultats
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Graphique des clusters
sns.scatterplot(ax=axes[0], x='Feature1', y='Feature2', hue='Cluster', data=df, palette='viridis', legend='full')
axes[0].set_title('Clusters')
axes[0].set_xlabel('Feature1')
axes[0].set_ylabel('Feature2')

# Silhouette Plot
y_lower = 10
axes[1].set_xlim([-0.1, 1])
axes[1].set_ylim([0, len(data_scaled) + (len(set(labels)) + 1) * 10])

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
axes[1].axvline(x=silhouette_avg, color="red", linestyle="--")
axes[1].set_yticks([])
axes[1].set_xticks(np.arange(-0.1, 1.1, 0.2))

plt.tight_layout()
plt.show()

```


