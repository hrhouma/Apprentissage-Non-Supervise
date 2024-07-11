# Clustering DBSCAN avec Visualisation Interactive
# Partie 1
## Mode d'utilisation

Ce programme permet de visualiser des clusters générés par l'algorithme DBSCAN et d'interagir avec les graphiques pour explorer les résultats. Voici les différentes interactions possibles :

# Environnement
```bash
python -m venv dbscan-silho
dbscan-silho-env\Scripts\activate
# source dbscan-silho/bin/activate (LINUX)
pip install numpy matplotlib scikit-learn seaborn
python dbscan_visualizer.py
```

### Clic Droit (Zoom)

- **Clic droit** sur une zone du graphique des clusters pour zoomer sur cette zone.
- La zone zoomée met à jour automatiquement le diagramme des silhouettes en fonction des points affichés.

### Double Clic (Réinitialisation)

- **Double clic** sur le graphique des clusters pour réinitialiser la vue à l'échelle originale.
- Le diagramme des silhouettes sera également réinitialisé pour afficher les silhouettes de tous les clusters.

### Clic sur les Points (Sélection)

- **Clic sur un point** du graphique des clusters pour sélectionner ce point.
- Le point sélectionné sera mis en évidence par un cercle bleu.
- Le diagramme des silhouettes mettra en évidence la valeur de silhouette correspondant au point sélectionné ou l'échantillon.

## Dépendances

Ce programme nécessite les bibliothèques suivantes :

- numpy
- matplotlib
- scikit-learn

Pour installer ces dépendances, exécutez la commande suivante :
```bash
pip install numpy matplotlib scikit-learn
```

## Code Source

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs

# Génération de données aléatoires
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.60, random_state=0)

# Application de l'algorithme DBSCAN
db = DBSCAN(eps=0.3, min_samples=5).fit(X)
labels = db.labels_

# Calcul du score de silhouette
silhouette_vals = silhouette_samples(X, labels)

# Visualisation des résultats
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Visualisation des clusters
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

scatter = []
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Couleur noire pour le bruit

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    sc = ax1.scatter(xy[:, 0], xy[:, 1], color=tuple(col), edgecolor='k', s=30, label=f"Cluster {k}", picker=True, pickradius=5)
    scatter.append(sc)

ax1.set_title('DBSCAN Clustering')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend()

# Fonction de visualisation du diagramme des silhouettes
def plot_silhouette(ax, labels, silhouette_vals):
    ax.clear()
    y_lower = 10
    y_ticks = []
    for i in range(len(unique_labels)):
        ith_cluster_silhouette_values = silhouette_vals[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.Spectral(float(i) / len(unique_labels))
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        y_ticks.append(y_lower + 0.5 * size_cluster_i)
        y_lower = y_upper + 10

    silhouette_avg = silhouette_score(X, labels)
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_title("Silhouette Plot")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(range(len(unique_labels)))
    highlight_line = ax.axhline(y=-1, color='blue', linestyle='--', visible=False)

    return highlight_line

highlight_line = plot_silhouette(ax2, labels, silhouette_vals)

# Ajout d'un cercle pour mettre en évidence le point cliqué
highlight_point, = ax1.plot([], [], 'o', markersize=15, markeredgewidth=2, markeredgecolor='blue', fillstyle='none')

def onpick(event):
    ind = event.ind[0]
    label = labels[ind]
    silhouette_val = silhouette_vals[ind]

    print(f"Point index: {ind}, Label: {label}, Silhouette value: {silhouette_val}")

    # Mettre en évidence le point cliqué
    highlight_point.set_data([X[ind, 0]], [X[ind, 1]])
    highlight_point.set_visible(True)

    # Trouver la position y du point correspondant dans le diagramme des silhouettes
    y_lower = 10
    for i in range(len(unique_labels)):
        if i == label:
            ith_cluster_silhouette_values = silhouette_vals[labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            break
        y_lower += (silhouette_vals[labels == i].shape[0] + 10)

    y_tick_pos = y_lower + np.where(ith_cluster_silhouette_values == silhouette_val)[0][0]

    highlight_line.set_ydata([y_tick_pos])
    highlight_line.set_visible(True)
    fig.canvas.draw_idle()

# Fonction de mise à jour du graphique des silhouettes en fonction de la zone zoomée
def onselect(event):
    if event.button == 3:  # Right-click to zoom
        ax1.set_xlim(event.xdata - 1, event.xdata + 1)
        ax1.set_ylim(event.ydata - 1, event.ydata + 1)
        ax1.figure.canvas.draw()

        # Mise à jour du diagramme des silhouettes en fonction de la zone zoomée
        mask = (X[:, 0] >= event.xdata - 1) & (X[:, 0] <= event.xdata + 1) & (X[:, 1] >= event.ydata - 1) & (X[:, 1] <= event.ydata + 1)
        zoomed_labels = labels[mask]
        zoomed_silhouette_vals = silhouette_samples(X[mask], zoomed_labels)  # Recalculate silhouettes for the zoomed area

        if len(zoomed_labels) > 0:
            global highlight_line
            highlight_line = plot_silhouette(ax2, zoomed_labels, zoomed_silhouette_vals)
            fig.canvas.draw_idle()

# Fonction de réinitialisation des graphiques
def ondoubleclick(event):
    if event.dblclick:  # Double-click to reset
        ax1.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        ax1.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        ax1.figure.canvas.draw()

        # Réinitialisation du diagramme des silhouettes complet
        global highlight_line
        highlight_line = plot_silhouette(ax2, labels, silhouette_vals)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('pick_event', onpick)
fig.canvas.mpl_connect('button_press_event', onselect)
fig.canvas.mpl_connect('button_press_event', ondoubleclick)

plt.show()

```

---

# Annexe 1 - Explications supplémentaires

### Fonction `plot_silhouette`

Cette fonction génère un diagramme de silhouettes pour visualiser la qualité du clustering. Chaque cluster est représenté par une couleur différente, et une ligne rouge pointillée montre le score moyen de silhouette pour l'ensemble des clusters.

### Fonction `onpick`

Cette fonction est déclenchée lorsqu'un utilisateur clique sur un point dans le graphique des clusters. Elle met en évidence le point cliqué et affiche la valeur de silhouette correspondante dans le diagramme des silhouettes.

### Fonction `onselect`

Cette fonction est déclenchée lorsqu'un utilisateur effectue un clic droit sur le graphique des clusters. Elle zoome sur la zone cliquée et met à jour le diagramme des silhouettes en conséquence.

### Fonction `ondoubleclick`

Cette fonction est déclenchée lorsqu'un utilisateur effectue un double-clic sur le graphique des clusters. Elle réinitialise la vue du graphique des clusters et du diagramme des silhouettes à leurs états originaux.

# Partie 2 - Version 2 du code

Voici un README complet pour le code mis à jour, incluant des instructions détaillées pour les utilisateurs.

---

# README - Clustering DBSCAN avec Visualisation Interactive

## Mode d'utilisation

Ce programme permet de visualiser des clusters générés par l'algorithme DBSCAN et d'interagir avec les graphiques pour explorer les résultats. Voici les différentes interactions possibles :

### Lasso Selector (Sélection de points)

- **Lasso Selector** : Cliquez et maintenez le bouton gauche de la souris pour dessiner une forme libre autour des points que vous souhaitez sélectionner dans le graphique des clusters.
- Les points sélectionnés seront mis en évidence dans le diagramme des silhouettes avec un contour bleu.

### Double Clic (Réinitialisation)

- **Double clic** sur le graphique des clusters pour réinitialiser la vue à l'échelle originale.
- Le diagramme des silhouettes sera également réinitialisé pour afficher les silhouettes de tous les clusters.

## Dépendances

Ce programme nécessite les bibliothèques suivantes :

- numpy
- matplotlib
- scikit-learn

Pour installer ces dépendances, exécutez la commande suivante :
```bash
pip install numpy matplotlib scikit-learn
```

## Code Source

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs

# Génération de données aléatoires
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.60, random_state=0)

# Application de l'algorithme DBSCAN
db = DBSCAN(eps=0.3, min_samples=5).fit(X)
labels = db.labels_

# Calcul du score de silhouette
silhouette_vals = silhouette_samples(X, labels)

# Visualisation des résultats
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Visualisation des clusters
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

scatter = []
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Couleur noire pour le bruit

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    sc = ax1.scatter(xy[:, 0], xy[:, 1], color=tuple(col), edgecolor='k', s=30, label=f"Cluster {k}", picker=True)
    scatter.append(sc)

ax1.set_title('DBSCAN Clustering')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend()

# Fonction de visualisation du diagramme des silhouettes
def plot_silhouette(ax, labels, silhouette_vals):
    ax.clear()
    y_lower = 10
    y_ticks = []
    for i in range(len(unique_labels)):
        ith_cluster_silhouette_values = silhouette_vals[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.Spectral(float(i) / len(unique_labels))
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        y_ticks.append(y_lower + 0.5 * size_cluster_i)
        y_lower = y_upper + 10

    silhouette_avg = silhouette_score(X, labels)
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_title("Silhouette Plot")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(range(len(unique_labels)))
    highlight_line = ax.axhline(y=-1, color='blue', linestyle='--', visible=False)

    return highlight_line

highlight_line = plot_silhouette(ax2, labels, silhouette_vals)

def onselect_verts(verts):
    path = Path(verts)
    selected_indices = np.nonzero(path.contains_points(X))[0]
    selected_labels = labels[selected_indices]
    selected_silhouette_vals = silhouette_vals[selected_indices]

    # Clear and plot the selected points in the silhouette plot
    ax2.clear()
    highlight_line = plot_silhouette(ax2, labels, silhouette_vals)
    y_lower = 10
    for i in range(len(unique_labels)):
        ith_cluster_silhouette_values = silhouette_vals[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.Spectral(float(i) / len(unique_labels))
        if i in selected_labels:
            ax2.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor='blue', alpha=0.7)
        else:
            ax2.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

        y_lower = y_upper + 10

    fig.canvas.draw_idle()

lasso = LassoSelector(ax1, onselect_verts)

# Fonction de réinitialisation des graphiques
def ondoubleclick(event):
    if event.dblclick:  # Double-click to reset
        ax1.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        ax1.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        ax1.figure.canvas.draw()

        # Réinitialisation du diagramme des silhouettes complet
        global highlight_line
        highlight_line = plot_silhouette(ax2, labels, silhouette_vals)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', ondoubleclick)

plt.show()

```

---

## Explications supplémentaires

### Fonction `plot_silhouette`

Cette fonction génère un diagramme de silhouettes pour visualiser la qualité du clustering. Chaque cluster est représenté par une couleur différente, et une ligne rouge pointillée montre le score moyen de silhouette pour l'ensemble des clusters.

### Fonction `onselect_verts`

Cette fonction est déclenchée lorsqu'un utilisateur utilise le Lasso Selector pour sélectionner des points dans le graphique des clusters. Les points sélectionnés sont mis en évidence dans le diagramme des silhouettes avec un contour bleu.

### Fonction `ondoubleclick`

Cette fonction est déclenchée lorsqu'un utilisateur effectue un double-clic sur le graphique des clusters. Elle réinitialise la vue du graphique des clusters et du diagramme des silhouettes à leurs états originaux.

---

