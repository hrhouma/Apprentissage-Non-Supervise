### 📊 KMeans Bank Customer Segmentation

- Bienvenue dans le projet KMeans Bank Customer Segmentation ! Ce projet vous guidera à travers la création d'un visualiseur KMeans pour la segmentation des clients bancaires en Python, étape par étape.

## 🚀 Étape 1 : Création du dossier de travail

1. Créez un dossier sur votre bureau nommé `kmeans_bank_customer_clustering`.

2. Ouvrez une fenêtre de terminal (ou PowerShell) et naviguez vers le dossier créé :
   ```sh
   cd Desktop/kmeans_bank_customer_clustering
   ```

## 🛠️ Étape 2 : Installation de l'environnement virtuel

1. Créez un environnement virtuel Python :
   ```sh
   python -m venv kmeans_bank_customer_clustering-env
   ```

2. Activez l'environnement virtuel :
   - Sur Windows :
     ```sh
     kmeans_bank_customer_clustering-env\Scripts\activate
     ```
   - Sur macOS/Linux :
     ```sh
     source kmeans_bank_customer_clustering-env/bin/activate
     ```

3. Installez les dépendances nécessaires :
   ```sh
   pip install -r requirements.txt
   ```

## 📝 Étape 3 : Exécution des scripts

### Version 1 : `kmeans_bank_customer_clustering-v1.py`

**Commandes :**
```sh
python kmeans_bank_customer_clustering-v1.py
```

**Description :**
- Ce script implémente une version de base de l'algorithme KMeans pour visualiser les clusters de données des clients bancaires.
- Charge les données d'un fichier CSV.
- Applique l'algorithme KMeans pour regrouper les données en clusters.
- Affiche les résultats des clusters dans une interface graphique Tkinter.
- Génère un graphique de dispersion interactif avec Plotly.
- Génère une carte géographique des clusters avec Plotly.

### Version 2 : `kmeans_bank_customer_clustering-v2.py`

**Commandes :**
```sh
python kmeans_bank_customer_clustering-v2.py
```

**Nouvelles fonctionnalités ajoutées :**
- Ce script améliore la version 1 en ajoutant des fonctionnalités de sélection du nombre de clusters et une meilleure interaction utilisateur.
- Permet à l'utilisateur de sélectionner le nombre de clusters via une combobox.
- Maintient les fonctionnalités de base de la version 1.

## 🧩 Résumé des commandes

### Créer et activer l'environnement virtuel :
```sh
python -m venv kmeans_bank_customer_clustering-env
kmeans_bank_customer_clustering-env\Scripts\activate
```

### Installer les dépendances :
```sh
pip install -r requirements.txt
```

### Exécuter les scripts :
```sh
python kmeans_bank_customer_clustering-v1.py
python kmeans_bank_customer_clustering-v2.py
```

## 📜 `requirements.txt`

Voici le contenu du fichier `requirements.txt` :

```
tk
pandas
numpy
scikit-learn
plotly
matplotlib
seaborn
```

Assurez-vous de suivre chaque étape attentivement et d'exécuter les commandes dans l'ordre. Bon développement ! 🚀

### Contenu des fichiers `kmeans_bank_customer_clustering-v1.py` et `kmeans_bank_customer_clustering-v2.py`

#### `kmeans_bank_customer_clustering-v1.py`

```python
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from plotly.offline import plot

class BankCustomerClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bank Customer Segmentation")
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

        # Text box to display status and results
        self.text = tk.Text(self.root, wrap='word', width=80, height=20)
        self.text.pack(pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data = pd.read_csv(file_path)
            self.data.rename(columns={
                'CUST_ID': 'ID_CLIENT',
                'BALANCE': 'SOLDE',
                'BALANCE_FREQUENCY': 'FREQUENCE_SOLDE',
                'PURCHASES': 'ACHATS',
                'ONEOFF_PURCHASES': 'ACHATS_UNIQUES',
                'INSTALLMENTS_PURCHASES': 'ACHATS_EN_PLUSIEURS_FOIS',
                'CASH_ADVANCE': 'AVANCE_CASH',
                'PURCHASES_FREQUENCY': 'FREQUENCE_ACHATS',
                'ONEOFF_PURCHASES_FREQUENCY': 'FREQUENCE_ACHATS_UNIQUES',
                'PURCHASES_INSTALLMENTS_FREQUENCY': 'FREQUENCE_ACHATS_EN_PLUSIEURS_FOIS',
                'CASH_ADVANCE_FREQUENCY': 'FREQUENCE_AVANCE_CASH',
                'CASH_ADVANCE_TRX': 'TRANSACTIONS_AVANCE_CASH',
                'PURCHASES_TRX': 'TRANSACTIONS_ACHATS',
                'CREDIT_LIMIT': 'LIMITE_CREDIT',
                'PAYMENTS': 'PAIEMENTS',
                'MINIMUM_PAYMENTS': 'PAIEMENTS_MINIMUMS',
                'PRC_FULL_PAYMENT': 'PRC_PAIEMENT_COMPLET',
                'TENURE': 'ANCIENNETE'
            }, inplace=True)
            
            # Fill missing values
            self.data['PAIEMENTS_MINIMUMS'].fillna(self.data['PAIEMENTS_MINIMUMS'].mean(), inplace=True)
            self.data['LIMITE_CREDIT'].fillna(self.data['LIMITE_CREDIT'].mean(), inplace=True)
            
            # Drop 'ID_CLIENT' column
            self.data.drop("ID_CLIENT", axis=1, inplace=True)
            
            self.text.insert(tk.END, f"Data Loaded: {file_path}\n")
            self.text.insert(tk.END, self.data.head().to_string() + "\n")
            print("Data loaded successfully")

    def cluster_data(self):
        if hasattr(self, 'data'):
            print("Clustering data...")
            # Scale the data
            scaler = StandardScaler()
            self.data_scaled = scaler.fit_transform(self.data)
            
            # K-Means clustering
            self.kmeans = KMeans(n_clusters=4, random_state=42)
            self.kmeans.fit(self.data_scaled)
            self.data['cluster'] = self.kmeans.labels_
            
            self.text.insert(tk.END, "Data clustered into 4 clusters.\n")
            self.text.insert(tk.END, self.data.head().to_string() + "\n")  # Debug statement to show clustered data
            print("Data clustered successfully")
        else:
            self.text.insert(tk.END, "No data loaded.\n")
            print("No data loaded")

    def plot_data(self):
        if hasattr(self, 'data') and 'cluster' in self.data.columns:
            self.text.insert(tk.END, "Plotting data...\n")  # Debug statement to indicate plotting
            
            # PCA for dimensionality reduction
            pca = PCA(n_components=2)
            principal_comp = pca.fit_transform(self.data_scaled)
            
            # Create a DataFrame with PCA components
            pca_df = pd.DataFrame(data=principal_comp, columns=['pca1', 'pca2'])
            pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': self.data['cluster']})], axis=1)
            
            # Plot the clusters
            fig = px.scatter(pca_df, x='pca1', y='pca2', color='cluster', 
                             title='Customer Segmentation with PCA')
            plot(fig, filename='cluster_plot.html', auto_open=True)
            
            self.text.insert(tk.END, "Cluster plot created and opened in browser.\n")
            print("Plot created successfully")
        else:
            self.text.insert(tk.END, "No data loaded or clustering not performed.\n")
            print("No data loaded or clustering not performed")

if __name__ == "__main__":
    root = tk.Tk()
    app = BankCustomerClusteringApp(root)
    root.mainloop()
```

#### `kmeans_bank_customer_clustering-v2.py`

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

class BankCustomerClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bank Customer Segmentation")
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

        # Text box to display status and results
        self.text = tk.Text(self.root, wrap='word', width=80, height=20)
        self.text.pack(pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data = pd.read_csv(file_path)
            self.data.rename(columns={
                'CUST_ID': '

ID_CLIENT',
                'BALANCE': 'SOLDE',
                'BALANCE_FREQUENCY': 'FREQUENCE_SOLDE',
                'PURCHASES': 'ACHATS',
                'ONEOFF_PURCHASES': 'ACHATS_UNIQUES',
                'INSTALLMENTS_PURCHASES': 'ACHATS_EN_PLUSIEURS_FOIS',
                'CASH_ADVANCE': 'AVANCE_CASH',
                'PURCHASES_FREQUENCY': 'FREQUENCE_ACHATS',
                'ONEOFF_PURCHASES_FREQUENCY': 'FREQUENCE_ACHATS_UNIQUES',
                'PURCHASES_INSTALLMENTS_FREQUENCY': 'FREQUENCE_ACHATS_EN_PLUSIEURS_FOIS',
                'CASH_ADVANCE_FREQUENCY': 'FREQUENCE_AVANCE_CASH',
                'CASH_ADVANCE_TRX': 'TRANSACTIONS_AVANCE_CASH',
                'PURCHASES_TRX': 'TRANSACTIONS_ACHATS',
                'CREDIT_LIMIT': 'LIMITE_CREDIT',
                'PAYMENTS': 'PAIEMENTS',
                'MINIMUM_PAYMENTS': 'PAIEMENTS_MINIMUMS',
                'PRC_FULL_PAYMENT': 'PRC_PAIEMENT_COMPLET',
                'TENURE': 'ANCIENNETE'
            }, inplace=True)
            self.text.insert(tk.END, f"Data Loaded: {file_path}\n")
            self.text.insert(tk.END, self.data.head().to_string() + "\n")

    def cluster_data(self):
        if hasattr(self, 'data'):
            num_clusters = int(self.cluster_combobox.get())
            self.data.fillna(self.data.mean(), inplace=True)
            self.data.drop("ID_CLIENT", axis=1, inplace=True)
            self.data_numeric = self.data.select_dtypes(include=[np.number])
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
            fig = px.scatter(self.data, x='LIMITE_CREDIT', y='SOLDE', color='cluster',
                             hover_name='ID_CLIENT', size='PAIEMENTS', 
                             title='Customer Segmentation based on Credit Limit and Balance')
            plot(fig, filename='cluster_plot.html', auto_open=True)
            self.text.insert(tk.END, "Cluster plot created and opened in browser.\n")
        else:
            self.text.insert(tk.END, "No data loaded.\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = BankCustomerClusteringApp(root)
    root.mainloop()
```

Assurez-vous de suivre chaque étape attentivement et d'exécuter les commandes dans l'ordre. Bon développement ! 🚀
