import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, Birch, HDBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import f1_score, accuracy_score, r2_score
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.model_selection import train_test_split
from collections import Counter

# Define clustering models in a dictionary for flexibility
clustering_models = {
    "KMeans": lambda n: KMeans(n_clusters=n, random_state=0),
    "Spectral": lambda n: SpectralClustering(n_clusters=n, assign_labels='discretize', random_state=0, affinity='nearest_neighbors'),
    "Birch": lambda n: Birch(n_clusters=n)
}

# Initialize result container
metrics = []
metric_kmeans = []
latents = []

for n_latents in [2, 3, 5, 8, 13, 21, 34, 55]:
    data = pd.read_csv(f"/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_derived/dfcs/latent_representations_{n_latents}.csv", index_col=0)
    
    for n_clusters in [3, 5, 8, 13, 21, 34, 55]:  
        for name, model_func in clustering_models.items():
            model = model_func(n_clusters)
            labels = model.fit_predict(data)
    
            # Evaluation with distance metrics
            unique_labels = np.unique(labels)
            ss = silhouette_score(data, labels) if len(unique_labels) > 1 else np.nan
            db = davies_bouldin_score(data, labels) if len(unique_labels) > 1 else np.nan
            ch = calinski_harabasz_score(data, labels) if len(unique_labels) > 1 else np.nan

            # Evaluation with a classifier
            y = labels
            X = data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            ##
            model = LGBMClassifier(n_estimators=100, verbose=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            ##
            model = LGBMRegressor(verbose=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            # Evaluation with count
            counts = np.array(list(Counter(labels).values()))
            empty_ratio = 1 - (len(pd.Series(labels).unique())/n_clusters)
            imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else np.nan
            entropy = -(p := counts/counts.sum()) @ np.log2(p) / np.log2(len(counts)) if np.log2(len(counts)) > 0 else np.nan
    
            # Store results
            metrics.append([name, n_latents, n_clusters, ss, db, ch, acc, f1,r2, empty_ratio, imbalance_ratio, entropy])
            latents.append(pd.Series(name = f"{name}_{n_latents}_{n_clusters}", data =labels))


# Latents data
Y = pd.DataFrame(latents).T
Y.to_csv("/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_derived/dfcs/all_dfcs_flattened_clusters.csv")

# Metrics data
df_metrics = pd.DataFrame(metrics, columns=["Model", "n_latents", "n_clusters", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Accuracy","f1","r2", "Empty_ratio","Imbalance_ratio","Entropy"])
df_metrics['parameters'] = df_metrics.apply(lambda row: f"{row['n_latents']}_{row['n_clusters']}", axis=1)
df_metrics.to_csv("/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_derived/dfcs/clusters_metrics.csv")

# Metrics plots
metrics = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Accuracy","f1", "r2","Empty_ratio","Imbalance_ratio","Entropy"]
fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 20), sharex=True)
for i, metric in enumerate(metrics):
    ax = axes[i]
    for model in ["KMeans", "Spectral", "Birch"]:
        dummy = df_metrics[df_metrics["Model"] == model][['parameters', metric]].set_index('parameters')
        ax.plot(dummy, label=model)
    
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis='x', rotation=45)
axes[-1].set_xlabel("Number of latent dimensions & clusters")
plt.tight_layout()
plt.savefig("/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_derived/dfcs/Clusters_Evaluation.png", dpi=200)
plt.close()
