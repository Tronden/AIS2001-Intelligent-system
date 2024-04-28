import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Data/FuelConsumption.csv'
data = pd.read_csv(file_path)

# Specify columns to drop
columns_to_drop = ['Year', 'MAKE', 'MODEL', 'VEHICLE CLASS', 'TRANSMISSION', 'FUEL']
data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1, inplace=True)

# Normalize feature scales
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)

# Define cluster counts to evaluate
cluster_counts = [2, 5, 10, 15, 20]

# Prepare plots
fig, axes = plt.subplots(len(cluster_counts), 1, figsize=(10, 5 * len(cluster_counts)))
silhouette_scores = {}
agglo_silhouette_scores = {}

# Perform clustering and plot results
for idx, k in enumerate(cluster_counts):
    # K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels_kmeans = kmeans.fit_predict(data_scaled)
    silhouette_kmeans = silhouette_score(data_scaled, labels_kmeans)
    silhouette_scores[k] = silhouette_kmeans
    
    # Agglomerative clustering
    agglomerative = AgglomerativeClustering(n_clusters=k)
    labels_agglo = agglomerative.fit_predict(data_scaled)
    silhouette_agglo = silhouette_score(data_scaled, labels_agglo)
    agglo_silhouette_scores[k] = silhouette_agglo
    
    # Plotting K-Means results
    ax = axes[idx]
    scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], c=labels_kmeans, cmap='viridis', edgecolor='k', alpha=0.6)
    ax.set_title(f'K-Means Clustering (k={k}) - Silhouette Score: {silhouette_kmeans:.2f}')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    plt.colorbar(scatter, ax=ax, label='Cluster Label')

plt.subplots_adjust(hspace=0.5)
plt.show()

# Print silhouette scores
print("K-Means Clustering Silhouette Scores:")
for k, score in silhouette_scores.items():
    print(f"Clusters: {k}, Silhouette Score: {score:.2f}")

print("Agglomerative Clustering Silhouette Scores:")
for k, score in agglo_silhouette_scores.items():
    print(f"Clusters: {k}, Silhouette Score: {score:.2f}")

# Function to plot dendrogram
def plot_dendrogram(model, ax, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, ax=ax, **kwargs)

# Plot dendrograms for Agglomerative Clustering with different linkages
def plot_agglomerative_dendrograms(data_scaled):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    linkages = ['ward', 'average', 'complete']
    titles = ['Hierarchical Clustering Dendrogram (Ward)', 
              'Hierarchical Clustering Dendrogram (Average)', 
              'Hierarchical Clustering Dendrogram (Complete)']

    for i, linkage in enumerate(linkages):
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=linkage)
        model.fit(data_scaled)
        axes[i].set_title(titles[i])
        plot_dendrogram(model, ax=axes[i], truncate_mode='level', p=3)
        axes[i].set_xlabel("Number of points in node.")

    plt.tight_layout()
    plt.show()

# Plot dendrograms for different linkages
plot_agglomerative_dendrograms(data_scaled)
