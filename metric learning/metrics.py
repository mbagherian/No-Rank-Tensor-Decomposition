# no-rank-tensor-decomposition/metric_learning/metrics.py
import numpy as np
import torch
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def calculate_separation_ratio(embeddings, labels):
    """Calculate separation ratio between inter-class and intra-class distances"""
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    
    intra_class_dists = []
    inter_class_dists = []
    
    # Calculate pairwise distances
    dist_matrix = pairwise_distances(embeddings_np)
    
    for i in range(len(embeddings_np)):
        for j in range(i + 1, len(embeddings_np)):
            dist = dist_matrix[i, j]
            if labels[i] == labels[j]:
                intra_class_dists.append(dist)
            else:
                inter_class_dists.append(dist)
    
    if len(intra_class_dists) == 0 or len(inter_class_dists) == 0:
        return 0.0
    
    mean_inter = np.mean(inter_class_dists)
    mean_intra = np.mean(intra_class_dists)
    
    separation_ratio = mean_inter / mean_intra if mean_intra > 0 else float('inf')
    return separation_ratio

def calculate_clustering_metrics(embeddings, true_labels, n_clusters=None):
    """Calculate ARI and NMI for clustering performance"""
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(embeddings_np)
    
    # Calculate metrics
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    
    return ari, nmi, predicted_labels

def calculate_all_metrics(embeddings, true_labels, original_data=None, n_clusters=None):
    """Calculate comprehensive metrics including ARI and NMI"""
    metrics = {}
    
    # Traditional clustering metrics
    metrics['Silhouette'] = silhouette_score(embeddings, true_labels)
    metrics['Davies-Bouldin'] = davies_bouldin_score(embeddings, true_labels)
    
    # Structural metrics
    metrics['Separation_Ratio'] = calculate_separation_ratio(embeddings, true_labels)
    
    if original_data is not None:
        metrics['Continuity'] = calculate_continuity(embeddings, original_data)
        metrics['Trustworthiness'] = calculate_trustworthiness(embeddings, original_data)
    else:
        metrics['Continuity'] = 0.0
        metrics['Trustworthiness'] = 0.0
    
    # Clustering agreement metrics
    metrics['ARI'], metrics['NMI'], _ = calculate_clustering_metrics(embeddings, true_labels, n_clusters)
    
    return metrics

def calculate_continuity(embeddings, original_data, k=5):
    """Calculate continuity metric for manifold preservation"""
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    original_flat = original_data.reshape(original_data.shape[0], -1)
    n = len(embeddings_np)

    # Get full rank neighbors in embedding space
    nbrs_embedding = NearestNeighbors(n_neighbors=n).fit(embeddings_np)
    _, indices_embedding = nbrs_embedding.kneighbors(embeddings_np)

    # Top-k neighbors in original space
    nbrs_original = NearestNeighbors(n_neighbors=k+1).fit(original_flat)
    _, indices_original = nbrs_original.kneighbors(original_flat)

    continuity = 0.0
    for i in range(n):
        embed_ranks = {idx: rank for rank, idx in enumerate(indices_embedding[i])}
        orig_neighbors = indices_original[i][1:]  # Exclude self

        for neighbor in orig_neighbors:
            if neighbor not in indices_embedding[i][:k+1]:
                rank = embed_ranks[neighbor]
                continuity += (rank - k)

    normalizer = n * k * (2 * n - 3 * k - 1)
    continuity = 1 - (2 / normalizer) * continuity
    return continuity

def calculate_trustworthiness(embeddings, original_data, k=5):
    """Calculate trustworthiness metric for manifold preservation"""
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    original_flat = original_data.reshape(original_data.shape[0], -1)
    n = len(embeddings_np)

    # Get the full rank lists (not just top-k) in original space
    nbrs_original = NearestNeighbors(n_neighbors=n).fit(original_flat)
    _, indices_original = nbrs_original.kneighbors(original_flat)

    # Top-k neighbors in embedding space
    nbrs_embedding = NearestNeighbors(n_neighbors=k+1).fit(embeddings_np)
    _, indices_embedding = nbrs_embedding.kneighbors(embeddings_np)

    trustworthiness = 0.0

    for i in range(n):
        orig_ranks = {idx: rank for rank, idx in enumerate(indices_original[i])}
        embed_neighbors = indices_embedding[i][1:]  # remove self

        for neighbor in embed_neighbors:
            if neighbor not in indices_original[i][:k+1]:
                rank = orig_ranks[neighbor]
                trustworthiness += (rank - k)

    normalizer = n * k * (2 * n - 3 * k - 1)
    trustworthiness = 1 - (2 / normalizer) * trustworthiness
    return trustworthiness