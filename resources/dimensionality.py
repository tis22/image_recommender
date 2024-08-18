import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap


def reduce_dimensionality(df_to_reduce, algorithm, dimensions=2):
    """
    Reduces the dimensionality of the data using the specified algorithm.

    Args:
        df_to_reduce (np.array): Data to reduce.
        algorithm (str): Dimensionality reduction algorithm ('tsne', 'umap', 'pca').
        dimensions (int, optional): Number of dimensions to reduce to. Defaults to 2.

    Returns:
        np.array: Reduced data.
    """
    if algorithm == "tsne":
        tsne = TSNE(n_components=dimensions)
        reduced_data = tsne.fit_transform(df_to_reduce)
    elif algorithm == "umap":
        umap_model = umap.UMAP(n_components=dimensions)
        reduced_data = umap_model.fit_transform(df_to_reduce)
    else:
        pca = PCA(n_components=dimensions)
        reduced_data = pca.fit_transform(df_to_reduce)
    return reduced_data


def save_dimensionality_results(df, algorithm_name):
    """
    Saves the dimensionality reduction results to a .npy file.

    Args:
        df (np.array): Reduced data.
        algorithm_name (str): Name of the algorithm used.
    """
    np.save(f"{algorithm_name}_results.npy", df)


def create_clusters(df_to_reduce, cluster_amount=100):
    """
    Creates clusters using KMeans.

    Args:
        df_to_reduce (np.array): Data to cluster.
        cluster_amount (int, optional): Number of clusters. Defaults to 100.

    Returns:
        np.array: Cluster labels.
    """
    kmeans = KMeans(n_clusters=cluster_amount)
    labels = kmeans.fit_predict(df_to_reduce)
    return labels
