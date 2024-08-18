import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import plotly.express as px
import sqlite3
import pandas as pd

try:
    from database_operations import get_result_paths
except ModuleNotFoundError:
    pass

try:
    from .database_operations import get_result_paths
except ImportError:
    pass


def print_images(input_images, result_paths, similarities, best_n):
    """
    Plots input images and their most similar results.

    Args:
        input_images (pd.DataFrame): DataFrame containing the input images' paths and related information.
        result_paths (list): List of file paths for the most similar images.
        similarities (list): List of similarity scores or IDs corresponding to the similar images.
        best_n (int): Number of top similar images to display.

    Returns:
        None
    """
    input_images_number = len(input_images)
    max_images = max(input_images_number, best_n)
    figsize = (20, 5)
    fig, axes = plt.subplots(1, max_images, figsize=figsize)
    for i in range(input_images_number):
        image = Image.open(input_images.iloc[i, 1])
        axes[i].imshow(image)
        axes[i].set_title(f"Input: {i+1}")
        axes[i].axis("off")
    for i in range(input_images_number, max_images):
        axes[i].axis("off")
    fig, axes = plt.subplots(1, max_images, figsize=figsize)
    for i in range(best_n):
        image = Image.open(result_paths[i])
        axes[i].imshow(image)
        axes[i].set_title(f"Result ID: {similarities[i]}")
        axes[i].axis("off")
    for i in range(best_n, max_images):
        axes[i].axis("off")
    plt.show()


def plot_selected_images(id_list):
    """
    Plots images based on a list of IDs.

    Args:
        id_list (list): List of image IDs to be plotted.

    Returns:
        None
    """
    conn = sqlite3.connect("database/bd_database.db")
    curs = conn.cursor()
    result_paths_list = get_result_paths(curs, id_list)
    fig, axes = plt.subplots(1, len(id_list), figsize=(20, 5))
    for i in range(len(id_list)):
        if i < len(id_list):
            image = Image.open(result_paths_list[i])
            axes[i].imshow(image)
            axes[i].set_title(f"ID: {id_list[i]}")
        axes[i].axis("off")
    plt.show()


def plot_dimensionality_reduction(algorithm_data, labels=None, output_file=None):
    """
    Plots the result of dimensionality reduction in 2D or 3D.

    Args:
        algorithm_data (np.ndarray): Array containing the reduced dimensionality data (2D or 3D).
        labels (np.ndarray, optional): Array of labels to color the data points. Defaults to None.
        output_file (str, optional): Path to save the plot as an HTML file. Defaults to None.

    Returns:
        None
    """

    if labels is None:
        labels = np.array([""] * len(algorithm_data))
    num_dims = algorithm_data.shape[1]
    df_embedding_ids = pd.read_pickle("Embedding.pkl")
    ids = df_embedding_ids["ID"].values
    if num_dims == 2:
        fig = px.scatter(
            x=algorithm_data[:, 0],
            y=algorithm_data[:, 1],
            color=labels.astype(str),
            hover_name=ids,
            labels={"x": "Dim1", "y": "Dim2"},
            title="2D Plot",
        )
        if output_file is not None:
            fig.write_html(output_file)
        fig.show()
    else:
        fig = px.scatter_3d(
            x=algorithm_data[:, 0],
            y=algorithm_data[:, 1],
            z=algorithm_data[:, 2],
            color=labels.astype(str),
            hover_name=ids,
            labels={"x": "Dim1", "y": "Dim2", "z": "Dim3"},
            title="3D Plot",
        )
        fig.update_traces(marker=dict(size=3, opacity=0.6))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        if output_file is not None:
            fig.write_html(output_file)
        fig.show()
