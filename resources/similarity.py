import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import pickle
import sqlite3

try:
    from image_features import extract_image_details, load_embedding_model
    from database_operations import get_result_paths
    from visualization import print_images
    from data_processing import load_pickles
except ModuleNotFoundError:
    pass

try:
    from .image_features import extract_image_details, load_embedding_model
    from .database_operations import get_result_paths
    from .visualization import print_images
    from .data_processing import load_pickles
except ImportError:
    pass




def calculate_mean_similarity(df_input_measurements, df_comparison_data, similarity_function, best_n):
    """
    Calculates the mean similarity between input and comparison images.

    Args:
        df_input_measurements (pd.DataFrame): DataFrame of input image features.
        df_comparison_data (pd.DataFrame): DataFrame of comparison image features.
        similarity_function (str): Similarity metric (e.g., 'euclidean', 'cosine').
        best_n (int): Number of top similar images to return.

    Returns:
        list: List of IDs for the most similar images.
    """
    input_ids = df_input_measurements["ID"].values
    comparison_ids = df_comparison_data["ID"].values

    input_histogram_column = df_input_measurements.drop(columns=["ID"]).columns[0]
    comparison_histogram_column = df_comparison_data.drop(columns=["ID"]).columns[0]

    input_features = np.vstack(df_input_measurements[input_histogram_column].values)
    comparison_features = np.vstack(df_comparison_data[comparison_histogram_column].values)

    similarity_matrix = cdist(comparison_features, input_features, metric=similarity_function)
    similarity_results = pd.DataFrame(similarity_matrix, columns=input_ids)

    similarity_results["ID"] = df_comparison_data["ID"]
    similarity_results = similarity_results[["ID"] + list(input_ids)]

    if len(input_ids) > 1:
        similarity_results["Mean"] = similarity_results.drop(columns=["ID"]).mean(axis=1)
        sorted_results = similarity_results.sort_values(by="Mean", ascending=True)
    else:
        sorted_results = similarity_results.sort_values(by=input_ids[0], ascending=True)

    best_ids = sorted_results.head(best_n)["ID"].tolist()
    return best_ids

def find_similar_ids(measurement, similarity, df_input, best_n):
    """
    Finds the IDs of images that are most similar to the input image.

    Args:
        measurement (str): Type of measurement to use ('RGB', 'HSV', or 'Embedding').
        similarity (str): Similarity metric to use ('euclidean', 'manhattan', 'cosine').
        df_input (pd.DataFrame): DataFrame containing features of the input image(s).
        best_n (int): Number of top similar images to return.

    Returns:
        list: List of IDs for the most similar images.
    """
    rgb_df, hsv_df, embedding_df, path_df, other_data_df = load_pickles()
    
    similarity_functions = {"euclidean": "euclidean", "manhattan": "cityblock", "cosine": "cosine"}

    histogram_columns = {"RGB": "RGB_Histogram", "HSV": "HSV_Histogram", "Embedding": "Model_Embedding"}

    dataframes = {"RGB": rgb_df, "HSV": hsv_df, "Embedding": embedding_df}

    # Create new input-df with the needed column
    df_input_selected = df_input[["ID", histogram_columns[measurement]]]

    # Select needed comparison-df
    target_df = dataframes[measurement]

    # Select comparison function
    similarity_function = similarity_functions[similarity]

    similarity_ids = calculate_mean_similarity(df_input_selected, target_df, similarity_function, best_n)
    return similarity_ids


# Required because removing entries with 'none' (during extraction) sometimes causes IDs to be missing
def correct_data():

    """
    Corrects mismatched IDs in the dataframes and saves the corrected data.

    This function ensures that the IDs in the dataframes are sequential and consistent.
    It checks for mismatches and corrects them, then saves the corrected data back to 
    the pickle files.

    Returns:
        None
    """
        
    rgb_df, hsv_df, embedding_df, path_df, other_data_df = load_pickles()

    dataframes = [
        (rgb_df, "RGB_Hist.pkl"),
        (hsv_df, "HSV_Hist.pkl"),
        (embedding_df, "Embedding.pkl"),
        (path_df, "Path.pkl"),
        (other_data_df, "Other_data.pkl"),
    ]

    mismatch_found = False

    # Corrects mismatches from the mismatch-position until all mismatches are gone
    for df, filename in dataframes:
        corrected = False
        while True:
            # IDs are index + 1 always
            expected_ids = df.index + 1

            mismatch_index = (df["ID"] != expected_ids).idxmax()

            # Decide if there was a mismatch
            if df.loc[mismatch_index, "ID"] == expected_ids[mismatch_index]:
                if corrected:
                    print(f"Corrected: {filename}")

                else:
                    print(f"No mismatch: {filename}")

                break

            # Reduce all IDs - 1 beginning from the mismatch
            df.loc[mismatch_index:, "ID"] -= 1
            corrected = True

        # If no mismatch was found in the first file, exit the loop
        if not corrected and not mismatch_found:
            print(f"No mismatch found in {filename}. Skipping remaining files.")
            break

def main_finding_similarities(input_images_number, measurement, similarity, best_n):

    """
    Main function to find and display the most similar images based on input criteria.

    Args:
        input_images_number (int): Number of input images to consider.
        measurement (str): Type of measurement to use ('RGB', 'HSV', or 'Embedding').
        similarity (str): Similarity metric to use ('euclidean', 'manhattan', 'cosine').
        best_n (int): Number of top similar images to return.

    Returns:
        None
    """

    rgb_df, hsv_df, embedding_df, path_df, other_data_df = load_pickles()

    # Load pickles (doing this outside is better for perform the main more than one time)
    # rgb_df, hsv_df, embedding_df, path_df, other_data_df = load_pickles()

    specific_image_path = [r"C:\Users\timsa\Desktop\sample_pictures\testing\test_image_1.jpg"]

    all_image_paths = [
        r"C:\Users\timsa\Desktop\sample_pictures\testing\test_image_1.jpg",
        r"C:\Users\timsa\Desktop\sample_pictures\testing\test_image_2.jpg",
        r"C:\Users\timsa\Desktop\sample_pictures\testing\test_image_3.jpg",
        r"C:\Users\timsa\Desktop\sample_pictures\testing\test_image_4.jpg",
        r"C:\Users\timsa\Desktop\sample_pictures\testing\test_image_5.jpg",
        r"C:\Users\timsa\Desktop\sample_pictures\testing\test_image_6.jpg",
    ]

    # Decide which image(s)
    if input_images_number == 1:
        input_images = specific_image_path
    else:
        input_images = all_image_paths[:input_images_number]

    # print(input_images)
    resize_size = (224, 224)
    max_id = path_df["ID"].max()

    load_embedding_model()

    current_id = max_id + 1  # Start ID from the maximum existing ID + 1
    details_list = []

    for i, path in enumerate(input_images):
        features = extract_image_details(current_id, path, resize_size)
        if features is not None:
            details_list.append(features)
            current_id += 1  # Increase only if an image has been successfully read

    df_input = pd.DataFrame(details_list)
    # print(df_input)

    id_list = find_similar_ids(measurement, similarity, df_input, best_n)
    # print(id_list)

    conn = sqlite3.connect("database/bd_database.db")
    curs = conn.cursor()

    result_paths_list = get_result_paths(curs, id_list)
    # print(result_paths_list)

    print_images(df_input, result_paths_list, id_list, best_n)