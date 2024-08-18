import os
import pickle
import pandas as pd
from tqdm.notebook import tqdm

try:
    from image_features import extract_image_details, load_embedding_model
except ModuleNotFoundError:
    pass

try:
    from .image_features import extract_image_details, load_embedding_model
except ImportError:
    pass


PATH_TO_IMAGES = r"C:\Users\timsa\Desktop\Daten_Joschua\data\image_data\extra_collection\city"


def find_image_files(root_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")):
    """
    Recursively finds all image files in the specified directory with the given extensions.

    Args:
        root_dir (str): The root directory to search for image files.
        extensions (tuple): A tuple of file extensions to include in the search.

    Returns:
        list: A list of paths to image files that match the specified extensions.
    """
    image_files = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(subdir, file))
    return image_files


def load_checkpoint():
    """
    Loads the checkpoint from a pickle file, if it exists.

    Returns:
        tuple: A tuple containing the following elements:
            - batch_index (int): The index of the last processed batch.
            - paths (list): A list of image paths.
            - rgb_hists (list): A list of RGB histograms.
            - hsv_hists (list): A list of HSV histograms.
            - embeddings (list): A list of model embeddings.
            - other_data (list): A list of other data associated with the images.
    """
    if not os.path.exists("checkpoint.pkl"):
        return 0, [], [], [], [], []

    with open("checkpoint.pkl", "rb") as f:
        batch_index, paths, rgb_hists, hsv_hists, embeddings, other_data = pickle.load(f)
        print(f"Loaded checkpoint.\nStarting from path with ID: {batch_index + 1}")
    return batch_index, paths, rgb_hists, hsv_hists, embeddings, other_data


def save_checkpoint(batch_index, paths, rgb_hists, hsv_hists, embeddings, other_data):
    """
    Saves the current progress into a checkpoint pickle file.

    Args:
        batch_index (int): The index of the current batch being processed.
        paths (list): A list of image paths.
        rgb_hists (list): A list of RGB histograms.
        hsv_hists (list): A list of HSV histograms.
        embeddings (list): A list of model embeddings.
        other_data (list): A list of other data associated with the images.

    Returns:
        None
    """
    with open("checkpoint.pkl", "wb") as f:
        pickle.dump((batch_index, paths, rgb_hists, hsv_hists, embeddings, other_data), f)


def load_pickles():
    """
    Loads the saved data from pickle files.

    Returns:
        tuple: A tuple containing the following pandas DataFrames:
            - rgb_df (pd.DataFrame): DataFrame containing RGB histograms.
            - hsv_df (pd.DataFrame): DataFrame containing HSV histograms.
            - embedding_df (pd.DataFrame): DataFrame containing model embeddings.
            - path_df (pd.DataFrame): DataFrame containing image paths.
            - other_data_df (pd.DataFrame): DataFrame containing other image data.
    """
    rgb_df = pd.read_pickle("RGB_Hist.pkl")
    hsv_df = pd.read_pickle("HSV_Hist.pkl")
    embedding_df = pd.read_pickle("Embedding.pkl")
    path_df = pd.read_pickle("Path.pkl")
    other_data_df = pd.read_pickle("Other_data.pkl")

    return rgb_df, hsv_df, embedding_df, path_df, other_data_df


def image_batch_generator(image_files, batch_size, resize_size, start_index=0, show_progress=True):
    """
    Generates batches of image data for processing.

    Args:
        image_files (list): A list of image file paths to be processed.
        batch_size (int): The number of images to process in each batch.
        resize_size (tuple): The desired size to which each image should be resized.
        start_index (int, optional): The index from which to start processing images. Defaults to 0.
        show_progress (bool, optional): Whether to display a progress bar. Defaults to True.

    Yields:
        tuple: A tuple containing the following elements:
            - df (pd.DataFrame): A DataFrame containing extracted features for the current batch of images.
            - batch_index (int): The index of the next batch to be processed.
    """
    total_batches = (len(image_files) - start_index + batch_size - 1) // batch_size
    progress_bar = tqdm(total=total_batches, desc="Processing images") if show_progress else None

    current_id = start_index + 1

    for index in range(start_index, len(image_files), batch_size):
        batch = image_files[index : index + batch_size]
        details_list = []

        for i, path in enumerate(batch):
            features = extract_image_details(current_id, path, resize_size)
            if features is not None:
                details_list.append(features)
                current_id += 1

        df = pd.DataFrame(details_list)
        yield df, index + batch_size
        if show_progress:
            progress_bar.update(1)

    if show_progress:
        progress_bar.close()


def main_load_images(batch_size, desired_size):
    """
    Main function to load images, calculate features, and save them.

    Args:
        batch_size (int): The number of images to process in each batch.
        desired_size (tuple): The desired size to which each image should be resized.

    Returns:
        None
    """
    start_index, paths, rgb_hists, hsv_hists, embeddings, other_data = load_checkpoint()

    image_paths = find_image_files(PATH_TO_IMAGES)
    load_embedding_model()

    for df, batch_index in image_batch_generator(
        image_paths, batch_size, desired_size, start_index=start_index, show_progress=True
    ):
        paths.extend(df[["ID", "Path"]].values.tolist())
        rgb_hists.extend(df[["ID", "RGB_Histogram"]].values.tolist())
        hsv_hists.extend(df[["ID", "HSV_Histogram"]].values.tolist())
        embeddings.extend(df[["ID", "Model_Embedding"]].values.tolist())
        other_data.extend(
            df.drop(columns=["Path", "RGB_Histogram", "HSV_Histogram", "Model_Embedding"]).values.tolist()
        )

        save_checkpoint(batch_index, paths, rgb_hists, hsv_hists, embeddings, other_data)

    df_paths = pd.DataFrame(paths, columns=["ID", "Path"])
    df_rgb = pd.DataFrame(rgb_hists, columns=["ID", "Histogram"])
    df_hsv = pd.DataFrame(hsv_hists, columns=["ID", "Histogram"])
    df_embeddings = pd.DataFrame(embeddings, columns=["ID", "Embedding"])
    df_other_data = pd.DataFrame(
        other_data,
        columns=[
            "ID",
            "Average_Color",
            "Brightness",
            "Average_HSV",
            "Resolution",
            "DPI",
            "File_Size",
            "File_Type",
            "Metadata",
        ],
    )

    df_paths.to_pickle("Path.pkl")
    df_rgb.to_pickle("RGB_Hist.pkl")
    df_hsv.to_pickle("HSV_Hist.pkl")
    df_embeddings.to_pickle("Embedding.pkl")
    df_other_data.to_pickle("Other_data.pkl")

    if os.path.exists("checkpoint.pkl"):
        os.remove("checkpoint.pkl")
