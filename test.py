import pytest
import os
import numpy as np
import pandas as pd
import sqlite3
from resources.data_processing import find_image_files, load_checkpoint, save_checkpoint, load_pickles, main_load_images
from resources.image_features import image_rgb_calculation, image_hsv_calculation, load_embedding_model, extract_image_details
from resources.database_operations import create_database, save_to_db, get_result_paths
from resources.similarity import calculate_mean_similarity
from resources.visualization import print_images, plot_selected_images, plot_dimensionality_reduction
from resources.dimensionality import reduce_dimensionality, create_clusters, save_dimensionality_results

# Setup dummy data before running tests
@pytest.fixture(scope="module", autouse=True)
def setup_dummy_data():
    # Create a resources folder if it doesn't exist
    os.makedirs("resources", exist_ok=True)

    # Create dummy RGB_Hist.pkl
    dummy_rgb_data = pd.DataFrame({"ID": [1, 2, 3], "Histogram": [np.zeros(512), np.zeros(512), np.zeros(512)]})
    dummy_rgb_data.to_pickle("resources/RGB_Hist.pkl")

    # Create dummy HSV_Hist.pkl
    dummy_hsv_data = pd.DataFrame({"ID": [1, 2, 3], "Histogram": [np.zeros(512), np.zeros(512), np.zeros(512)]})
    dummy_hsv_data.to_pickle("resources/HSV_Hist.pkl")

    # Create dummy Embedding.pkl
    dummy_embedding_data = pd.DataFrame({"ID": [1, 2, 3], "Embedding": [np.zeros(1280), np.zeros(1280), np.zeros(1280)]})
    dummy_embedding_data.to_pickle("resources/Embedding.pkl")

    # Create dummy Path.pkl
    dummy_path_data = pd.DataFrame({"ID": [1, 2, 3], "Path": ["./test_images/sample1.jpg", "./test_images/sample2.jpg", "./test_images/sample3.jpg"]})
    dummy_path_data.to_pickle("resources/Path.pkl")

    # Create dummy Other_data.pkl
    dummy_other_data = pd.DataFrame({
        "ID": [1, 2, 3],
        "Average_Color": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "Brightness": [0, 0, 0],
        "Average_HSV": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "Resolution": [(224, 224), (224, 224), (224, 224)],
        "DPI": [(72, 72), (72, 72), (72, 72)],
        "File_Size": [1024, 1024, 1024],
        "File_Type": [".jpg", ".jpg", ".jpg"],
        "Metadata": [{}, {}, {}]
    })
    dummy_other_data.to_pickle("resources/Other_data.pkl")

    yield

    # Cleanup after tests
    os.remove("resources/RGB_Hist.pkl")
    os.remove("resources/HSV_Hist.pkl")
    os.remove("resources/Embedding.pkl")
    os.remove("resources/Path.pkl")
    os.remove("resources/Other_data.pkl")

# Your existing tests
def test_find_image_files():
    files = find_image_files("./test_images")
    assert isinstance(files, list)

def test_load_checkpoint_no_file():
    batch_index, paths, rgb_hists, hsv_hists, embeddings, other_data = load_checkpoint()
    assert batch_index == 0
    assert len(paths) == 0

def test_simple():
    assert 1 == 1

def test_save_and_load_checkpoint():
    save_checkpoint(1, [], [], [], [], [])
    batch_index, _, _, _, _, _ = load_checkpoint()
    assert batch_index == 1
    os.remove("checkpoint.pkl")

def test_image_rgb_calculation():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb_hist = image_rgb_calculation(img)
    assert len(rgb_hist) == 512

def test_image_hsv_calculation():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    hsv_hist = image_hsv_calculation(img)
    assert len(hsv_hist) == 512

def test_load_embedding_model():
    load_embedding_model()
    assert load_embedding_model is not None

def test_create_database():
    create_database()
    assert os.path.exists("database/bd_database.db")

def test_calculate_mean_similarity():
    df_input = pd.DataFrame({"ID": [1], "RGB_Histogram": [np.zeros(512)]})
    df_comparison = pd.DataFrame({"ID": [2], "RGB_Histogram": [np.zeros(512)]})
    result = calculate_mean_similarity(df_input, df_comparison, "euclidean", 1)
    assert len(result) == 1


def test_reduce_dimensionality_pca():
    data = np.random.rand(10, 512)
    reduced_data = reduce_dimensionality(data, "pca", 2)
    assert reduced_data.shape[1] == 2

def test_create_clusters():
    data = np.random.rand(10, 512)
    labels = create_clusters(data, 2)
    assert len(labels) == 10

def test_main_load_images(monkeypatch):
    def mock_load_embedding_model():
        pass

    def mock_find_image_files(path):
        return ["./test_images/sample.jpg"]

    def mock_image_batch_generator(image_files, batch_size, resize_size, start_index=0, show_progress=True):
        df = pd.DataFrame([{
            "ID": 1,
            "Path": "./test_images/sample.jpg",
            "RGB_Histogram": np.zeros(512),
            "HSV_Histogram": np.zeros(512),
            "Model_Embedding": np.zeros(1280),
            "Average_Color": [0, 0, 0],
            "Brightness": 0,
            "Average_HSV": [0, 0, 0],
            "Resolution": (224, 224),
            "DPI": (72, 72),
            "File_Size": 1024,
            "File_Type": ".jpg",
            "Metadata": {}
        }])
        yield df, 1000

    monkeypatch.setattr('resources.data_processing.load_embedding_model', mock_load_embedding_model)
    monkeypatch.setattr('resources.data_processing.find_image_files', mock_find_image_files)
    monkeypatch.setattr('resources.data_processing.image_batch_generator', mock_image_batch_generator)

    batch_size = 1000
    desired_size = (224, 224)
    main_load_images(batch_size, desired_size)

    assert os.path.exists("resources/Path.pkl")

if __name__ == "__main__":
    pytest.main()