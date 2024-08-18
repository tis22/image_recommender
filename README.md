# Image Recommender
## Overview
This project is designed to help you find the top five images similar to a given image from a large dataset of nearly 450,000 images. By leveraging Python, this software can recommend images based on various similarity metrics, such as color histograms and embeddings. The project focuses on image processing, feature extraction, similarity calculation, and clustering.

## Getting Started
Prerequisites
Before you begin, ensure that Python is installed on your system. We recommend using Python 3.8 or higher for optimal compatibility.

### Installation
## Clone the Repository
Begin by cloning the repository to your local machine:

```bash
git clone https://github.com/yourusername/image-recommender.git
```

Navigate into the Project Directory and install the dependencies.

The necessary dependencies are:

numpy==1.24.3
pandas==2.0.3
tqdm==4.66.1
matplotlib==3.7.1
Pillow==9.4.0
opencv-python==4.7.0.72
scipy==1.11.2
sqlite==3.41.2
torch==2.0.1
torchvision==0.15.2
scikit-learn==1.3.0
umap-learn==0.5.4
plotly==5.15.0
pytest==8.3.2



### Setting Up the Dataset
Before you can start recommending images, you'll need to set up your dataset with all the necessary similarities and metadata.


#### Load and Process Images
In your main Python script or Jupyter notebook, use the following code to load and process images:

```Python
batch_size = 1000
desired_size = (224, 224)
main_load_images(batch_size, desired_size)
```
This script will process the images from your specified directory, calculate the relevant features, and save them as pickle files for later use.


### Recommending Images
Load the Pickled Data
Make sure your data is properly loaded from the pickle files:
```python
initialize_data()
```

### Find Similar Images
To find a specified number of similar images based on your chosen feature type and distance metric, use the following command:

```python
main_finding_similarities(1, "RGB", "euclidean", 5)
```
This function will compute the similarities between your input images and the dataset, displaying the top n similar images. 
You can specify the number of input images, choose between 'RGB', 'HSV' or 'Embedding' as the feature type
and select 'euclidean', 'manhattan' or 'cosine' as the distance metric.
Important: You need to specify the paths to your input images before running the function.


### Correct Data Inconsistencies
If you encounter any inconsistencies in the data, you can correct them by running:

```python
correct_data()
```

### Directory Structure
- resources/: Contains all the Python modules for image processing, feature extraction, and similarity calculations.
- main.ipynb: The main Jupyter notebook where functions are called, and processes are executed.

### Contact
For any questions or further information, feel free to contact:

- Tim Sandrock: tim.sandrock@study.hs-duesseldorf.de
- Joschua Schramm: joschua.schramm@study.hs-duesseldorf.de
