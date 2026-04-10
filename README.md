# KNN CNN Plant Disease Project

Live demo:

- GitHub Pages: https://pi128.github.io/knn-cnn-demo-site/
- Main page file: `index.html`

This repo contains the GitHub Pages demo site and the project code.

## Main model

The main model is:

- `KNN` classifier
- pretrained `ResNet18` used as the feature extractor

## How to run

Install the Python packages:

```bash
pip install torch torchvision pillow numpy opencv-python
```

Run the main version:

```bash
python project_code/knn_cnn_project.py \
  --train-zip /path/to/Train.zip \
  --val-zip /path/to/Validation.zip
```

Run the older OpenCV feature version:

```bash
python project_code/knn_image_project.py \
  --train-zip /path/to/Train.zip \
  --val-zip /path/to/Validation.zip
```

## Dataset used

The project uses a combined plant disease and pest image dataset with 42 classes across:

- cotton
- wheat
- rice
- maize
- sugarcane

The combined dataset used for the project includes classes from these crop groups:

- cotton disease and pest images
- wheat disease and pest images
- rice disease images
- maize disease and pest images
- sugarcane disease images

Examples of classes in the dataset:

- American Bollworm on Cotton
- Anthracnose on Cotton
- Cotton Aphid
- Wheat black rust
- Wheat powdery mildew
- Rice Blast
- Tungro
- RedRot sugarcane
- Leaf Curl
- Healthy cotton

The full train and validation dataset zip files are not included in this repo.

## Dataset links

Public source pages for the datasets used in the combined dataset:

- https://universe.roboflow.com/search?q=American%20Bollworm%20on%20Cotton%20Anthracnose%20on%20Cotton%20Army%20worm%20Cotton%20Aphid%20RedRot%20sugarcane
- https://universe.roboflow.com/search?q=moth%20class%3Aborer%20classification
