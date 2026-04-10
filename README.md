# KNN CNN Plant Disease Project

This repo contains:

- the GitHub Pages demo site
- the project code

## Main model

The main model is:

- `KNN` classifier
- pretrained `ResNet18` used as the feature extractor

## Files

- `index.html`
- `knn_cnn_demo_pool.js`
- `knn_cnn_outputs/demo_pool/...`
- `project_code/knn_cnn_project.py`
- `project_code/knn_image_project.py`

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

The project uses a plant disease and pest image dataset with 42 classes across:

- cotton
- wheat
- rice
- maize
- sugarcane

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
