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
