# KNN CNN Plant Disease Project

Live demo:

- GitHub Pages: https://pi128.github.io/knn-cnn-demo-site/
- Main page file: `index.html`

This repo contains the GitHub Pages demo site and the project code.

## Browser CNN model

The demo site also loads `tfjs_graph_model/model.json`, converted from
`/Users/jameswidner/Downloads/cnn_crop_id.keras`, and runs predictions directly
in the browser with TensorFlow.js.

The class labels are in `cnn_model_labels.js`. The Keras archive did not embed a
label map, so verify that file against the original training class order before
using the browser predictions for grading or reporting.

## Browser SVM model

The SVM panel loads `svm_model.js`, exported from
`/Users/jameswidner/Downloads/SVMDemo.zip`. It runs the downloaded RBF SVM in
JavaScript against held-out 15-feature test vectors from the zip.

The zip did not include the image-to-15-feature preprocessing code, so the SVM
panel classifies exported test vectors rather than arbitrary uploaded images.

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

Example classes in the dataset include:

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
