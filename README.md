# KNN CNN Demo Site

This repo contains two things:

1. A static GitHub Pages demo site
2. The project code used to build the KNN-based plant disease classifier

## What is in this repo

- `index.html`
- `knn_cnn_demo_pool.js`
- `knn_cnn_outputs/demo_pool/...`
- `project_code/knn_cnn_project.py`
- `project_code/knn_cnn_project_README.md`
- `project_code/knn_image_project.py`
- `project_code/knn_image_project_README.md`

## What is not in this repo

- The full training dataset zip
- The full validation dataset zip
- The cached KNN model file
- The full embedding cache files

Those were left out on purpose to keep the repo smaller and easier to share.

## Model summary

The main project version is:

- classifier: `KNN`
- feature extractor: pretrained `ResNet18`

So the pipeline is:

1. input image
2. pretrained ResNet18 embedding
3. KNN classification

## Dataset note

The local dataset used for the project is a combined multi-class plant disease / pest image dataset covering crops such as cotton, wheat, rice, maize, and sugarcane.

Based on the class names, it appears to be assembled from public agricultural disease datasets rather than coming from a single small built-in sample set.

I could not verify one exact public page for the full combined 42-class bundle just from the local archive names alone, so I do not want to overstate that.

Related public sources with overlapping class names:

- Roboflow Universe search results:
  https://universe.roboflow.com/search?q=American%20Bollworm%20on%20Cotton%20Anthracnose%20on%20Cotton%20Army%20worm%20Cotton%20Aphid%20RedRot%20sugarcane
- Roboflow search result showing matching plant disease class groups:
  https://universe.roboflow.com/search?q=moth%20class%3Aborer%20classification

## Pages

This repo is set up for GitHub Pages from the `main` branch root.
