This keeps `KNN` as the classifier, but replaces the old handcrafted image features with embeddings from a pretrained `ResNet18`.

Run it from the repo root with a Python environment that has:

- `torch`
- `torchvision`
- `pillow`
- `numpy`

```bash
python project_code/knn_cnn_project.py \
  --train-zip /path/to/Train.zip \
  --val-zip /path/to/Validation.zip
```

Live demo after the first run:

```bash
python project_code/knn_cnn_project.py \
  --train-zip /path/to/Train.zip \
  --val-zip /path/to/Validation.zip \
  --predict-image knn_cnn_outputs/demo_images/example.jpg
```

GUI demo:

```bash
python project_code/knn_cnn_project.py \
  --train-zip /path/to/Train.zip \
  --val-zip /path/to/Validation.zip \
  --demo-gui \
  --predict-image knn_cnn_outputs/demo_images/example.jpg
```

Outputs go to:

- `knn_cnn_outputs/`
- `knn_cnn_cache/`

Notes:

- The scripts expect the train and validation image dataset to be provided separately.
- The full dataset zips and model cache files are intentionally not committed in this repo.
