Use this with a Python environment that already has:

- `numpy`
- `opencv-python`

```bash
python project_code/knn_image_project.py \
  --train-zip /path/to/Train.zip \
  --val-zip /path/to/Validation.zip
```

What it does:

- Reads the train and validation image dataset directly from the zip files in `Downloads`
- Extracts simple image features with OpenCV
- Runs KNN on a class-balanced sample for a faster demo
- Uses stronger texture and shape features plus PCA before KNN
- Picks the best `k` from the validation split
- Saves metrics, predictions, and demo images to `knn_outputs/`

Useful commands:

```bash
python project_code/knn_image_project.py \
  --train-zip /path/to/Train.zip \
  --val-zip /path/to/Validation.zip
```

```bash
python project_code/knn_image_project.py \
  --train-zip /path/to/Train.zip \
  --val-zip /path/to/Validation.zip \
  --predict-image knn_outputs/demo_images/example.jpg
```

The first run builds feature caches, so later runs should be faster.
