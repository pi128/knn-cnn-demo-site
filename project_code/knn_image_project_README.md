Use this with the Python environment that already has `numpy` and `cv2`:

```bash
Downloads/.venv_mv/bin/python "Documents/Spring 2026/CS4267 ML/knn_image_project.py"
```

What it does:

- Reads the train and validation image dataset directly from the zip files in `Downloads`
- Extracts simple image features with OpenCV
- Runs KNN on a class-balanced sample for a faster demo
- Uses stronger texture and shape features plus PCA before KNN
- Picks the best `k` from the validation split
- Saves metrics, predictions, and demo images to `Documents/Spring 2026/CS4267 ML/knn_outputs`

Useful commands:

```bash
Downloads/.venv_mv/bin/python "Documents/Spring 2026/CS4267 ML/knn_image_project.py"
```

```bash
Downloads/.venv_mv/bin/python "Documents/Spring 2026/CS4267 ML/knn_image_project.py" \
  --predict-image "Documents/Spring 2026/CS4267 ML/knn_outputs/demo_images/Anthracnose_on_Cotton__Image_5.jpg"
```

The first run builds feature caches, so later runs should be faster.
