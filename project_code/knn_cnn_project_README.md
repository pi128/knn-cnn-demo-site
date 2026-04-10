This keeps `KNN` as the classifier, but replaces the old handcrafted image features with embeddings from a pretrained `ResNet18`.

Run it with the environment that has `torch` and `torchvision`:

```bash
Documents/Fall2025/CS3642/demo_venv/bin/python "Documents/Spring 2026/CS4267 ML/knn_cnn_project.py"
```

Live demo after the first run:

```bash
Documents/Fall2025/CS3642/demo_venv/bin/python "Documents/Spring 2026/CS4267 ML/knn_cnn_project.py" \
  --predict-image "Documents/Spring 2026/CS4267 ML/knn_cnn_outputs/demo_images/Cotton_Aphid__10.jpg"
```

GUI demo:

```bash
Documents/Fall2025/CS3642/demo_venv/bin/python "Documents/Spring 2026/CS4267 ML/knn_cnn_project.py" \
  --demo-gui \
  --predict-image "Documents/Spring 2026/CS4267 ML/knn_outputs/demo_images/Cotton_Aphid__10.jpg"
```

Outputs go to:

- `Documents/Spring 2026/CS4267 ML/knn_cnn_outputs`
- `Documents/Spring 2026/CS4267 ML/knn_cnn_cache`
