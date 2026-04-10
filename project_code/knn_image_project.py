import argparse
import csv
import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_ZIP = Path("/Users/jameswidner/Downloads/Train-20260410T182813Z-3-001.zip")
DEFAULT_VAL_ZIP = Path("/Users/jameswidner/Downloads/Validation-20260410T182812Z-3-001.zip")
DEFAULT_CACHE_DIR = ROOT_DIR / "knn_cache"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "knn_outputs"
FEATURE_PIPELINE_VERSION = "v2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image KNN pipeline for the plant disease dataset."
    )
    parser.add_argument("--train-zip", type=Path, default=DEFAULT_TRAIN_ZIP)
    parser.add_argument("--val-zip", type=Path, default=DEFAULT_VAL_ZIP)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=120,
        help="Use at most this many training images per class for faster KNN.",
    )
    parser.add_argument(
        "--max-val-per-class",
        type=int,
        default=30,
        help="Use at most this many validation images per class for faster evaluation.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Resize the image to this square size before extracting color features.",
    )
    parser.add_argument(
        "--gray-size",
        type=int,
        default=24,
        help="Resize grayscale texture features to this square size.",
    )
    parser.add_argument(
        "--edge-size",
        type=int,
        default=16,
        help="Resize edge map features to this square size.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=160,
        help="Reduce the feature vector to this many dimensions before KNN.",
    )
    parser.add_argument(
        "--k-values",
        default="1,3,5",
        help="Comma-separated k values to test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used during vectorized distance computation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--predict-image",
        type=Path,
        help="Optional local image path to classify after training.",
    )
    parser.add_argument(
        "--demo-count",
        type=int,
        default=8,
        help="Number of validation images to include in the demo output.",
    )
    parser.add_argument(
        "--distance-weighted",
        action="store_true",
        default=True,
        help="Use inverse-distance weighted voting. Enabled by default.",
    )
    parser.add_argument(
        "--no-distance-weighted",
        dest="distance_weighted",
        action="store_false",
        help="Disable inverse-distance weighted voting.",
    )
    return parser.parse_args()


def parse_k_values(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("No k values were provided.")
    return values


def list_zip_records(zip_path):
    records = []
    with ZipFile(zip_path) as archive:
        for name in archive.namelist():
            parts = name.split("/")
            if len(parts) >= 3 and not name.endswith("/"):
                records.append({"member": name, "label": parts[1]})
    return records


def sample_records(records, max_per_class, seed):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)

    rng = random.Random(seed)
    sampled = []
    for label in sorted(grouped):
        items = grouped[label][:]
        rng.shuffle(items)
        if max_per_class is not None:
            items = items[:max_per_class]
        sampled.extend(items)
    return sampled


def load_image_bytes_from_zip(archive, member_name):
    raw = archive.read(member_name)
    image_array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {member_name}")
    return image


def extract_features_from_image(image, image_size, gray_size, edge_size):
    # The classifier is still KNN, but better features matter a lot because
    # KNN only knows how to compare points in feature space.
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    h_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    color_hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    color_hist /= max(color_hist.sum(), 1.0)

    gray_small = cv2.resize(gray, (gray_size, gray_size), interpolation=cv2.INTER_AREA)
    gray_small = gray_small.astype(np.float32).flatten() / 255.0

    edges = cv2.Canny(gray, 80, 160)
    edge_small = cv2.resize(edges, (edge_size, edge_size), interpolation=cv2.INTER_AREA)
    edge_small = edge_small.astype(np.float32).flatten() / 255.0

    # HOG adds a stronger description of local shape and vein/leaf structure.
    hog_window = (64, 64)
    gray_for_hog = cv2.resize(gray, hog_window, interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(
        _winSize=hog_window,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    hog_features = hog.compute(gray_for_hog).flatten().astype(np.float32)
    hog_features /= max(np.linalg.norm(hog_features), 1e-6)

    # LBP-style texture is useful because many plant diseases show up as
    # repeated local texture changes instead of only color changes.
    gray_lbp = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    lbp = compute_lbp(gray_lbp)
    lbp_hist, _ = np.histogram(lbp, bins=16, range=(0, 256))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= max(lbp_hist.sum(), 1.0)

    channel_means = resized.mean(axis=(0, 1)).astype(np.float32) / 255.0
    channel_stds = resized.std(axis=(0, 1)).astype(np.float32) / 255.0

    return np.concatenate(
        [
            color_hist,
            gray_small,
            edge_small,
            hog_features,
            lbp_hist,
            channel_means,
            channel_stds,
        ]
    )


def compute_lbp(gray):
    center = gray[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]
    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)
    return lbp


def cache_key(zip_path, records, image_size, gray_size, edge_size):
    record_names = "|".join(record["member"] for record in records)
    payload = (
        f"{FEATURE_PIPELINE_VERSION}|{zip_path.resolve()}|{zip_path.stat().st_mtime}|{len(records)}|"
        f"{image_size}|{gray_size}|{edge_size}|{record_names}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_or_build_zip_features(zip_path, records, cache_dir, image_size, gray_size, edge_size):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(zip_path, records, image_size, gray_size, edge_size)
    cache_path = cache_dir / f"{zip_path.stem}_{key}.npz"

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return (
            cached["features"].astype(np.float32),
            cached["labels"].tolist(),
            cached["members"].tolist(),
            cache_path,
            True,
        )

    features = []
    labels = []
    members = []

    with ZipFile(zip_path) as archive:
        for record in records:
            image = load_image_bytes_from_zip(archive, record["member"])
            feature_vector = extract_features_from_image(
                image,
                image_size=image_size,
                gray_size=gray_size,
                edge_size=edge_size,
            )
            features.append(feature_vector)
            labels.append(record["label"])
            members.append(record["member"])

    feature_matrix = np.vstack(features).astype(np.float32)
    np.savez_compressed(
        cache_path,
        features=feature_matrix,
        labels=np.array(labels, dtype=object),
        members=np.array(members, dtype=object),
    )
    return feature_matrix, labels, members, cache_path, False


def fit_standardizer(train_X):
    means = train_X.mean(axis=0)
    stds = train_X.std(axis=0)
    stds[stds == 0] = 1.0
    return means, stds


def apply_standardizer(X, means, stds):
    return (X - means) / stds


def label_to_int(labels):
    classes = sorted(set(labels))
    mapping = {label: idx for idx, label in enumerate(classes)}
    return mapping, classes


def knn_predict(train_X, train_y_idx, test_X, k, batch_size, distance_weighted):
    predictions = []
    train_norms = np.sum(train_X * train_X, axis=1)

    for start in range(0, len(test_X), batch_size):
        stop = start + batch_size
        batch = test_X[start:stop]
        batch_norms = np.sum(batch * batch, axis=1, keepdims=True)
        distances_sq = batch_norms + train_norms[None, :] - 2.0 * batch @ train_X.T
        distances_sq = np.maximum(distances_sq, 0.0)

        nearest_indices = np.argpartition(distances_sq, kth=k - 1, axis=1)[:, :k]
        nearest_distances = np.take_along_axis(distances_sq, nearest_indices, axis=1)
        order = np.argsort(nearest_distances, axis=1)
        nearest_indices = np.take_along_axis(nearest_indices, order, axis=1)
        nearest_distances = np.take_along_axis(nearest_distances, order, axis=1)

        for row, row_distances in zip(nearest_indices, nearest_distances):
            if distance_weighted:
                votes = defaultdict(float)
                for index, distance_sq in zip(row, row_distances):
                    votes[train_y_idx[index]] += 1.0 / (math.sqrt(distance_sq) + 1e-6)
                predictions.append(max(votes.items(), key=lambda item: item[1])[0])
            else:
                votes = Counter(train_y_idx[index] for index in row)
                predictions.append(votes.most_common(1)[0][0])
    return np.array(predictions, dtype=np.int32)


def fit_pca(train_X, pca_dim):
    if pca_dim <= 0 or pca_dim >= train_X.shape[1]:
        return None, None
    mean, eigenvectors = cv2.PCACompute(train_X.astype(np.float32), mean=None, maxComponents=pca_dim)
    return mean, eigenvectors


def apply_pca(X, pca_mean, pca_eigenvectors):
    if pca_mean is None or pca_eigenvectors is None:
        return X
    return cv2.PCAProject(X.astype(np.float32), pca_mean, pca_eigenvectors)


def accuracy_score(actual_idx, predicted_idx):
    return float(np.mean(actual_idx == predicted_idx))


def per_class_metrics(actual_idx, predicted_idx, class_names):
    rows = []
    for class_idx, class_name in enumerate(class_names):
        tp = int(np.sum((actual_idx == class_idx) & (predicted_idx == class_idx)))
        fp = int(np.sum((actual_idx != class_idx) & (predicted_idx == class_idx)))
        fn = int(np.sum((actual_idx == class_idx) & (predicted_idx != class_idx)))
        support = int(np.sum(actual_idx == class_idx))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        rows.append(
            {
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": support,
            }
        )
    return rows


def save_metrics_csv(path, metric_rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "precision", "recall", "f1_score", "support"],
        )
        writer.writeheader()
        writer.writerows(metric_rows)


def save_predictions_csv(path, members, actual_idx, predicted_idx, class_names):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "actual_label", "predicted_label"])
        for member, actual, predicted in zip(members, actual_idx, predicted_idx):
            writer.writerow([member, class_names[actual], class_names[predicted]])


def save_demo_csv(path, demo_rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "actual_label", "predicted_label", "correct"],
        )
        writer.writeheader()
        writer.writerows(demo_rows)


def save_model_cache(path, train_X, train_y_idx, class_names, means, stds, pca_mean, pca_eigenvectors, best_k):
    np.savez_compressed(
        path,
        train_X=train_X.astype(np.float32),
        train_y_idx=train_y_idx.astype(np.int32),
        class_names=np.array(class_names, dtype=object),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
        pca_mean=np.array([] if pca_mean is None else pca_mean, dtype=np.float32),
        pca_eigenvectors=np.array([] if pca_eigenvectors is None else pca_eigenvectors, dtype=np.float32),
        best_k=np.array([best_k], dtype=np.int32),
    )


def load_model_cache(path):
    cached = np.load(path, allow_pickle=True)
    pca_mean = cached["pca_mean"]
    pca_eigenvectors = cached["pca_eigenvectors"]
    if pca_mean.size == 0:
        pca_mean = None
    if pca_eigenvectors.size == 0:
        pca_eigenvectors = None
    return {
        "train_X": cached["train_X"].astype(np.float32),
        "train_y_idx": cached["train_y_idx"].astype(np.int32),
        "class_names": cached["class_names"].tolist(),
        "means": cached["means"].astype(np.float32),
        "stds": cached["stds"].astype(np.float32),
        "pca_mean": pca_mean,
        "pca_eigenvectors": pca_eigenvectors,
        "best_k": int(cached["best_k"][0]),
    }


def print_top_metrics(metric_rows, limit=12):
    ranked = sorted(metric_rows, key=lambda row: (-row["f1_score"], -row["support"], row["class_name"]))
    print("\nTop per-class F1 scores")
    for row in ranked[:limit]:
        print(
            f"  {row['class_name']:<28} "
            f"precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} "
            f"f1={row['f1_score']:.3f} "
            f"support={row['support']}"
        )


def export_demo_images(val_zip, members, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    with ZipFile(val_zip) as archive:
        for member in members:
            filename = Path(member).name
            label = member.split("/")[1]
            safe_label = label.replace("/", "_").replace(" ", "_")
            out_path = output_dir / f"{safe_label}__{filename}"
            out_path.write_bytes(archive.read(member))
            exported.append(out_path)
    return exported


def predict_single_image(image_path, train_X, train_y_idx, class_names, means, stds, pca_mean, pca_eigenvectors, k):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read local image: {image_path}")

    feature_vector = extract_features_from_image(
        image,
        image_size=args.image_size,
        gray_size=args.gray_size,
        edge_size=args.edge_size,
    )
    feature_vector = apply_standardizer(feature_vector[None, :], means, stds)
    feature_vector = apply_pca(feature_vector, pca_mean, pca_eigenvectors)
    prediction_idx = knn_predict(
        train_X,
        train_y_idx,
        feature_vector,
        k,
        batch_size=1,
        distance_weighted=args.distance_weighted,
    )[0]
    return class_names[prediction_idx]


def main():
    global args
    args = parse_args()
    k_values = parse_k_values(args.k_values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_cache_path = args.output_dir / "knn_model_cache.npz"

    if args.predict_image and model_cache_path.exists():
        cached_model = load_model_cache(model_cache_path)
        predicted_label = predict_single_image(
            args.predict_image,
            cached_model["train_X"],
            cached_model["train_y_idx"],
            cached_model["class_names"],
            cached_model["means"],
            cached_model["stds"],
            cached_model["pca_mean"],
            cached_model["pca_eigenvectors"],
            cached_model["best_k"],
        )
        print("Image KNN Project")
        print(f"Loaded cached model: {model_cache_path}")
        print("\nSingle image prediction")
        print(f"  Image: {args.predict_image}")
        print(f"  Predicted label: {predicted_label}")
        return

    print("Image KNN Project")
    print(f"Train zip: {args.train_zip}")
    print(f"Validation zip: {args.val_zip}")

    train_records = list_zip_records(args.train_zip)
    val_records = list_zip_records(args.val_zip)

    sampled_train = sample_records(train_records, args.max_train_per_class, args.seed)
    sampled_val = sample_records(val_records, args.max_val_per_class, args.seed)

    print(f"Training images selected: {len(sampled_train)}")
    print(f"Validation images selected: {len(sampled_val)}")
    print(f"Classes: {len(set(record['label'] for record in train_records))}")

    train_X_raw, train_labels, _, train_cache, train_cached = load_or_build_zip_features(
        args.train_zip,
        sampled_train,
        args.cache_dir,
        args.image_size,
        args.gray_size,
        args.edge_size,
    )
    val_X_raw, val_labels, val_members, val_cache, val_cached = load_or_build_zip_features(
        args.val_zip,
        sampled_val,
        args.cache_dir,
        args.image_size,
        args.gray_size,
        args.edge_size,
    )

    print(f"Train feature cache: {train_cache} ({'loaded' if train_cached else 'built'})")
    print(f"Validation feature cache: {val_cache} ({'loaded' if val_cached else 'built'})")

    class_map, class_names = label_to_int(train_labels + val_labels)
    train_y_idx = np.array([class_map[label] for label in train_labels], dtype=np.int32)
    val_y_idx = np.array([class_map[label] for label in val_labels], dtype=np.int32)

    means, stds = fit_standardizer(train_X_raw)
    train_X = apply_standardizer(train_X_raw, means, stds).astype(np.float32)
    val_X = apply_standardizer(val_X_raw, means, stds).astype(np.float32)
    pca_mean, pca_eigenvectors = fit_pca(train_X, args.pca_dim)
    train_X = apply_pca(train_X, pca_mean, pca_eigenvectors).astype(np.float32)
    val_X = apply_pca(val_X, pca_mean, pca_eigenvectors).astype(np.float32)

    # I tune k on the validation split here because the dataset already came
    # with train and validation folders, which makes the workflow simpler.
    scores = {}
    for k in k_values:
        predicted_idx = knn_predict(
            train_X,
            train_y_idx,
            val_X,
            k,
            args.batch_size,
            args.distance_weighted,
        )
        scores[k] = accuracy_score(val_y_idx, predicted_idx)

    best_k = max(scores.items(), key=lambda item: (item[1], -item[0]))[0]

    print("\nValidation accuracy by k")
    for k in k_values:
        print(f"  k={k:<2} -> {scores[k]:.4f}")
    print(f"Best k: {best_k}")

    predicted_idx = knn_predict(
        train_X,
        train_y_idx,
        val_X,
        best_k,
        args.batch_size,
        args.distance_weighted,
    )
    overall_accuracy = accuracy_score(val_y_idx, predicted_idx)
    metric_rows = per_class_metrics(val_y_idx, predicted_idx, class_names)

    metrics_path = args.output_dir / "knn_per_class_metrics.csv"
    predictions_path = args.output_dir / "knn_validation_predictions.csv"
    summary_path = args.output_dir / "knn_summary.txt"
    demo_csv_path = args.output_dir / "knn_demo_predictions.csv"
    demo_dir = args.output_dir / "demo_images"
    model_cache_path = args.output_dir / "knn_model_cache.npz"

    save_metrics_csv(metrics_path, metric_rows)
    save_predictions_csv(predictions_path, val_members, val_y_idx, predicted_idx, class_names)
    save_model_cache(
        model_cache_path,
        train_X,
        train_y_idx,
        class_names,
        means,
        stds,
        pca_mean,
        pca_eigenvectors,
        best_k,
    )

    demo_count = min(args.demo_count, len(val_members))
    demo_indices = random.Random(args.seed).sample(range(len(val_members)), demo_count)
    demo_rows = []
    demo_members = []
    for index in demo_indices:
        demo_members.append(val_members[index])
        demo_rows.append(
            {
                "image_path": val_members[index],
                "actual_label": class_names[val_y_idx[index]],
                "predicted_label": class_names[predicted_idx[index]],
                "correct": class_names[val_y_idx[index]] == class_names[predicted_idx[index]],
            }
        )
    save_demo_csv(demo_csv_path, demo_rows)
    exported_demo_paths = export_demo_images(args.val_zip, demo_members, demo_dir)

    print(f"\nOverall validation accuracy: {overall_accuracy:.4f}")
    print_top_metrics(metric_rows)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Image KNN Summary\n")
        handle.write(f"Overall validation accuracy: {overall_accuracy:.4f}\n")
        handle.write(f"Best k: {best_k}\n")
        handle.write(f"Distance weighted voting: {args.distance_weighted}\n")
        handle.write(f"PCA dimensions: {args.pca_dim}\n")
        handle.write(f"Training images selected: {len(sampled_train)}\n")
        handle.write(f"Validation images selected: {len(sampled_val)}\n")
        handle.write(f"Train feature cache: {train_cache}\n")
        handle.write(f"Validation feature cache: {val_cache}\n")
        handle.write(f"Metrics CSV: {metrics_path}\n")
        handle.write(f"Predictions CSV: {predictions_path}\n")
        handle.write(f"Demo CSV: {demo_csv_path}\n")
        handle.write(f"Model cache: {model_cache_path}\n")
        handle.write("Demo images:\n")
        for path in exported_demo_paths:
            handle.write(f"  {path}\n")

    print("\nOutput files")
    print(f"  Summary: {summary_path}")
    print(f"  Metrics CSV: {metrics_path}")
    print(f"  Validation predictions: {predictions_path}")
    print(f"  Demo predictions: {demo_csv_path}")
    print(f"  Demo images folder: {demo_dir}")
    print(f"  Model cache: {model_cache_path}")

    if args.predict_image:
        predicted_label = predict_single_image(
            args.predict_image,
            train_X,
            train_y_idx,
            class_names,
            means,
            stds,
            pca_mean,
            pca_eigenvectors,
            best_k,
        )
        print("\nSingle image prediction")
        print(f"  Image: {args.predict_image}")
        print(f"  Predicted label: {predicted_label}")


if __name__ == "__main__":
    main()
