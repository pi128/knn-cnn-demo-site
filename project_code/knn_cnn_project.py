import argparse
import csv
import hashlib
import io
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

# Keep the local demo on macOS from crashing in OpenMP initialization.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from torchvision.models import ResNet18_Weights, resnet18


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_ZIP = Path("/Users/jameswidner/Downloads/Train-20260410T182813Z-3-001.zip")
DEFAULT_VAL_ZIP = Path("/Users/jameswidner/Downloads/Validation-20260410T182812Z-3-001.zip")
DEFAULT_CACHE_DIR = ROOT_DIR / "knn_cnn_cache"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "knn_cnn_outputs"
PIPELINE_VERSION = "cnn_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="KNN classifier using pretrained CNN embeddings."
    )
    parser.add_argument("--train-zip", type=Path, default=DEFAULT_TRAIN_ZIP)
    parser.add_argument("--val-zip", type=Path, default=DEFAULT_VAL_ZIP)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-train-per-class", type=int, default=120)
    parser.add_argument("--max-val-per-class", type=int, default=30)
    parser.add_argument("--k-values", default="1,3,5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-count", type=int, default=8)
    parser.add_argument("--predict-image", type=Path)
    parser.add_argument("--demo-gui", action="store_true")
    parser.add_argument("--distance-weighted", action="store_true", default=True)
    parser.add_argument("--no-distance-weighted", dest="distance_weighted", action="store_false")
    return parser.parse_args()


def parse_k_values(text):
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("No k values provided.")
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
        sampled.extend(items[:max_per_class] if max_per_class is not None else items)
    return sampled


def build_backbone():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, weights.transforms()


def cache_key(zip_path, records):
    names = "|".join(record["member"] for record in records)
    payload = f"{PIPELINE_VERSION}|{zip_path.resolve()}|{zip_path.stat().st_mtime}|{len(records)}|{names}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_or_build_embeddings(zip_path, records, cache_dir, model, transform, batch_size):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{zip_path.stem}_{cache_key(zip_path, records)}.npz"
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
    tensors = []
    pending_labels = []
    pending_members = []

    with ZipFile(zip_path) as archive:
        for record in records:
            image_bytes = archive.read(record["member"])
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensors.append(transform(image))
            pending_labels.append(record["label"])
            pending_members.append(record["member"])

            if len(tensors) >= batch_size:
                batch_features = encode_batch(model, tensors)
                features.append(batch_features)
                labels.extend(pending_labels)
                members.extend(pending_members)
                tensors = []
                pending_labels = []
                pending_members = []

    if tensors:
        batch_features = encode_batch(model, tensors)
        features.append(batch_features)
        labels.extend(pending_labels)
        members.extend(pending_members)

    feature_matrix = np.vstack(features).astype(np.float32)
    np.savez_compressed(
        cache_path,
        features=feature_matrix,
        labels=np.array(labels, dtype=object),
        members=np.array(members, dtype=object),
    )
    return feature_matrix, labels, members, cache_path, False


def encode_batch(model, tensors):
    batch = torch.stack(tensors)
    with torch.no_grad():
        embeddings = model(batch).cpu().numpy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


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


def knn_prediction_details(train_X, train_y_idx, test_vector, class_names, k, distance_weighted):
    train_norms = np.sum(train_X * train_X, axis=1)
    test_norm = np.sum(test_vector * test_vector, axis=1, keepdims=True)
    distances_sq = test_norm + train_norms[None, :] - 2.0 * test_vector @ train_X.T
    distances_sq = np.maximum(distances_sq, 0.0)

    nearest_indices = np.argpartition(distances_sq, kth=k - 1, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(distances_sq, nearest_indices, axis=1)
    order = np.argsort(nearest_distances, axis=1)
    nearest_indices = np.take_along_axis(nearest_indices, order, axis=1)[0]
    nearest_distances = np.take_along_axis(nearest_distances, order, axis=1)[0]

    if distance_weighted:
        votes = defaultdict(float)
        for index, distance_sq in zip(nearest_indices, nearest_distances):
            votes[train_y_idx[index]] += 1.0 / (math.sqrt(distance_sq) + 1e-6)
        winning_label_idx, winning_score = max(votes.items(), key=lambda item: item[1])
        total_score = sum(votes.values()) or 1.0
        vote_share = winning_score / total_score
        vote_breakdown = sorted(
            ((class_names[label_idx], score / total_score) for label_idx, score in votes.items()),
            key=lambda item: item[1],
            reverse=True,
        )
    else:
        counts = Counter(train_y_idx[index] for index in nearest_indices)
        winning_label_idx, winning_score = counts.most_common(1)[0]
        total_score = sum(counts.values()) or 1.0
        vote_share = winning_score / total_score
        vote_breakdown = sorted(
            ((class_names[label_idx], count / total_score) for label_idx, count in counts.items()),
            key=lambda item: item[1],
            reverse=True,
        )

    neighbor_labels = [class_names[train_y_idx[index]] for index in nearest_indices]
    return {
        "predicted_label": class_names[winning_label_idx],
        "vote_share": vote_share,
        "neighbor_labels": neighbor_labels,
        "neighbor_distances": [math.sqrt(value) for value in nearest_distances.tolist()],
        "vote_breakdown": vote_breakdown,
    }


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
        writer = csv.DictWriter(handle, fieldnames=["class_name", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        writer.writerows(metric_rows)


def save_predictions_csv(path, members, actual_idx, predicted_idx, class_names):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "actual_label", "predicted_label"])
        for member, actual, predicted in zip(members, actual_idx, predicted_idx):
            writer.writerow([member, class_names[actual], class_names[predicted]])


def save_model_cache(path, train_X, train_y_idx, class_names, best_k):
    np.savez_compressed(
        path,
        train_X=train_X.astype(np.float32),
        train_y_idx=train_y_idx.astype(np.int32),
        class_names=np.array(class_names, dtype=object),
        best_k=np.array([best_k], dtype=np.int32),
    )


def load_model_cache(path):
    cached = np.load(path, allow_pickle=True)
    return {
        "train_X": cached["train_X"].astype(np.float32),
        "train_y_idx": cached["train_y_idx"].astype(np.int32),
        "class_names": cached["class_names"].tolist(),
        "best_k": int(cached["best_k"][0]),
    }


def predict_single_image(image_path, model, transform, model_cache, distance_weighted):
    image = Image.open(image_path).convert("RGB")
    embedding = encode_batch(model, [transform(image)]).astype(np.float32)
    details = knn_prediction_details(
        model_cache["train_X"],
        model_cache["train_y_idx"],
        embedding,
        model_cache["class_names"],
        model_cache["best_k"],
        distance_weighted,
    )
    return details


def launch_demo_gui(model, transform, model_cache, distance_weighted, initial_image=None):
    root = tk.Tk()
    root.title("KNN CNN Demo")
    root.geometry("1120x760")

    state = {"photo": None}

    outer = ttk.Frame(root, padding=16)
    outer.pack(fill="both", expand=True)

    left = ttk.Frame(outer)
    left.pack(side="left", fill="both", expand=True)

    right = ttk.Frame(outer)
    right.pack(side="right", fill="y", padx=(20, 0))

    title = ttk.Label(right, text="CNN-Embedding KNN Demo", font=("Helvetica", 20, "bold"))
    title.pack(anchor="w", pady=(0, 12))

    subtitle = ttk.Label(
        right,
        text=f"KNN classifier with cached ResNet18 embeddings | k={model_cache['best_k']}",
        font=("Helvetica", 11),
        wraplength=360,
        justify="left",
    )
    subtitle.pack(anchor="w", pady=(0, 14))

    image_label = ttk.Label(left)
    image_label.pack(fill="both", expand=True)

    path_var = tk.StringVar(value="No image selected")
    predicted_var = tk.StringVar(value="Predicted label: -")
    share_var = tk.StringVar(value="Vote share: -")
    nearest_var = tk.StringVar(value="Nearest neighbor labels: -")
    distance_var = tk.StringVar(value="Nearest distance: -")

    ttk.Label(right, textvariable=path_var, wraplength=360, justify="left").pack(anchor="w", pady=(0, 12))
    ttk.Label(right, textvariable=predicted_var, font=("Helvetica", 16, "bold"), wraplength=360, justify="left").pack(anchor="w", pady=(0, 8))
    ttk.Label(right, textvariable=share_var, wraplength=360, justify="left").pack(anchor="w", pady=(0, 8))
    ttk.Label(right, textvariable=nearest_var, wraplength=360, justify="left").pack(anchor="w", pady=(0, 8))
    ttk.Label(right, textvariable=distance_var, wraplength=360, justify="left").pack(anchor="w", pady=(0, 12))

    ttk.Label(right, text="Vote breakdown", font=("Helvetica", 13, "bold")).pack(anchor="w")
    vote_text = tk.Text(right, width=42, height=14, wrap="word")
    vote_text.pack(anchor="w", pady=(6, 0))

    def render_image(path):
        image = Image.open(path).convert("RGB")
        display = image.copy()
        display.thumbnail((700, 700))
        photo = ImageTk.PhotoImage(display)
        state["photo"] = photo
        image_label.configure(image=photo)

    def evaluate_image(path):
        details = predict_single_image(path, model, transform, model_cache, distance_weighted)
        path_var.set(f"Image: {path}")
        predicted_var.set(f"Predicted label: {details['predicted_label']}")
        share_var.set(f"Vote share: {details['vote_share'] * 100:.1f}%")
        nearest_var.set("Nearest neighbor labels: " + ", ".join(details["neighbor_labels"]))
        distance_var.set(
            "Nearest distances: "
            + ", ".join(f"{distance:.3f}" for distance in details["neighbor_distances"][: min(3, len(details["neighbor_distances"]))])
        )
        vote_text.delete("1.0", tk.END)
        for label, share in details["vote_breakdown"]:
            vote_text.insert(tk.END, f"{label}: {share * 100:.1f}%\n")
        render_image(path)

    def choose_image():
        selected = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if selected:
            evaluate_image(selected)

    ttk.Button(right, text="Choose Image", command=choose_image).pack(anchor="w", pady=(16, 8))

    if initial_image:
        evaluate_image(initial_image)

    root.mainloop()


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


def main():
    args = parse_args()
    k_values = parse_k_values(args.k_values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_cache_path = args.output_dir / "knn_cnn_model_cache.npz"

    model, transform = build_backbone()
    if args.demo_gui:
        if not model_cache_path.exists():
            raise FileNotFoundError(
                f"Missing cached model: {model_cache_path}. Run the script once without --demo-gui first."
            )
        model_cache = load_model_cache(model_cache_path)
        launch_demo_gui(
            model,
            transform,
            model_cache,
            args.distance_weighted,
            initial_image=args.predict_image,
        )
        return

    if args.predict_image and model_cache_path.exists():
        model_cache = load_model_cache(model_cache_path)
        details = predict_single_image(
            args.predict_image,
            model,
            transform,
            model_cache,
            args.distance_weighted,
        )
        print("CNN-Embedding KNN Project")
        print(f"Loaded cached model: {model_cache_path}")
        print(f"Image: {args.predict_image}")
        print(f"Predicted label: {details['predicted_label']}")
        print(f"Vote share: {details['vote_share'] * 100:.1f}%")
        print("Nearest neighbors:", ", ".join(details["neighbor_labels"]))
        return

    train_records = sample_records(list_zip_records(args.train_zip), args.max_train_per_class, args.seed)
    val_records = sample_records(list_zip_records(args.val_zip), args.max_val_per_class, args.seed)

    print("CNN-Embedding KNN Project")
    print(f"Training images selected: {len(train_records)}")
    print(f"Validation images selected: {len(val_records)}")

    train_X, train_labels, _, train_cache, train_cached = load_or_build_embeddings(
        args.train_zip, train_records, args.cache_dir, model, transform, args.batch_size
    )
    val_X, val_labels, val_members, val_cache, val_cached = load_or_build_embeddings(
        args.val_zip, val_records, args.cache_dir, model, transform, args.batch_size
    )

    print(f"Train embedding cache: {train_cache} ({'loaded' if train_cached else 'built'})")
    print(f"Validation embedding cache: {val_cache} ({'loaded' if val_cached else 'built'})")

    class_map, class_names = label_to_int(train_labels + val_labels)
    train_y_idx = np.array([class_map[label] for label in train_labels], dtype=np.int32)
    val_y_idx = np.array([class_map[label] for label in val_labels], dtype=np.int32)

    scores = {}
    for k in k_values:
        predicted_idx = knn_predict(train_X, train_y_idx, val_X, k, args.batch_size, args.distance_weighted)
        scores[k] = accuracy_score(val_y_idx, predicted_idx)

    best_k = max(scores.items(), key=lambda item: (item[1], -item[0]))[0]
    predicted_idx = knn_predict(train_X, train_y_idx, val_X, best_k, args.batch_size, args.distance_weighted)
    overall_accuracy = accuracy_score(val_y_idx, predicted_idx)
    metric_rows = per_class_metrics(val_y_idx, predicted_idx, class_names)

    metrics_path = args.output_dir / "knn_cnn_per_class_metrics.csv"
    predictions_path = args.output_dir / "knn_cnn_validation_predictions.csv"
    summary_path = args.output_dir / "knn_cnn_summary.txt"
    demo_csv_path = args.output_dir / "knn_cnn_demo_predictions.csv"
    demo_dir = args.output_dir / "demo_images"

    save_metrics_csv(metrics_path, metric_rows)
    save_predictions_csv(predictions_path, val_members, val_y_idx, predicted_idx, class_names)
    save_model_cache(model_cache_path, train_X, train_y_idx, class_names, best_k)

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
    with demo_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "actual_label", "predicted_label", "correct"])
        writer.writeheader()
        writer.writerows(demo_rows)
    exported_demo_paths = export_demo_images(args.val_zip, demo_members, demo_dir)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("CNN-Embedding KNN Summary\n")
        handle.write(f"Overall validation accuracy: {overall_accuracy:.4f}\n")
        handle.write(f"Best k: {best_k}\n")
        handle.write(f"Distance weighted voting: {args.distance_weighted}\n")
        handle.write(f"Training images selected: {len(train_records)}\n")
        handle.write(f"Validation images selected: {len(val_records)}\n")
        handle.write(f"Train embedding cache: {train_cache}\n")
        handle.write(f"Validation embedding cache: {val_cache}\n")
        handle.write(f"Model cache: {model_cache_path}\n")
        handle.write(f"Metrics CSV: {metrics_path}\n")
        handle.write(f"Predictions CSV: {predictions_path}\n")
        handle.write(f"Demo CSV: {demo_csv_path}\n")
        handle.write("Demo images:\n")
        for path in exported_demo_paths:
            handle.write(f"  {path}\n")

    print("\nValidation accuracy by k")
    for k in k_values:
        print(f"  k={k:<2} -> {scores[k]:.4f}")
    print(f"Best k: {best_k}")
    print(f"\nOverall validation accuracy: {overall_accuracy:.4f}")
    print(f"Summary: {summary_path}")
    print(f"Model cache: {model_cache_path}")


if __name__ == "__main__":
    main()
