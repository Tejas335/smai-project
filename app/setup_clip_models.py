"""
Setup script: generate CLIP & BioCLIP embeddings, train KNN/SVM classifiers.
Run once before using KNN/SVM models in the Streamlit app.

Usage:
    python app/setup_clip_models.py
"""

import os
import numpy as np
import torch
import open_clip
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
from tqdm import tqdm

CATEGORIES = ["mammals", "birds", "butterfly"]
DATASET_NAME = "ViratGarg/animal_species_SMAI"


def compute_embeddings(model, preprocess, device, category):
    """Compute embeddings for a dataset category, returning train/test splits."""
    print(f"\n--- Processing {category} ---")
    dataset = load_dataset(DATASET_NAME, data_dir=category, split="train")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    results = {}
    for split_name in ["train", "test"]:
        split_data = dataset[split_name]
        X, y = [], []
        has_labels = "label" in split_data.features
        labels = split_data.features["label"].names if has_labels else None

        batch_size = 32
        for i in tqdm(range(0, len(split_data), batch_size), desc=f"{category}/{split_name}"):
            batch = split_data[i : i + batch_size]
            images = [preprocess(img).unsqueeze(0) for img in batch["image"]]

            with torch.no_grad():
                image_tensors = torch.cat(images).to(device)
                features = model.encode_image(image_tensors)
                features = features / features.norm(dim=-1, keepdim=True)

            for j in range(len(batch["image"])):
                X.append(features[j].cpu().numpy())
                if has_labels:
                    class_name = labels[batch["label"][j]]
                    y.append(f"{category}_{class_name}")
                else:
                    y.append(f"{category}_unknown")

        results[split_name] = (np.vstack(X), y)

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        ("clip", "ViT-B-16", "openai"),
        ("bioclip", "hf-hub:imageomics/bioclip", None),
    ]

    for config_name, model_name, pretrained in configs:
        print(f"\n{'='*60}")
        print(f"Processing {config_name} ({model_name})")
        print(f"{'='*60}")

        # Load CLIP model
        if pretrained:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        model = model.to(device).eval()

        # Compute embeddings
        all_train_X, all_train_y = [], []
        all_test_X, all_test_y = [], []

        for category in CATEGORIES:
            results = compute_embeddings(model, preprocess, device, category)
            train_X, train_y = results["train"]
            test_X, test_y = results["test"]
            all_train_X.append(train_X)
            all_train_y.extend(train_y)
            all_test_X.append(test_X)
            all_test_y.extend(test_y)

        X_train = np.vstack(all_train_X)
        X_test = np.vstack(all_test_X)

        print(f"\nTotal train: {len(X_train)}, Total test: {len(X_test)}")

        # ── Train KNN ──
        print("Training KNN (k=5)...")
        knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        knn.fit(X_train, all_train_y)
        knn_acc = knn.score(X_test, all_test_y)
        print(f"KNN accuracy: {knn_acc:.4f}")

        knn_path = os.path.join(output_dir, f"{config_name}_knn.joblib")
        joblib.dump(knn, knn_path)
        print(f"Saved KNN model → {knn_path}")

        # ── Train SVM ──
        print("Training SVM (linear, probability=True)... this may take a few minutes")
        svm = SVC(kernel="linear", probability=True)
        svm.fit(X_train, all_train_y)
        svm_acc = svm.score(X_test, all_test_y)
        print(f"SVM accuracy: {svm_acc:.4f}")

        svm_path = os.path.join(output_dir, f"{config_name}_svm.joblib")
        joblib.dump(svm, svm_path)
        print(f"Saved SVM model → {svm_path}")

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n✅ All models trained and saved!")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
