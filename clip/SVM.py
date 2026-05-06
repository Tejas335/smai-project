import os
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_embeddings(base_dir, category, split):
    """
    Loads embeddings and labels for a given category and split.
    Returns:
        X: numpy array of embeddings (N, feature_dim)
        y: list of string labels (N,)
    """
    split_dir = os.path.join(base_dir, category, split)
    X = []
    y = []
    
    if not os.path.exists(split_dir):
        print(f"Warning: Directory not found: {split_dir}")
        return np.array(X), y

    # Class folders
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.pt'):
                file_path = os.path.join(class_dir, file_name)
                # Load the embedding (it's stored as a numpy array with torch.save)
                embedding = torch.load(file_path, weights_only=False)
                X.append(embedding.flatten())
                # Use a combined label to prevent clashes across categories
                # when doing the full dataset evaluation
                y.append(f"{category}_{class_name}")
                
    if X:
        X = np.vstack(X)
        
    return X, y

def evaluate_svm(X_train, y_train, X_test, y_test, dataset_name=""):
    print(f"\n--- Evaluating SVM on {dataset_name} ---")
    if len(X_train) == 0 or len(X_test) == 0:
        print("Empty training or test set. Skipping.")
        return 0.0

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # We use a Linear Kernel here, which typically works exceptionally well for 
    # high-dimensional embeddings like CLIP generated vectors
    svm = SVC(kernel='linear')
    print("Training SVM... (this might take a moment)")
    svm.fit(X_train, y_train)
    
    print("Predicting on test set...")
    predictions = svm.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"Accuracy for {dataset_name}: {acc:.4f} ({acc*100:.2f}%)")
    return acc

def main():
    # Make sure to point to the correct clips embedding directory
    base_dir = "embeddings"
    categories = ["mammals", "birds", "butterfly"]
    
    # Store data to eventually do the full dataset evaluation
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    
    # 1. Evaluate each category separately
    for category in categories:
        print(f"\nLoading data for {category}...")
        X_train, y_train = load_embeddings(base_dir, category, "train")
        X_test, y_test = load_embeddings(base_dir, category, "test")
        
        # Accumulate for full evaluation
        if len(X_train) > 0:
            all_X_train.append(X_train)
            all_y_train.extend(y_train)
        if len(X_test) > 0:
            all_X_test.append(X_test)
            all_y_test.extend(y_test)
            
        evaluate_svm(X_train, y_train, X_test, y_test, dataset_name=category)
        
    # 2. Evaluate on full dataset
    if all_X_train and all_X_test:
        X_train_full = np.vstack(all_X_train)
        X_test_full = np.vstack(all_X_test)
        
        evaluate_svm(X_train_full, all_y_train, X_test_full, all_y_test, dataset_name="Full Combined Dataset")

if __name__ == "__main__":
    main()
