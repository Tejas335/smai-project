import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.decomposition import PCA

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

def plot_clusters(X, y, title, filename, max_legend_items=20):
    print(f"Generating PCA plot for {title}...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    unique_labels = list(set(y))
    
    # Calculate Cluster Quality Metric (Silhouette Score) on original high-dimensional data
    if len(unique_labels) > 1 and len(X) > 1:
        sil_score = silhouette_score(X, y)
        print(f"--> Cluster Quality (Silhouette Score) for {title} in original dimensional space: {sil_score:.4f}")
    
    plt.figure(figsize=(10, 8))
    
    # Use a colormap suitable for both few and many classes
    cmap = plt.get_cmap('tab20', max(2, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        idx = [j for j, val in enumerate(y) if val == label]
        # Modulo 20 to prevent index out of bounds on colormap if there's more than 20 classes
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, alpha=0.7, s=15, color=cmap(i % 20))
        
    plt.title(title)
    
    # Adding legend limits, if we have tons of classes it will stretch off page
    if len(unique_labels) <= max_legend_items:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    else:
        plt.text(0.01, 0.99, f"{len(unique_labels)} classes (legend hidden)", 
                 transform=plt.gca().transAxes, verticalalignment='top', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

def evaluate_knn(X_train, y_train, X_test, y_test, k=20, dataset_name="", verbose=True):
    if verbose:
        print(f"\n--- Evaluating {dataset_name} (K={k}) ---")
    if len(X_train) == 0 or len(X_test) == 0:
        if verbose:
            print("Empty training or test set. Skipping.")
        return 0.0

    if verbose:
        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    
    predictions = knn.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    if verbose:
        print(f"Accuracy for {dataset_name}: {acc:.4f} ({acc*100:.2f}%)")
    return acc

def main():
    base_dir = "embeddings"
    categories = ["mammals", "birds", "butterfly"]
    
    # Store data to eventually do the full dataset evaluation
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    
    # Dictionary to store accuracies: {category_name: [list of accuracies for k=1..20]}
    accuracies_by_k = {cat: [] for cat in categories}
    accuracies_by_k["Full Combined Dataset"] = []
    
    # Prepare list of k values
    k_values = list(range(1, 21))
    
    # 1. Evaluate each category separately
    for category in categories:
        print(f"\nLoading data for {category}...")
        X_train, y_train = load_embeddings(base_dir, category, "train")
        X_test, y_test = load_embeddings(base_dir, category, "test")
        
        # Accumulate for full evaluation
        if len(X_train) > 0:
            all_X_train.append(X_train)
            all_y_train.extend(y_train)
            
            # Plot the clusters for this particular category
            # To avoid the plot getting too busy, we can show just the training embeddings
            plot_title = f"PCA of {category.capitalize()} Classes"
            plot_file = f"{category}_pca_clusters.png"
            plot_clusters(X_train, y_train, plot_title, plot_file)
            
        if len(X_test) > 0:
            all_X_test.append(X_test)
            all_y_test.extend(y_test)
            
        print(f"Evaluating {category} across k=1 to 20...")
        for k in k_values:
            acc = evaluate_knn(X_train, y_train, X_test, y_test, k=k, dataset_name=category, verbose=False)
            accuracies_by_k[category].append(acc)
        
    # 2. Evaluate on full dataset
    if all_X_train and all_X_test:
        X_train_full = np.vstack(all_X_train)
        X_test_full = np.vstack(all_X_test)
        
        # We can also plot the overall distribution across mammals, birds, and butterflies
        super_categories = []
        for lbl in all_y_train:
            if lbl.startswith("mammals"): super_categories.append("Mammals")
            elif lbl.startswith("birds"): super_categories.append("Birds")
            elif lbl.startswith("butterfly"): super_categories.append("Butterflies")
            else: super_categories.append("Other")
            
        print("\nComputing silhouette score for the 3 main categories clustering together...")
        sil_score_cats = silhouette_score(X_train_full, super_categories)
        print(f"--> Silhouette Score (Mammals vs Birds vs Butterflies): {sil_score_cats:.4f}")
        
        plot_clusters(X_train_full, super_categories, "Broad PCA of All Categories", "all_categories_pca.png")
        
        print(f"Evaluating Full Combined Dataset across k=1 to 20...")
        for k in k_values:
            acc = evaluate_knn(X_train_full, all_y_train, X_test_full, all_y_test, k=k, dataset_name="Full Combined Dataset", verbose=False)
            accuracies_by_k["Full Combined Dataset"].append(acc)

    # 3. Create a table of results at the end
    print("\n================== KNN ACCURACY RESULTS (k=1 to 20) ==================")
    header = f"{'Dataset':<25} | " + " | ".join([f"k={k:<2}" for k in k_values])
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for dataset_name, acc_list in accuracies_by_k.items():
        if not acc_list: continue
        row = f"{dataset_name:<25} | " + " | ".join([f"{acc:.2f}" for acc in acc_list])
        print(row)
    print("-" * len(header))
    
    # 4. Plot the overall accuracy for different values of k
    plt.figure(figsize=(10, 6))
    for dataset_name, acc_list in accuracies_by_k.items():
        if acc_list:
            plt.plot(k_values, acc_list, marker='o', label=dataset_name)
    
    plt.title('KNN Accuracy vs. Value of K (1 to 20)')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy Score')
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('knn_k_accuracies.png')
    plt.close()
    print("\nSaved accuracy plot: knn_k_accuracies.png")

if __name__ == "__main__":
    main()
