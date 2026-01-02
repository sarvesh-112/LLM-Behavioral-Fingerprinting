import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

# ------------------------------------------------------------------
# Fix project root
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.build_dataset import FingerprintDatasetBuilder

# ------------------------------------------------------------------
# Output directories
# ------------------------------------------------------------------
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
TABLE_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
LOG_DIR = os.path.join(PROJECT_ROOT, "results", "logs")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Visualization functions
# ------------------------------------------------------------------
def plot_pca(X, y):
    X_2d = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(6, 5))
    for label in sorted(set(y)):
        idx = (y == label)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label, s=90)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of LLM Behavioral Fingerprints")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pca_fingerprint.png"), dpi=300)
    plt.close()


def plot_tsne(X, y):
    X_2d = TSNE(n_components=2, perplexity=3, random_state=42).fit_transform(X)
    plt.figure(figsize=(6, 5))
    for label in sorted(set(y)):
        idx = (y == label)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label, s=90)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("t-SNE of LLM Behavioral Fingerprints")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tsne_fingerprint.png"), dpi=300)
    plt.close()


def plot_feature_bars(X, y):
    labels = sorted(set(y))
    means = {l: X[y == l].mean(axis=0) for l in labels}
    x = np.arange(X.shape[1])
    width = 0.35

    plt.figure(figsize=(9, 4))
    for i, label in enumerate(labels):
        plt.bar(x + i * width, means[label], width, label=label)

    plt.xlabel("Fingerprint Dimensions")
    plt.ylabel("Mean Feature Value")
    plt.title("Mean Behavioral Fingerprint Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fingerprint_features.png"), dpi=300)
    plt.close()


def plot_distance_heatmap(X, y):
    D = cosine_distances(X)
    plt.figure(figsize=(6, 5))
    plt.imshow(D, cmap="viridis")
    plt.colorbar(label="Cosine Distance")
    plt.xticks(range(len(y)), y, rotation=45)
    plt.yticks(range(len(y)), y)
    plt.title("Fingerprint Distance Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fingerprint_distance_heatmap.png"), dpi=300)
    plt.close()


# ------------------------------------------------------------------
# Table + log generation
# ------------------------------------------------------------------
def save_fingerprint_table(X, y):
    path = os.path.join(TABLE_DIR, "fingerprint_vectors.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["model"] + [f"F{i+1}" for i in range(X.shape[1])]
        writer.writerow(header)
        for vec, label in zip(X, y):
            writer.writerow([label] + list(vec))


def save_feature_statistics(X, y):
    path = os.path.join(TABLE_DIR, "feature_statistics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "feature", "mean", "std"])
        for label in sorted(set(y)):
            subset = X[y == label]
            for i in range(X.shape[1]):
                writer.writerow([
                    label,
                    f"F{i+1}",
                    subset[:, i].mean(),
                    subset[:, i].std()
                ])


def save_inter_model_distances(X, y):
    path = os.path.join(TABLE_DIR, "inter_model_distances.csv")
    D = cosine_distances(X)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_i", "sample_j", "model_i", "model_j", "distance"])
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                writer.writerow([i, j, y[i], y[j], D[i, j]])


def save_log(X, y):
    path = os.path.join(LOG_DIR, "experiment_log.txt")
    with open(path, "w") as f:
        f.write("LLM Behavioral Fingerprinting Experiment\n")
        f.write("=" * 45 + "\n")
        f.write(f"Samples: {len(y)}\n")
        f.write(f"Fingerprint dimension: {X.shape[1]}\n")
        f.write(f"Models: {set(y)}\n")
        f.write("Figures generated:\n")
        f.write("- PCA\n- t-SNE\n- Feature bars\n- Distance heatmap\n")
        f.write("Tables generated:\n")
        f.write("- fingerprint_vectors.csv\n")
        f.write("- feature_statistics.csv\n")
        f.write("- inter_model_distances.csv\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    builder = FingerprintDatasetBuilder()

    model_dirs = [
        "data/raw/responses/baseline/distilgpt2",
        "data/raw/responses/baseline/google/flan-t5-small"
    ]

    labels = ["distilgpt2", "flan_t5"]

    X, y = builder.build(model_dirs, labels)
    y = np.array(y)

    plot_pca(X, y)
    plot_tsne(X, y)
    plot_feature_bars(X, y)
    plot_distance_heatmap(X, y)

    save_fingerprint_table(X, y)
    save_feature_statistics(X, y)
    save_inter_model_distances(X, y)
    save_log(X, y)

    print("Figures saved to:", FIG_DIR)
    print("Tables saved to:", TABLE_DIR)
    print("Logs saved to:", LOG_DIR)
