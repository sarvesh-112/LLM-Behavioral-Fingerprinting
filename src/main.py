from src.models.build_dataset import FingerprintDatasetBuilder
from src.models.classifiers import ModelIdentifier
from src.evaluation.visualization import plot_pca


def main():
    # 1️⃣ Build dataset
    builder = FingerprintDatasetBuilder()

    model_dirs = [
        "data/raw/responses/baseline/distilgpt2",
        "data/raw/responses/baseline/google/flan-t5-small"
    ]

    labels = ["distilgpt2", "flan_t5"]

    X, y = builder.build(model_dirs, labels)

    # 2️⃣ Train & evaluate classifiers
    clf = ModelIdentifier()
    results = clf.train_and_evaluate(X, y)

    for model, res in results.items():
        print(f"\n=== {model.upper()} ===")
        print("Accuracy:", res["accuracy"])
        print("Confusion Matrix:\n", res["confusion_matrix"])
        print("Report:\n", res["report"])

    # 3️⃣ PCA visualization
    plot_pca(X, y)


if __name__ == "__main__":
    main()
