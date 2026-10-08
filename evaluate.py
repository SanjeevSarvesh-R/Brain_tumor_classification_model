import os
import json
import subprocess
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model import load_model, preprocess_image, CLASSES


TEST_DIR = "Testing"


print("=" * 60)
print("BRAIN TUMOR MODEL EVALUATION")
print("=" * 60)

print("\nLoading trained model...")

model = load_model()

print("Model loaded successfully!")
print("Device:", next(model.parameters()).device)


test_data = []

for label, class_name in enumerate(CLASSES):

    class_folder = os.path.join(
        TEST_DIR,
        class_name
    )

    if not os.path.isdir(class_folder):
        raise FileNotFoundError(
            f"Testing folder not found: {class_folder}"
        )

    for filename in os.listdir(class_folder):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_path = os.path.join(
                class_folder,
                filename
            )

            test_data.append({
                "image": image_path,
                "label": label
            })


print("\nTotal testing images:", len(test_data))

if len(test_data) == 0:
    raise RuntimeError(
        "No images found in Testing folder."
    )


print("\nTesting dataset:")

for label, class_name in enumerate(CLASSES):

    count = sum(
        1
        for item in test_data
        if item["label"] == label
    )

    print(
        f"{class_name:12s}: {count}"
    )


print("\n" + "=" * 60)
print("RUNNING EVALUATION")
print("=" * 60)

true_labels = []
predicted_labels = []

correct = 0
total = 0


for item in test_data:

    image_path = item["image"]
    true_label = item["label"]

    try:

        image = Image.open(image_path)

        image_tensor = preprocess_image(image)

        with torch.no_grad():

            outputs = model(image_tensor)

            predicted_class = torch.argmax(
                outputs,
                dim=1
            ).item()

        true_labels.append(true_label)
        predicted_labels.append(predicted_class)

        if predicted_class == true_label:
            correct += 1

        total += 1

        if total % 100 == 0:

            current_accuracy = (
                correct / total
            ) * 100

            print(
                f"Processed: {total}/{len(test_data)} "
                f"| Accuracy: {current_accuracy:.2f}%"
            )

    except Exception as e:

        print(
            f"\nError processing: {image_path}"
        )

        print("Error:", e)


accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")
print("-" * 60)

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASSES,
        digits=4
    )
)


matrix = confusion_matrix(
    true_labels,
    predicted_labels
)


print("\nConfusion Matrix:")
print("-" * 60)

print("Rows = Actual")
print("Columns = Predicted")

print("\nClasses:")

for index, class_name in enumerate(CLASSES):
    print(f"{index} = {class_name}")

print()
print(matrix)


print("\nConfusion Matrix with class names:")
print("-" * 60)

print(
    f"{'Actual':12s}"
    + "".join(
        f"{class_name[:10]:>12s}"
        for class_name in CLASSES
    )
)

for index, class_name in enumerate(CLASSES):

    row = matrix[index]

    print(
        f"{class_name:12s}"
        + "".join(
            f"{value:>12d}"
            for value in row
        )
    )


print("\nPer-class accuracy:")
print("-" * 60)

for index, class_name in enumerate(CLASSES):

    actual_count = matrix[index].sum()

    correctly_predicted = matrix[index][index]

    if actual_count > 0:
        class_accuracy = (
            correctly_predicted /
            actual_count
        ) * 100
    else:
        class_accuracy = 0

    print(
        f"{class_name:12s}: "
        f"{class_accuracy:.2f}% "
        f"({correctly_predicted}/{actual_count})"
    )


# ------------------------------------------------------------
# GENERATE MACHINE-READABLE METRICS (metrics.json)
# ------------------------------------------------------------
report_dict = classification_report(
    true_labels,
    predicted_labels,
    target_names=CLASSES,
    digits=4,
    output_dict=True
)

metrics_data = {
    "test_accuracy": round(float(accuracy), 4),
    "precision_macro": round(float(report_dict["macro avg"]["precision"]), 4),
    "recall_macro": round(float(report_dict["macro avg"]["recall"]), 4),
    "f1_macro": round(float(report_dict["macro avg"]["f1-score"]), 4),
    "precision_weighted": round(float(report_dict["weighted avg"]["precision"]), 4),
    "recall_weighted": round(float(report_dict["weighted avg"]["recall"]), 4),
    "f1_weighted": round(float(report_dict["weighted avg"]["f1-score"]), 4),
}

for index, class_name in enumerate(CLASSES):
    actual_count = int(matrix[index].sum())
    correctly_predicted = int(matrix[index][index])
    class_acc = (correctly_predicted / actual_count) if actual_count > 0 else 0.0

    metrics_data[f"{class_name}_precision"] = round(float(report_dict[class_name]["precision"]), 4)
    metrics_data[f"{class_name}_recall"] = round(float(report_dict[class_name]["recall"]), 4)
    metrics_data[f"{class_name}_f1"] = round(float(report_dict[class_name]["f1-score"]), 4)
    metrics_data[f"{class_name}_accuracy"] = round(float(class_acc), 4)

with open("metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print("\nMetrics written to: metrics.json")

# ------------------------------------------------------------
# PLOT AND SAVE CONFUSION MATRIX (confusion_matrix.png)
# ------------------------------------------------------------
confusion_matrix_path = "confusion_matrix.png"
try:
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("BrainVerse DenseNet121 — Confusion Matrix", fontsize=12, pad=12)
    plt.colorbar()
    tick_marks = range(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASSES)

    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{matrix[i, j]:d}",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=11
            )

    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to: {confusion_matrix_path}")
except Exception as e:
    print(f"Warning: Failed to generate confusion matrix plot: {e}")

# ------------------------------------------------------------
# LOCAL MLFLOW LOGGING
# ------------------------------------------------------------
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "BrainVerse-DenseNet121")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)



def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        return "unknown"


active_run = mlflow.active_run()
run_context = (
    mlflow.start_run(run_name="evaluate_densenet121", nested=True)
    if active_run
    else mlflow.start_run(run_name="evaluate_densenet121")
)

with run_context as run:
    mlflow.set_tag("git_commit", get_git_commit())
    mlflow.set_tag("model_architecture", "DenseNet121")
    mlflow.set_tag("stage", "evaluation")
    mlflow.set_tag("framework", "PyTorch + MONAI")

    mlflow.log_param("test_dir", TEST_DIR)
    mlflow.log_param("num_test_samples", len(test_data))
    mlflow.log_param("model_path", "best_brain_tumor_monai.pth")

    for metric_name, metric_val in metrics_data.items():
        mlflow.log_metric(metric_name, float(metric_val))

    if os.path.exists("metrics.json"):
        mlflow.log_artifact("metrics.json")

    if os.path.exists(confusion_matrix_path):
        mlflow.log_artifact(confusion_matrix_path)

    print("\nMLflow Tracking:")
    print(f"Tracking URI : {tracking_uri}")
    print(f"Experiment   : {experiment_name}")
    print(f"Run ID       : {run.info.run_id}")


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)