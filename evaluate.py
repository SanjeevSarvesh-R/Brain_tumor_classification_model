import os
import torch

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


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)