import torch
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog

from monai.networks.nets import DenseNet121



MODEL_PATH = "best_brain_tumor_monai.pth"

IMAGE_SIZE = (224, 224)

CLASSES = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor"
]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 60)
print("BRAIN TUMOR CLASSIFICATION")
print("=" * 60)

print("Device:", device)
print("Loading model...")


model = DenseNet121(
    spatial_dims=2,
    in_channels=1,
    out_channels=4
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model = model.to(device)

model.eval()


print("Model loaded successfully.")


root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Brain MRI Image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png"),
        ("JPG files", "*.jpg"),
        ("JPEG files", "*.jpeg"),
        ("PNG files", "*.png")
    ]
)


if not image_path:

    print("\nNo image selected.")

    root.destroy()

    exit()


print("\nSelected image:")
print(image_path)


image = Image.open(image_path).convert("RGB")


image = image.resize(
    IMAGE_SIZE
)


image = np.array(
    image,
    dtype=np.float32
)


red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]


image = (
    0.299 * red
    + 0.587 * green
    + 0.114 * blue
)


image = image / 255.0


image = torch.tensor(
    image,
    dtype=torch.float32
)


image = image.unsqueeze(0)
image = image.unsqueeze(0)


image = image.to(device)


print("\nPredicting...")


with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()


prediction = CLASSES[predicted_class]


confidence = (
    probabilities[0][predicted_class].item()
    * 100
)


print("\n")
print("=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(
    f"Prediction: {prediction.upper()}"
)

print(
    f"Confidence: {confidence:.2f}%"
)

print("\nClass probabilities:")

for class_name, probability in zip(
    CLASSES,
    probabilities[0]
):

    print(
        f"{class_name:12s}: "
        f"{probability.item() * 100:.2f}%"
    )

print("=" * 60)


if prediction == "notumor":

    print("Result: No tumor detected by the model.")

else:

    print(
        f"Result: The model predicts "
        f"{prediction}."
    )


print("=" * 60)


root.destroy()