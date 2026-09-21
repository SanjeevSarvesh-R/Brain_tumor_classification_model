from PIL import Image
from tkinter import Tk, filedialog

from model import load_model, predict_image


print("=" * 60)
print("TESTING model.py")
print("=" * 60)

# Load model
print("\nLoading model...")

model = load_model()

print("Model loaded successfully!")


root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Brain MRI Image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png")
    ]
)

root.destroy()

if not image_path:
    print("No image selected.")
    exit()

print("\nSelected image:")
print(image_path)


image = Image.open(image_path)

# Predict
print("\nRunning prediction...")

prediction, confidence, probabilities = predict_image(
    image,
    model
)

# Display result
print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print("Prediction:", prediction.upper())
print(f"Confidence: {confidence:.2f}%")

print("\nClass probabilities:")

for class_name, probability in probabilities.items():
    print(f"{class_name:12s}: {probability:.2f}%")

print("=" * 60)
print("model.py is working correctly!")
print("=" * 60)