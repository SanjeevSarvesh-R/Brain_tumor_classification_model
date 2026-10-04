import os
import torch
import numpy as np
from PIL import Image
from monai.networks.nets import DenseNet121



# Use absolute path so it works from any working directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_brain_tumor_monai.pth")

IMAGE_SIZE = (224, 224)

CLASSES = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor"
]

NUM_CLASSES = 4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



def load_model():
    """
    Create the same DenseNet121 architecture used during training
    and load the trained model weights.
    """

    model = DenseNet121(
        spatial_dims=2,
        in_channels=1,
        out_channels=NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model



def preprocess_image(image):
    """
    Apply the same basic preprocessing used by predict.py:

    RGB image
        -> resize to 224x224
        -> grayscale
        -> scale intensity to 0-1
        -> add channel dimension
        -> add batch dimension
    """

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

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

    return image.to(device)



def predict_image(image, model):
    """
    Predict the class of a brain MRI image.

    Returns:
        prediction   : predicted class name
        confidence   : model confidence as a percentage
        probabilities: probability for each class
    """

    image = preprocess_image(image)

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

    probability_dict = {
        class_name: probabilities[0][index].item() * 100
        for index, class_name in enumerate(CLASSES)
    }

    return prediction, confidence, probability_dict