import os

from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    ScaleIntensityd
)




TRAIN_DIR = "Training"




classes = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor"
]




data = []

for label, class_name in enumerate(classes):

    class_path = os.path.join(
        TRAIN_DIR,
        class_name
    )

    for filename in os.listdir(class_path):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_path = os.path.join(
                class_path,
                filename
            )

            data.append({
                "image": image_path,
                "label": label
            })




print("Total images:", len(data))

print("\nClasses:")

for i, class_name in enumerate(classes):
    print(i, "=", class_name)




transforms = Compose([

    LoadImaged(keys=["image"]),

    EnsureChannelFirstd(keys=["image"]),

    Resized(
        keys=["image"],
        spatial_size=(224, 224)
    ),

    ScaleIntensityd(
        keys=["image"]
    )
])




dataset = Dataset(
    data=data,
    transform=transforms
)



sample = dataset[0]


print("\nFirst image information:")

print("Image shape:", sample["image"].shape)

print("Label:", sample["label"])

print(
    "Class:",
    classes[int(sample["label"])]
)