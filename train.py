import os
# pyrefly: ignore [missing-import]
import torch
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Lambdad,
    Resized,
    ScaleIntensityd,
    RandFlipd,
    RandRotate90d,
    RandZoomd
)

from monai.networks.nets import DenseNet121


TRAIN_DIR = "Training"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

NUM_CLASSES = 4

EPOCHS = 10

LEARNING_RATE = 0.0001

RANDOM_STATE = 42


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. PyTorch cannot access your NVIDIA GPU. "
        "Install a CUDA-enabled PyTorch version before training."
    )

device = torch.device("cuda")

print("=" * 60)
print("BRAIN TUMOR CLASSIFICATION - MONAI")
print("=" * 60)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))

gpu_properties = torch.cuda.get_device_properties(0)

print(
    f"GPU Memory: "
    f"{gpu_properties.total_memory / (1024 ** 3):.2f} GB"
)


classes = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor"
]


print("\nClass mapping:")

for index, class_name in enumerate(classes):
    print(f"{index} -> {class_name}")


data = []


for label, class_name in enumerate(classes):

    class_folder = os.path.join(
        TRAIN_DIR,
        class_name
    )

    if not os.path.isdir(class_folder):
        raise FileNotFoundError(
            f"Folder not found: {class_folder}"
        )

    for filename in os.listdir(class_folder):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            data.append({
                "image": os.path.join(
                    class_folder,
                    filename
                ),
                "label": label
            })


print("\nTotal images:", len(data))


if len(data) == 0:
    raise RuntimeError(
        "No images found in the Training folder."
    )


labels = [
    item["label"]
    for item in data
]


train_data, validation_data = train_test_split(
    data,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=labels
)


print("Training images:", len(train_data))
print("Validation images:", len(validation_data))


def convert_to_grayscale(x):

    if x.shape[0] >= 3:

        x = (
            0.299 * x[0:1]
            + 0.587 * x[1:2]
            + 0.114 * x[2:3]
        )

    else:

        x = x[0:1]

    return x


train_transforms = Compose([

    LoadImaged(
        keys=["image"],
        image_only=True
    ),

    EnsureChannelFirstd(
        keys=["image"]
    ),

    Lambdad(
        keys=["image"],
        func=convert_to_grayscale
    ),

    Resized(
        keys=["image"],
        spatial_size=IMAGE_SIZE
    ),

    ScaleIntensityd(
        keys=["image"]
    ),

    RandFlipd(
        keys=["image"],
        prob=0.5,
        spatial_axis=1
    ),

    RandRotate90d(
        keys=["image"],
        prob=0.5,
        max_k=3
    ),

    RandZoomd(
        keys=["image"],
        prob=0.5,
        min_zoom=0.9,
        max_zoom=1.1
    )
])


validation_transforms = Compose([

    LoadImaged(
        keys=["image"],
        image_only=True
    ),

    EnsureChannelFirstd(
        keys=["image"]
    ),

    Lambdad(
        keys=["image"],
        func=convert_to_grayscale
    ),

    Resized(
        keys=["image"],
        spatial_size=IMAGE_SIZE
    ),

    ScaleIntensityd(
        keys=["image"]
    )
])


print("\nCreating MONAI datasets...")


train_dataset = Dataset(
    data=train_data,
    transform=train_transforms
)


validation_dataset = Dataset(
    data=validation_data,
    transform=validation_transforms
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


print("DataLoaders created successfully.")


print("\nChecking training batch...")


sample_batch = next(iter(train_loader))


sample_images = sample_batch["image"]

sample_labels = sample_batch["label"]


print(
    "Batch image shape:",
    sample_images.shape
)

print(
    "Batch label shape:",
    sample_labels.shape
)

print(
    "Image channels:",
    sample_images.shape[1]
)


if sample_images.shape[1] != 1:

    raise RuntimeError(
        f"Channel conversion failed. "
        f"Expected 1 channel but got "
        f"{sample_images.shape[1]}"
    )


if sample_images.shape[2:] != IMAGE_SIZE:

    raise RuntimeError(
        f"Image resize failed. "
        f"Expected {IMAGE_SIZE} but got "
        f"{sample_images.shape[2:]}"
    )


print("\nImage preprocessing verified successfully.")


model = DenseNet121(
    spatial_dims=2,
    in_channels=1,
    out_channels=NUM_CLASSES
)


model = model.to(device)


print("\nMONAI DenseNet121 created successfully.")

print(
    "Model device:",
    next(model.parameters()).device
)


loss_function = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_validation_accuracy = 0.0


print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    correct_train = 0

    total_train = 0


    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} - Training",
        unit="batch"
    )


    for batch in train_progress:

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad()


        outputs = model(images)


        loss = loss_function(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        total_train_loss += loss.item()


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct_train += (
            predictions == labels
        ).sum().item()


        total_train += labels.size(0)


        current_accuracy = (
            correct_train /
            total_train
        ) * 100


        train_progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_accuracy:.2f}%"
        )


    average_train_loss = (
        total_train_loss /
        len(train_loader)
    )


    train_accuracy = (
        correct_train /
        total_train
    )


    model.eval()


    total_validation_loss = 0.0

    correct_validation = 0

    total_validation = 0


    validation_progress = tqdm(
        validation_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation",
        unit="batch"
    )


    with torch.no_grad():

        for batch in validation_progress:

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            labels = batch["label"].to(
                device,
                non_blocking=True
            )


            outputs = model(images)


            loss = loss_function(
                outputs,
                labels
            )


            total_validation_loss += loss.item()


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            correct_validation += (
                predictions == labels
            ).sum().item()


            total_validation += labels.size(0)


            current_validation_accuracy = (
                correct_validation /
                total_validation
            ) * 100


            validation_progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{current_validation_accuracy:.2f}%"
            )


    average_validation_loss = (
        total_validation_loss /
        len(validation_loader)
    )


    validation_accuracy = (
        correct_validation /
        total_validation
    )


    print("\n")
    print("-" * 60)


    print(
        f"Epoch {epoch + 1}/{EPOCHS} COMPLETED"
    )


    print(
        f"Training Loss: "
        f"{average_train_loss:.4f}"
    )


    print(
        f"Training Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )


    print(
        f"Validation Loss: "
        f"{average_validation_loss:.4f}"
    )


    print(
        f"Validation Accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )


    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy


        torch.save(
            model.state_dict(),
            "best_brain_tumor_monai.pth"
        )


        print("Best model saved!")


print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)


print(
    f"Best Validation Accuracy: "
    f"{best_validation_accuracy * 100:.2f}%"
)


print("\nModel saved as:")

print("best_brain_tumor_monai.pth")