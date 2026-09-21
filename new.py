import os

TRAIN_DIR = "Training"
TEST_DIR = "Testing"

classes = ["glioma", "meningioma", "pituitary", "notumor"]

print("===== TRAINING DATA =====")

total_train = 0

for cls in classes:
    path = os.path.join(TRAIN_DIR, cls)

    count = len([
        f for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"{cls}: {count}")
    total_train += count

print("Total training images:", total_train)


print("\n===== TESTING DATA =====")

total_test = 0

for cls in classes:
    path = os.path.join(TEST_DIR, cls)

    count = len([
        f for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"{cls}: {count}")
    total_test += count

print("Total testing images:", total_test)
