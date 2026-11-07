
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Define dataset paths
train_path = "imagewoof_train/train"
val_path = "imagewoof_val/val"

# Step 2: Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Step 3: Load datasets
train_dataset = datasets.ImageFolder(
    root=train_path,
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=val_path,
    transform=transform
)

# Step 4: Required class order from classes.txt
class_names = [
    "shih_tzu",
    "rhodesian_ridgeback",
    "beagle",
    "english_foxhound",
    "border_terrier",
    "australian_terrier",
    "golden_retriever",
    "old_english_sheepdog",
    "samoyed",
    "dingo"
]

# Step 5: Correct class labels for both datasets
def apply_class_order(dataset):
    mapping = {
        dataset.class_to_idx[name]: idx
        for idx, name in enumerate(class_names)
    }

    dataset.samples = [
        (path, mapping[label])
        for path, label in dataset.samples
    ]

    dataset.targets = [
        mapping[label]
        for label in dataset.targets
    ]

    # ImageFolder uses samples through imgs
    dataset.imgs = dataset.samples

    dataset.classes = class_names

    dataset.class_to_idx = {
        name: idx
        for idx, name in enumerate(class_names)
    }


apply_class_order(train_dataset)
apply_class_order(val_dataset)

# Step 6: Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# Step 7: Display dataset information
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Number of classes:", len(class_names))

print("\nCorrect class mapping:")
for idx, name in enumerate(class_names):
    print(f"{idx}: {name}")

# Step 8: Test a batch of images
images, labels = next(iter(train_loader))

print("\nImage batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("First 10 batch labels:", labels[:10])

# Step 9: Verify correct labels
assert images.shape == (32, 3, 128, 128)
assert labels.min() >= 0
assert labels.max() < 10

assert train_dataset.class_to_idx["shih_tzu"] == 0
assert train_dataset.class_to_idx["dingo"] == 9

print("\nSUCCESS: Dataset is ready for CNN training!")



# STAGE 2: BUILD CNN ARCHITECTURE

import torch.nn as nn

# Define our custom CNN
class DogBreedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            # Block 1: 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Classification
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)


# Choose CPU or GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Create model
model = DogBreedCNN().to(device)

print("\nCNN model created successfully!")
print("Device:", device)

# Count model parameters
total_params = sum(
    p.numel() for p in model.parameters()
)

print("Total model parameters:", total_params)

# Test model output
model.eval()

with torch.no_grad():
    sample_images = images.to(device)
    outputs = model(sample_images)

print("Model output shape:", outputs.shape)

assert outputs.shape == (32, 10)

print("\nSUCCESS: CNN architecture is working!")
