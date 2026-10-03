# %% ============================================================
# CELL 1 - IMPORT LIBRARIES
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset


print("Libraries imported successfully.")


# %% ============================================================
# CELL 2 - CHECK PYTHON, PYTORCH AND GPU
# ============================================================

import sys

print("Python:", sys.executable)
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %% ============================================================
# CELL 3 - SET RANDOM SEED
# ============================================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

print("Random seed set to:", seed)


# %% ============================================================
# CELL 4 - CIFAR-10 CLASS NAMES
# ============================================================

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

print("Number of classes:", len(class_names))

for index, name in enumerate(class_names):
    print(index, "=", name)


# %% ============================================================
# CELL 5 - DATA AUGMENTATION AND PREPROCESSING
# ============================================================

# Training images receive augmentation.
# This helps reduce overfitting.

train_transform = transforms.Compose([

    # Randomly flip image horizontally
    transforms.RandomHorizontalFlip(p=0.5),

    # Randomly rotate by up to 10 degrees
    transforms.RandomRotation(10),

    # Random shift/crop
    transforms.RandomCrop(
        size=32,
        padding=4
    ),

    # Convert PIL image to tensor
    transforms.ToTensor(),

    # Normalize RGB channels from approximately [0, 1]
    # to approximately [-1, 1]
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


# Validation and test data should NOT receive
# random augmentation.

test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

print("Transforms created successfully.")


# %% ============================================================
# CELL 6 - DOWNLOAD CIFAR-10
# ============================================================

# Training version with augmentation
train_data_augmented = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)


# Same training images without augmentation.
# We use these for validation.
train_data_normal = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=test_transform
)


# Official CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)


print("Dataset loaded successfully.")
print("Original training images:", len(train_data_augmented))
print("Test images:", len(test_dataset))

# %% ============================================================
# CELL 7 - CREATE TRAIN / VALIDATION SPLIT
# ============================================================

dataset_size = len(train_data_augmented)

indices = np.arange(dataset_size)

# Shuffle using fixed seed
rng = np.random.default_rng(seed)
rng.shuffle(indices)

# 80% training, 20% validation
split = int(0.8 * dataset_size)

train_indices = indices[:split]
val_indices = indices[split:]


train_dataset = Subset(
    train_data_augmented,
    train_indices
)


val_dataset = Subset(
    train_data_normal,
    val_indices
)


print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images:", len(test_dataset))

# %% ============================================================
# CELL 8 - CREATE DATALOADERS
# ============================================================

batch_size = 64


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,

    # Safe setting for Windows + VS Code
    num_workers=0,

    pin_memory=False
)


val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


print("DataLoaders created.")
print("Batch size:", batch_size)


# %% ============================================================
# CELL 9 - TEST ONE BATCH
# ============================================================

images, labels = next(iter(train_loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Image datatype:", images.dtype)

# %% ============================================================
# CELL 10 - DISPLAY SAMPLE CIFAR-10 IMAGES
# ============================================================

images, labels = next(iter(train_loader))

plt.figure(figsize=(12, 6))


for i in range(10):

    plt.subplot(2, 5, i + 1)

    # Copy image
    image = images[i].clone()

    # Undo normalization:
    # [-1,1] -> [0,1]
    image = image * 0.5 + 0.5

    # PyTorch images:
    # Channel x Height x Width
    #
    # Matplotlib needs:
    # Height x Width x Channel

    image = image.permute(1, 2, 0)

    image = image.numpy()

    image = np.clip(
        image,
        0,
        1
    )

    plt.imshow(image)

    plt.title(
        class_names[
            labels[i].item()
        ]
    )

    plt.axis("off")


plt.suptitle(
    "Sample CIFAR-10 Training Images"
)

plt.tight_layout()

plt.show()

# %% ============================================================
# CELL 11 - DEFINE CNN ARCHITECTURE
# ============================================================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()


        # =======================================================
        # CONVOLUTION BLOCK 1
        # =======================================================

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )


        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1
        )


        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # =======================================================
        # CONVOLUTION BLOCK 2
        # =======================================================

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )


        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )


        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # =======================================================
        # CONVOLUTION BLOCK 3
        # =======================================================

        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )


        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # Activation function
        self.relu = nn.ReLU()


        # Helps prevent overfitting
        self.dropout = nn.Dropout(
            p=0.5
        )


        # =======================================================
        # FULLY CONNECTED LAYERS
        # =======================================================

        # Original:
        # 32 x 32
        #
        # After pool1:
        # 16 x 16
        #
        # After pool2:
        # 8 x 8
        #
        # After pool3:
        # 4 x 4
        #
        # Final feature map:
        # 128 x 4 x 4
        #
        # Flatten:
        # 128 * 4 * 4 = 2048


        self.fc1 = nn.Linear(
            128 * 4 * 4,
            128
        )


        self.fc2 = nn.Linear(
            128,
            10
        )


    def forward(self, x):

        # Block 1

        x = self.relu(
            self.conv1(x)
        )

        x = self.relu(
            self.conv2(x)
        )

        x = self.pool1(x)


        # Block 2

        x = self.relu(
            self.conv3(x)
        )

        x = self.relu(
            self.conv4(x)
        )

        x = self.pool2(x)


        # Block 3

        x = self.relu(
            self.conv5(x)
        )

        x = self.pool3(x)


        # Flatten feature maps

        x = torch.flatten(
            x,
            start_dim=1
        )


        # Fully connected layer

        x = self.relu(
            self.fc1(x)
        )


        # Dropout

        x = self.dropout(x)


        # Output layer

        x = self.fc2(x)


        return x


print("CNN class created successfully.")

# %% ============================================================
# CELL 12 - CREATE MODEL AND MOVE IT TO GPU
# ============================================================

model = CNN()

model = model.to(device)

print(model)

print()
print(
    "Model device:",
    next(model.parameters()).device
)

# %% ============================================================
# CELL 13 - COUNT MODEL PARAMETERS
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    f"Total parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

# %% ============================================================
# CELL 14 - DEFINE LOSS AND OPTIMIZER
# ============================================================

# CrossEntropyLoss is appropriate for
# multi-class classification.

criterion = nn.CrossEntropyLoss()


# Adam updates CNN weights during training.

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


print("Loss function:", criterion)
print("Optimizer:", optimizer)

# %% ============================================================
# CELL 15 - PREPARE TRAINING HISTORY
# ============================================================

train_losses = []
validation_losses = []

train_accuracies = []
validation_accuracies = []

best_validation_accuracy = 0.0

print("Training history initialized.")

# %% ============================================================
# CELL 16 - TRAIN THE CNN
# ============================================================

epochs = 20


for epoch in range(epochs):

    # ==========================================================
    # TRAINING PHASE
    # ==========================================================

    model.train()


    total_training_loss = 0.0

    training_correct = 0

    training_total = 0


    for images, labels in train_loader:

        # Move data to GPU

        images = images.to(device)

        labels = labels.to(device)


        # Clear old gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(images)


        # Calculate loss

        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        # Add loss

        total_training_loss += (
            loss.item()
        )


        # Find predicted class

        _, predicted = torch.max(
            outputs,
            dim=1
        )


        training_total += (
            labels.size(0)
        )


        training_correct += (
            predicted == labels
        ).sum().item()


    # Average training loss

    train_loss = (
        total_training_loss /
        len(train_loader)
    )


    # Training accuracy

    train_accuracy = (
        100 *
        training_correct /
        training_total
    )


    # ==========================================================
    # VALIDATION PHASE
    # ==========================================================

    model.eval()


    total_validation_loss = 0.0

    validation_correct = 0

    validation_total = 0


    # No gradients needed during validation

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            total_validation_loss += (
                loss.item()
            )


            _, predicted = torch.max(
                outputs,
                dim=1
            )


            validation_total += (
                labels.size(0)
            )


            validation_correct += (
                predicted == labels
            ).sum().item()


    validation_loss = (
        total_validation_loss /
        len(val_loader)
    )


    validation_accuracy = (
        100 *
        validation_correct /
        validation_total
    )


    # ==========================================================
    # STORE HISTORY
    # ==========================================================

    train_losses.append(
        train_loss
    )

    validation_losses.append(
        validation_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    validation_accuracies.append(
        validation_accuracy
    )


    # ==========================================================
    # SAVE BEST MODEL
    # ==========================================================

    if (
        validation_accuracy
        >
        best_validation_accuracy
    ):

        best_validation_accuracy = (
            validation_accuracy
        )


        torch.save(
            model.state_dict(),
            "best_cifar10_cnn.pth"
        )


    # ==========================================================
    # DISPLAY RESULTS
    # ==========================================================

    print(
        f"Epoch "
        f"{epoch + 1:02d}/{epochs} | "
        f"Train Loss: "
        f"{train_loss:.4f} | "
        f"Train Acc: "
        f"{train_accuracy:.2f}% | "
        f"Val Loss: "
        f"{validation_loss:.4f} | "
        f"Val Acc: "
        f"{validation_accuracy:.2f}%"
    )


print()
print("Training completed.")

print(
    f"Best Validation Accuracy: "
    f"{best_validation_accuracy:.2f}%"
)

# %% ============================================================
# CELL 17 - TRAINING VS VALIDATION ACCURACY GRAPH
# ============================================================

epoch_numbers = range(
    1,
    len(train_accuracies) + 1
)


plt.figure(
    figsize=(9, 5)
)


plt.plot(
    epoch_numbers,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)


plt.plot(
    epoch_numbers,
    validation_accuracies,
    marker="o",
    label="Validation Accuracy"
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(
    "accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()

# %% ============================================================
# CELL 18 - TRAINING VS VALIDATION LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.plot(
    epoch_numbers,
    train_losses,
    marker="o",
    label="Training Loss"
)


plt.plot(
    epoch_numbers,
    validation_losses,
    marker="o",
    label="Validation Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(
    "loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()

# %% ============================================================
# CELL 19 - LOAD BEST TRAINED MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        "best_cifar10_cnn.pth",
        map_location=device
    )
)


model = model.to(device)

model.eval()


print(
    "Best validation model loaded."
)

# %% ============================================================
# CELL 20 - FINAL TEST EVALUATION
# ============================================================

model.eval()


total_test_loss = 0.0

test_correct = 0

test_total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        total_test_loss += (
            loss.item()
        )


        _, predicted = torch.max(
            outputs,
            dim=1
        )


        test_total += (
            labels.size(0)
        )


        test_correct += (
            predicted == labels
        ).sum().item()


test_loss = (
    total_test_loss /
    len(test_loader)
)


test_accuracy = (
    100 *
    test_correct /
    test_total
)


print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)


print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

# %% ============================================================
# CELL 21 - GET SAMPLE PREDICTIONS
# ============================================================

images, labels = next(
    iter(test_loader)
)


images_gpu = images.to(device)


model.eval()


with torch.no_grad():

    outputs = model(
        images_gpu
    )


    # Convert logits to probabilities
    probabilities = torch.softmax(
        outputs,
        dim=1
    )


    confidence, predicted = torch.max(
        probabilities,
        dim=1
    )


# Bring results back to CPU

predicted = predicted.cpu()

confidence = confidence.cpu()


print(
    "Predictions generated."
)

# %% ============================================================
# CELL 22 - DISPLAY SAMPLE PREDICTIONS
# ============================================================

plt.figure(
    figsize=(15, 9)
)


for i in range(15):

    plt.subplot(
        3,
        5,
        i + 1
    )


    image = images[i].clone()


    # Undo normalization

    image = (
        image * 0.5
        + 0.5
    )


    image = image.permute(
        1,
        2,
        0
    )


    image = image.numpy()


    image = np.clip(
        image,
        0,
        1
    )


    plt.imshow(image)


    actual_class = class_names[
        labels[i].item()
    ]


    predicted_class = class_names[
        predicted[i].item()
    ]


    confidence_percentage = (
        confidence[i].item()
        * 100
    )


    plt.title(
        f"Actual: {actual_class}\n"
        f"Pred: {predicted_class}\n"
        f"Confidence: "
        f"{confidence_percentage:.1f}%",
        fontsize=9
    )


    plt.axis("off")


plt.suptitle(
    "CIFAR-10 CNN Predictions"
)


plt.tight_layout()


plt.savefig(
    "sample_predictions.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()

# %% ============================================================
# CELL 23 - SHOW PROBABILITIES FOR ONE IMAGE
# ============================================================

sample_index = 0


actual_label = (
    labels[
        sample_index
    ].item()
)


predicted_label = (
    predicted[
        sample_index
    ].item()
)


print(
    "Actual:",
    class_names[
        actual_label
    ]
)


print(
    "Predicted:",
    class_names[
        predicted_label
    ]
)


print()
print("Probabilities:")


for i, class_name in enumerate(
    class_names
):

    probability = (
        probabilities[
            sample_index,
            i
        ]
        .cpu()
        .item()
        * 100
    )


    print(
        f"{class_name:12s}: "
        f"{probability:6.2f}%"
    )

# %% ============================================================
# CELL 24 - PER-CLASS TEST ACCURACY
# ============================================================

class_correct = [
    0
] * 10

class_total = [
    0
] * 10


model.eval()


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        outputs = model(images)


        _, predicted = torch.max(
            outputs,
            dim=1
        )


        for label, prediction in zip(
            labels,
            predicted
        ):

            label_index = (
                label.item()
            )

            class_total[
                label_index
            ] += 1


            if (
                prediction.item()
                ==
                label_index
            ):

                class_correct[
                    label_index
                ] += 1


print(
    "Accuracy by class:"
)


for i, class_name in enumerate(
    class_names
):

    accuracy = (
        100 *
        class_correct[i] /
        class_total[i]
    )


    print(
        f"{class_name:12s}: "
        f"{accuracy:.2f}%"
    )

# %% ============================================================
# CELL 25 - SAVE FINAL MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "cifar10_cnn_final.pth"
)


print(
    "Final model saved as:"
)

print(
    "cifar10_cnn_final.pth"
)

# %% ============================================================
# CELL 26 - GPU MEMORY INFORMATION
# ============================================================

if torch.cuda.is_available():

    gpu_name = (
        torch.cuda.get_device_name(0)
    )


    allocated_memory = (
        torch.cuda.memory_allocated(0)
        / (1024 ** 3)
    )


    reserved_memory = (
        torch.cuda.memory_reserved(0)
        / (1024 ** 3)
    )


    print(
        "GPU:",
        gpu_name
    )


    print(
        f"Allocated GPU memory: "
        f"{allocated_memory:.2f} GB"
    )


    print(
        f"Reserved GPU memory: "
        f"{reserved_memory:.2f} GB"
    )
# %%
