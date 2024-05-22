import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from torchvision.transforms import v2
from torchvision.io import read_image


class CUBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Load image-to-class mapping with updated delimiter
        image_class_path = os.path.join(data_dir, 'images.txt')
        self.image_class_itterate = pd.read_csv(
            image_class_path, sep=r"\s+", engine='python', header=None)
        # Construct image paths
        self.image_paths = [os.path.join(data_dir, 'images', f"{row[1]}") for _, row in self.image_class_itterate.iterrows()]
        image_class_labels_path = os.path.join(
            data_dir, 'image_class_labels.txt')
        self.image_class_labels = pd.read_csv(
            image_class_labels_path, sep=r"\s+", engine='python', header=None)

    def __len__(self):
        return len(self.image_class_labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Check if file exists to prevent "File not found" error
        if not os.path.exists(image_path):
            raise ValueError(f"File not found: {image_path}")

        image = read_image(image_path)  # Read the image
        # Adjust for zero-based index
        label = int(self.image_class_labels.iloc[idx, 1]) - 1

        if self.transform:
            image = self.transform(image)  # Apply transformation

        return image, label


class GrayscaleToRGB:
    def __call__(self, img):
        # If the image is grayscale, convert it to RGB by repeating the channel
        if img.shape[0] == 1:
            # Repeat the single channel to make it 3-channel (RGB)
            img = img.repeat(3, 1, 1)
        return img
    
def plot_samples(samples, title):
    plt.figure(figsize=(10, 5))
    for i, (image, label, pred) in enumerate(samples):
        image = image.cpu().numpy().transpose((1, 2, 0))  # rearrange for matplotlib
        mean = np.array([0.5])
        std = np.array([0.5])
        image = std * image + mean  # unnormalize
        image = np.clip(image, 0, 1)
        plt.subplot(2, 5, i + 1)
        plt.imshow(image[:, :, 0], cmap="gray")
        plt.title(f"Label: {label}\nPred: {pred}")
        plt.axis("off")
    plt.suptitle(title)



train_transform = transforms.Compose([
    GrayscaleToRGB(),  # Convert grayscale to RGB if needed
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [
                         0.229, 0.224, 0.225])  # Normalization for RGB
])

test_transform = transforms.Compose([
    GrayscaleToRGB(),  # Ensure RGB consistency
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load a pre-trained ResNet model
model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")

# Modify the output layer to match your bird dataset's class count
num_classes = 200  # Change to your specific bird class count
model.fc = nn.Linear(model.fc.in_features, num_classes)


data_dir = "./data/CUB-200/CUB_200_2011/"
device = ("cuda")

save_dir = "./models/CUB-200_pretrained"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

timestamp = time.strftime("%Y%m%d-%H%M%S")

weight_dir = f"{save_dir}/model_{timestamp}/weights"
results_dir = f"{save_dir}/model_{timestamp}/results"

if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

save_path = weight_dir+"/epoch_{}.pth"

# Create custom datasets for training and testing
train_dataset = CUBDataset(data_dir, transform=train_transform)
test_dataset = CUBDataset(data_dir, transform=test_transform)

# Create DataLoaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64,
                         shuffle=False, pin_memory=True)


input_shape = (3, 224, 224)
output_size = 200 
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scaler = GradScaler()
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10)

train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

model.eval()
initial_predictions = []
initial_targets = []

with torch.no_grad(): #no training for first test
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        initial_loss =  F.cross_entropy(outputs, labels)

        _, predicted = torch.max(outputs, 1) 
        
        initial_predictions.extend(predicted.cpu().numpy())
        initial_targets.extend(labels.cpu().numpy())

initial_accuracy = 100 * (np.array(initial_predictions) == np.array(initial_targets)).sum() / len(initial_targets)

print(f"First test, initial loss: {initial_loss.item():.20f}, initial accuracy: {initial_accuracy}")
with open(results_dir + timestamp, "a") as f:
    f.write(f"First test, initial loss: {initial_loss.item():.20f}, initial accuracy: {initial_accuracy} \n")


num_epochs = 100

for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0
    total_test_correct = 0
    total_test_samples = 0

    train_predictions = []
    train_targets = []
    test_predictions = []
    test_targets = []

    correct_samples = []
    incorrect_samples = []

    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True)
        # Forward pass
        with autocast():
            outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    train_accuracy = 100 * total_correct / total_samples
    train_losses.append(loss.item())
    train_accuracies.append(train_accuracy)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            with autocast():
                outputs = model(images)
            test_loss = F.cross_entropy(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total_test_correct += (predicted == labels).sum().item()
            total_test_samples += labels.size(0)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

            matches = predicted == labels
            for i in range(images.size(0)):
                if matches[i]:
                    correct_samples.append(
                        (images[i], labels[i], predicted[i]))
                elif not matches[i]:
                    incorrect_samples.append(
                        (images[i], labels[i], predicted[i]))

    test_accuracy = 100 * total_test_correct / total_test_samples
    test_losses.append(test_loss.item())
    test_accuracies.append(test_accuracy)

    scheduler.step(loss)

    train_accuracy = 100 * total_correct / total_samples
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.20f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss.item():.20f}, Test Accuracy: {test_accuracy:.2f}%")
    
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), save_path.format(epoch + 1))
        print(f"Saved model at epoch {epoch + 1}")
        cm = confusion_matrix(test_targets, test_predictions)
        report = classification_report(test_targets, test_predictions, digits=3)
        print(f"Confusion Matrix after Epoch {epoch + 1}:\n{cm}")
        print(f"Classification Report after Epoch {epoch + 1}:\n{report}")

        with open(results_dir + timestamp, "a") as f:
            f.write(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.20f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss.item():.20f}, Test Accuracy: {test_accuracy:.2f}% \n")
            f.write(f"Confusion Matrix after Epoch {epoch + 1}:\n{cm} \n")
            f.write(f"Classification Report after Epoch {epoch + 1}:\n{report} \n")

print(f"Number of wrong guesses in last training = {len(incorrect_samples)}")
print(f"Number of right guesses in last training = {len(correct_samples)}")

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plotting accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plot_samples(correct_samples[0:10], "Correctly Predicted Samples")
plot_samples(incorrect_samples[0:10], "Incorrectly Predicted Samples")

plt.tight_layout()
plt.show()

