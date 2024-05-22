import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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


class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)  # batch normalization after conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # batch normalization after conv2
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)  #reduce overfitting
        
        # fully connected layers
        self.flatten = nn.Flatten()  # flatten it for fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        #convulusional
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        
        #dropout
        x = self.dropout(x)
        
        #fully connected layers
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Ensure that the tensor is a PyTorch tensor
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")

        # Add Gaussian noise with the correct size
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


def plot_samples(samples, title):
    plt.figure(figsize=(10, 5))
    for i, (image, label, pred) in enumerate(samples):
        image = image.cpu().numpy().transpose((1, 2, 0))  #rearrange for matplotlib
        mean = np.array([0.5])
        std = np.array([0.5])
        image = std * image + mean  #unnormalize
        image = np.clip(image, 0, 1)
        plt.subplot(2, 5, i + 1)
        plt.imshow(image[:, :, 0], cmap="gray")
        plt.title(f"Label: {label}\nPred: {pred}")
        plt.axis("off")
    plt.suptitle(title)

#make cuda for gpu use
device = "cuda" if torch.cuda.is_available() else "cpu"

#transformer
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # rotate random 10 degrees
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # for zero mean and unit variance
])


train_set = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, pin_memory=True)

model = Perceptron().to(device)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  

scaler = GradScaler()
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10)
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []
epsilon = 1e-10


timestamp = time.strftime("%Y%m%d-%H%M%S")

save_dir = "./models/D"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

weight_dir = f"{save_dir}/model_{timestamp}/weights"
results_dir = f"{save_dir}/model_{timestamp}/results"

if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

save_path = weight_dir+"/epoch_{}.pth"


model.eval()
initial_predictions = []
initial_targets = []

with torch.no_grad(): #no training for first test
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        initial_loss = criterion(outputs + epsilon, labels) * labels.size(0)

        _, predicted = torch.max(outputs, 1) 
        
        initial_predictions.extend(predicted.cpu().numpy())
        initial_targets.extend(labels.cpu().numpy())
        
initial_accuracy = 100 * (np.array(initial_predictions) == np.array(initial_targets)).sum() / len(initial_targets)

print(f"First test, initial loss: {initial_loss.item():.20f}, initial accuracy: {initial_accuracy}")
with open(results_dir + timestamp, "a") as f:
    f.write(f"First test, initial loss: {initial_loss.item():.20f}, initial accuracy: {initial_accuracy}")

num_epochs = 50
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
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs + epsilon, labels) * labels.size(0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, predicted = torch.max(outputs, 1)  
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        train_predictions.extend(predicted.cpu().numpy())  
        train_targets.extend(labels.cpu().numpy())  

    train_accuracy = 100 * total_correct / total_samples
    train_losses.append(loss.item())
    train_accuracies.append(train_accuracy)


    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels= labels.to(device)
            with autocast():
                outputs = model(images)
            test_loss = criterion(outputs + epsilon, labels) * labels.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total_test_correct += (predicted == labels).sum().item()
            total_test_samples += labels.size(0)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

            matches = predicted == labels
            for i in range(images.size(0)):
                if matches[i]: 
                    correct_samples.append((images[i], labels[i], predicted[i]))
                elif not matches[i]:
                    incorrect_samples.append((images[i], labels[i], predicted[i]))

    test_accuracy = 100 * total_test_correct / total_test_samples
    test_losses.append(test_loss.item())
    test_accuracies.append(test_accuracy)

    scheduler.step(loss)
    
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.20f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss.item():.20f}, Test Accuracy: {test_accuracy:.2f}%")
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), save_path.format(epoch + 1))
        print(f"Saved model at epoch {epoch + 1}")
        cm = confusion_matrix(test_targets, test_predictions)
        report = classification_report(test_targets, test_predictions, digits=3)
        print(f"Confusion Matrix after Epoch {epoch + 1}:\n{cm}")
        print(f"Classification Report after Epoch {epoch + 1}:\n{report}")

        with open(results_dir + timestamp, "a") as f:
            f.write(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.20f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss.item():.20f}, Test Accuracy: {test_accuracy:.2f}%")
            f.write(f"Confusion Matrix after Epoch {epoch + 1}:\n{cm}")
            f.write(f"Classification Report after Epoch {epoch + 1}:\n{report}")

print(f"Number of wrong guesses in last training = {len(incorrect_samples)}")
print(f"Number of right guesses in last training = {len(correct_samples)}")

# Plotting loss
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

# Plotting 
plot_samples(correct_samples[0:10], "Correctly Predicted Samples")
plot_samples(incorrect_samples[0:10], "Incorrectly Predicted Samples")

plt.tight_layout()
plt.show()