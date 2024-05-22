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
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=6, stride=1, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)  #reduce overfitting
        
        # fully connected layers
        self.flatten = nn.Flatten()  # flatten it for fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

        
    def forward(self, x):
        #convulusional
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))

        
        #dropout
        x = self.dropout(x)
        
        #fully connected layers
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
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
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # for zero mean and unit variance
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
model.load_state_dict(torch.load("./models/model_20240422-111854/weights/epoch_50.pth"))

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  

scaler = GradScaler()
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []


model.eval()
initial_predictions = []
initial_targets = []
correct_samples = []
incorrect_samples = []

with torch.no_grad(): #no training for first test
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        initial_loss = criterion(outputs, labels) * labels.size(0)

        _, predicted = torch.max(outputs, 1) 
        
        initial_predictions.extend(predicted.cpu().numpy())
        initial_targets.extend(labels.cpu().numpy())

        matches = predicted == labels
        for i in range(images.size(0)):
            if matches[i]: 
                correct_samples.append((images[i], labels[i], predicted[i]))
            elif not matches[i]:
                incorrect_samples.append((images[i], labels[i], predicted[i]))


initial_accuracy = 100 * (np.array(initial_predictions) == np.array(initial_targets)).sum() / len(initial_targets)

print(f"First test, initial loss: {initial_loss.item():.20f}, initial accuracy: {initial_accuracy}%")
print(f"Number of wrong guesses in last training = {len(incorrect_samples)}")
print(f"Number of right guesses in last training = {len(correct_samples)}")


plot_samples(correct_samples[0:10], "Correctly Predicted Samples")
plot_samples(incorrect_samples[0:10], "Incorrectly Predicted Samples")

plt.tight_layout()
plt.show()

