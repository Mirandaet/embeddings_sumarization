import torch
import torch.nn as nn
import torch.nn.functional as F

class AnimalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AnimalClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer to reduce dimensions
        self.fc1 = nn.Linear(64 * 32 * 32, 256)  # Fully connected layer
        self.fc2 = nn.Linear(256, num_classes)  # Output layer with num_classes
        
    def forward(self, x):
        # Forward pass through the model
        x = F.relu(self.conv1(x))  # ReLU activation for the first convolution
        x = self.pool(x)  # Apply pooling
        x = F.relu(self.conv2(x))  # ReLU activation for the second convolution
        x = self.pool(x)  # Apply pooling again
        x = x.view(-1, 64 * 32 * 32)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))  # ReLU activation for the first fully connected layer
        x = self.fc2(x)  # Output layer with no activation (suitable for classification)
        return x


