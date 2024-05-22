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

class Perceptron(nn.Module):
    def __init__(self, layers):
        super(Perceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()

        for i, layer in enumerate(layers):
            self.layers.append(nn.Linear(layer["input_size"], layer["output_size"]))
            if i < len(layers) - 1:  
                self.layers.append(nn.ReLU()) 

    def forward(self, x):
        x = self.flatten(x)  
        for layer in self.layers:
            x = layer(x)
        return x


def plot_samples(samples, title):
    plt.figure(figsize=(10, 5))
    for i, (image, label, pred) in enumerate(samples):
        image = image.cpu().numpy().transpose((1, 2, 0))  # Rearrange dimensions for matplotlib
        mean = np.array([0.5])
        std = np.array([0.5])
        image = std * image + mean  # Unnormalize
        image = np.clip(image, 0, 1)
        plt.subplot(2, 5, i + 1)
        plt.imshow(image[:, :, 0], cmap='gray')
        plt.title(f'Label: {label}\nPred: {pred}')
        plt.axis('off')
    plt.suptitle(title)

device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

transform = transforms.Compose([ 
    transforms.ToTensor(),   
    transforms.Normalize((0.5,), (0.5,)) 
])

train_set = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False)

layers = [{"input_size": 784, "output_size": 64},
          {"input_size": 64, "output_size": 64},
          {"input_size": 64, "output_size": 32},
          {"input_size": 32, "output_size": 10}]
model = Perceptron(layers=layers)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scaler = GradScaler()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
correct_samples = []
incorrect_samples = []
timestamp = time.strftime("%Y%m%d-%H%M%S")
save_path = "saved_models/model_"+timestamp+"/my_model_epoch_{}.pth"

num_epochs = 100
for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)  # Get the indices of max logit which are the predicted classes
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    train_accuracy = 100 * total_correct / total_samples
    train_losses.append(loss.item())
    train_accuracies.append(train_accuracy)


    model.eval()
    total_test_correct = 0
    total_test_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
                test_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total_test_correct += (predicted == labels).sum().item()
                total_test_samples += labels.size(0)

                matches = predicted == labels
                for i in range(images.size(0)):
                    if len(correct_samples) < 10 and matches[i]: 
                        correct_samples.append((images[i], labels[i], predicted[i]))
                    elif len(incorrect_samples) < 10 and not matches[i]:
                        incorrect_samples.append((images[i], labels[i], predicted[i]))

    test_accuracy = 100 * total_test_correct / total_test_samples
    test_losses.append(test_loss.item())
    test_accuracies.append(test_accuracy)

    scheduler.step(loss)
    
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.2f}%')
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), save_path.format(epoch + 1))
        print(f'Saved model at epoch {epoch + 1}')





# Plotting training and testing loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting training and testing accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plot_samples(correct_samples, "Correctly Predicted Samples")
plot_samples(incorrect_samples, "Incorrectly Predicted Samples")

plt.tight_layout()
plt.show()