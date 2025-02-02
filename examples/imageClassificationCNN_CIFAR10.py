#!/usr/bin/env python3

"""
Convolutional Neural Network (CNN) for CIFAR-10
______________________________________________________
Note: Cross-entropy is used for multi-class classification here
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

def prep_data(batch_size=64, for_training=True, shuffle=True):
    
    # CIFAR-10 = (3, 32, 32)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    if for_training:
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
    else:
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader, dataset


class CNN(nn.Module):

    def __init__(self):
        
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 8 * 8)
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(trainloader, net, criterion, optimizer, epochs=5):
    
    losses = []
    accuracies = []

    for epoch in tqdm(range(epochs), desc="Training epochs"):

        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100.0 * correct / total

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        tqdm.write(f"[Epoch {epoch+1}] Loss: {epoch_loss:.3f} | Accuracy: {epoch_accuracy:.2f}%")

    return losses, accuracies


def evaluate_model(testloader, net):
    
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * correct / total
    print(f"Accuracy of test set: {test_accuracy:.2f}%")
    return predictions, test_accuracy


def plot_metrics(losses, accuracies):
    
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, color='blue', label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, color='green', label='Training Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion(predictions, dataset):
    
    true_labels = dataset.targets 
    cm = confusion_matrix(true_labels, predictions)

    # CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()


def feature_maps_visual(net, image, layer="conv1"):
    
    x = image.unsqueeze(0)
    
    with torch.no_grad():
        if layer == "conv1":
            out = net.conv1(x)
        elif layer == "conv2":
            out = net.conv2(net.pool(torch.relu(net.conv1(x))))
        else:
            raise ValueError("layer must be 'conv1' or 'conv2'")

    out = torch.relu(out)
    
    feature_maps = out.cpu().numpy()[0]  

    num_maps = feature_maps.shape[0]
    plt.figure(figsize=(15, 8))
    for i in range(num_maps):
        plt.subplot(4, 8, i+1)  
        fm = feature_maps[i]
        fm_min, fm_max = fm.min(), fm.max()
        fm = (fm - fm_min)/(fm_max - fm_min + 1e-5)
        plt.imshow(fm, cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Feature Maps after {layer}")
    plt.show()


if __name__ == "__main__":

    learning_rates = [0.001, 0.01]
    batch_sizes = [32, 64]
    num_epochs = 5

    best_accuracy = 0.0
    best_lr = None
    best_bs = None
    best_losses = None
    best_accuracies = None
    best_predictions = None

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\nTraining model with Learning Rate={lr}, Batch Size={bs}")

            trainloader, _ = prep_data(batch_size=bs, for_training=True, shuffle=True)
            testloader, testset = prep_data(batch_size=bs, for_training=False, shuffle=False)

            net = CNN()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=lr)

            # Train
            losses, accuracies = train_model(trainloader, net, criterion, optimizer, num_epochs)

            # Test
            predictions, test_acc = evaluate_model(testloader, net)

            if accuracies[-1] > best_accuracy:
                best_accuracy = accuracies[-1]
                best_lr = lr
                best_bs = bs
                best_losses = losses
                best_accuracies = accuracies
                best_predictions = predictions

    print("\nBest Accuracy: %.2f%%" % best_accuracy)
    print("Best Learning Rate:", best_lr)
    print("Best Batch Size:", best_bs)

    plot_metrics(best_losses, best_accuracies)

    plot_confusion(best_predictions, testset)

    print("\nClassification Report for the best model:")
    print(classification_report(testset.targets, best_predictions, target_names=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')))

    images, labels = next(iter(testloader))
    test_img = images[0]
    
    feature_maps_visual(net, test_img, layer="conv1")
