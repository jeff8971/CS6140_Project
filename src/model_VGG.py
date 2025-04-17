#!/usr/bin/env python3
"""
Filename: model_VGG.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script implements a VGG-like convolutional neural network for facial expression recognition.
             It prepares the image data with specific transformations, constructs the VGG model, and trains it
             using stochastic gradient descent with momentum. The script evaluates both training and validation
             performance, updates the model weights, and saves the trained model. Configured for CPU, this script
             is intended for use with facial image datasets organized in specified directory paths.
Date: 2025-04-09
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def initialize_data_loaders(batch_size):
    """
    Initializes data loaders for training and validation datasets with appropriate transformations.
    
    Args:
        batch_size (int): Number of samples per batch
        
    Returns:
        tuple: (train_loader, valid_loader) containing the data loaders for training and validation
    """
    # Training transforms with data augmentation
    transforms_train = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Randomly adjust brightness and contrast
        transforms.ToTensor(),  # Convert to PyTorch tensor
    ])

    # Validation transforms (no augmentation)
    transforms_valid = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.ToTensor(),  # Convert to PyTorch tensor
    ])

    # Create datasets
    train_dataset = torchvision.datasets.ImageFolder(root='face_images/vgg_train_set', transform=transforms_train)
    valid_dataset = torchvision.datasets.ImageFolder(root='face_images/vgg_valid_set', transform=transforms_valid)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

class VGG(nn.Module):
    """
    VGG-like model for facial expression recognition.
    The architecture consists of:
    - Three blocks of convolutional layers with max pooling
    - Three fully connected layers with dropout
    """
    def __init__(self, num_classes=7):
        """
        Args:
            num_classes (int): Number of output classes (emotions)
        """
        super(VGG, self).__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block: two 3x3 conv layers with 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample by factor of 2
            
            # Second block: two 3x3 conv layers with 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample by factor of 2
            
            # Third block: two 3x3 conv layers with 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Downsample by factor of 2
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(128 * 6 * 6, 4096), nn.ReLU(),  # First fully connected layer
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(4096, 4096), nn.ReLU(),  # Second fully connected layer
            nn.Linear(4096, num_classes)  # Output layer
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)  # Extract features
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.classifier(x)  # Classify
        return x

def train_model(model, train_loader, valid_loader, epochs, learning_rate):
    """
    Trains the VGG model for the specified number of epochs.
    
    Args:
        model (nn.Module): The VGG model to train
        train_loader (DataLoader): DataLoader for training data
        valid_loader (DataLoader): DataLoader for validation data
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer
    """
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        train_epoch(model, train_loader, optimizer, criterion)
        validate_epoch(model, valid_loader, criterion)
        # Save model checkpoint every 10 epochs and at the end
        if epoch % 10 == 0 or epoch == epochs:
            torch.save(model.state_dict(), f'model_vgg_epoch_{epoch}.pth')

def train_epoch(model, train_loader, optimizer, criterion):
    """
    Trains the model for one epoch.
    
    Args:
        model (nn.Module): The VGG model
        train_loader (DataLoader): DataLoader for training data
        optimizer (Optimizer): Optimizer for training
        criterion (Loss): Loss function for training
    """
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    print(f'Training - Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}')

def validate_epoch(model, valid_loader, criterion):
    """
    Validates the model for one epoch.
    
    Args:
        model (nn.Module): The VGG model
        valid_loader (DataLoader): DataLoader for validation data
        criterion (Loss): Loss function for validation
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    print(f'Validation - Loss: {total_loss / len(valid_loader):.4f}, Accuracy: {correct / total:.4f}')

if __name__ == '__main__':
    # Global constants
    DEVICE = torch.device('cpu')  # Device to run the model on
    BATCH_SIZE = 128  # Number of samples per batch
    LEARNING_RATE = 0.01  # Learning rate for the optimizer
    EPOCHS = 60  # Number of training epochs
    
    # Initialize data loaders and model
    train_loader, valid_loader = initialize_data_loaders(BATCH_SIZE)
    model = VGG().to(DEVICE)
    
    # Train the model
    train_model(model, train_loader, valid_loader, EPOCHS, LEARNING_RATE)
