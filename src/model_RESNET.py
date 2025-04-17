#!/usr/bin/env python3
"""
Filename: model_RESNET.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script implements a ResNet model for facial expression recognition. It includes functions for loading image data with transformations,
             constructing a ResNet architecture, training the model with stochastic gradient descent, and monitoring training and validation
             progress with loss and accuracy metrics. The model state is saved upon completion. The script runs on the CPU and uses a custom
             data loading strategy for image datasets.
Date: 2025-04-09
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F

def setup_constants():
    """
    Initializes global constants for the training process:
    - Batch size for training
    - Learning rate for the optimizer
    - Number of training epochs
    - Device to run the model on (CPU in this case)
    """
    global BATCH_SIZE, LEARNING_RATE, EPOCHS, DEVICE
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    EPOCHS = 60
    DEVICE = torch.device('cpu')

def setup_paths():
    """
    Sets up the paths for training and validation datasets.
    These paths should point to directories containing the facial expression images.
    """
    global TRAIN_PATH, VALID_PATH
    TRAIN_PATH = 'face_images/resnet_train_set'
    VALID_PATH = 'face_images/resnet_valid_set'

def setup_transforms():
    """
    Configures image transformations for training and validation:
    - Training transforms include random horizontal flips and color jittering for data augmentation
    - Validation transforms only include basic preprocessing
    """
    global transforms_train, transforms_valid
    transforms_train = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Randomly adjust brightness and contrast
        transforms.ToTensor(),  # Convert to PyTorch tensor
    ])
    transforms_valid = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.ToTensor(),  # Convert to PyTorch tensor
    ])

def setup_datasets_and_loaders():
    """
    Creates PyTorch datasets and data loaders for training and validation:
    - Uses ImageFolder to load images from the specified directories
    - Applies the configured transforms to the images
    - Creates data loaders with the specified batch size
    """
    train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_PATH,
                                                     transform=transforms_train)
    valid_dataset = torchvision.datasets.ImageFolder(root=VALID_PATH,
                                                     transform=transforms_valid)
    global train_loader, valid_loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=BATCH_SIZE, shuffle=False)

class ResidualBlock(nn.Module):
    """
    Implements a residual block for the ResNet architecture.
    Each block consists of two convolutional layers with batch normalization and ReLU activation.
    Includes a shortcut connection that can be used to match dimensions when needed.
    """
    def __init__(self, in_channels, out_channels, stride=1, use_1x1conv=False):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the first convolution
            use_1x1conv (bool): Whether to use a 1x1 convolution in the shortcut connection
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=stride) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after applying the residual block
        """
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3 is not None:
            identity = self.conv3(identity)
        out += identity
        return F.relu(out)

def make_resnet():
    """
    Constructs a ResNet model for facial expression recognition.
    The architecture includes:
    - Initial convolution and pooling layers
    - Four stages of residual blocks
    - Final average pooling and fully connected layer
    
    Returns:
        nn.Sequential: The complete ResNet model
    """
    layers = []
    # Initial convolution and pooling
    layers.append(nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # Initial convolution
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))

    # Residual blocks configuration
    in_channels = 64
    num_blocks_list = [2, 2, 2, 2]  # Number of blocks in each stage
    out_channels_list = [64, 128, 256, 512]  # Output channels for each stage
    strides_list = [1, 2, 2, 2]  # Stride for the first block in each stage

    # Build residual blocks
    for out_channels, num_blocks, stride in zip(out_channels_list,
                                                num_blocks_list, strides_list):
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride,
                                    use_1x1conv=(in_channels != out_channels)))
        in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels))

    # Final layers
    layers.append(nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        nn.Flatten(),
        nn.Linear(512, 7)  # Output layer for 7 emotion classes
    ))

    model = nn.Sequential(*layers).to(DEVICE)
    return model

def train_epoch(model, device, train_loader, optimizer, criterion):
    """
    Trains the model for one epoch.
    
    Args:
        model (nn.Module): The ResNet model
        device (torch.device): Device to run the model on
        train_loader (DataLoader): DataLoader for training data
        optimizer (Optimizer): Optimizer for training
        criterion (Loss): Loss function for training
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    print(f"Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

def validate_epoch(model, device, valid_loader, criterion):
    """
    Validates the model for one epoch.
    
    Args:
        model (nn.Module): The ResNet model
        device (torch.device): Device to run the model on
        valid_loader (DataLoader): DataLoader for validation data
        criterion (Loss): Loss function for validation
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(valid_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

def main():
    """
    Main function that:
    1. Sets up constants, paths, and transforms
    2. Creates datasets and data loaders
    3. Initializes the model, optimizer, and loss function
    4. Trains the model for the specified number of epochs
    5. Saves the model after each epoch
    """
    setup_constants()
    setup_paths()
    setup_transforms()
    setup_datasets_and_loaders()
    model = make_resnet()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_epoch(model, DEVICE, train_loader, optimizer, criterion)
        validate_epoch(model, DEVICE, valid_loader, criterion)
        torch.save(model.state_dict(), 'resnet_model.pth')

if __name__ == '__main__':
    main()
