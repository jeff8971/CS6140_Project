#!/usr/bin/env python3
"""
Filename: model_CNN.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script defines and trains a convolutional neural network (CNN) for facial expression recognition.
             It manages the entire pipeline, from loading and preprocessing image data to training the model and
             validating its performance. The script uses PyTorch for model construction and training, and custom
             data loaders for handling image datasets.
Date: 2025-04-09
"""

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2

def gaussian_weights_init(m):
    """
    Applies Gaussian weights initialization to convolutional layers.
    
    Args:
        m (nn.Module): A PyTorch module to initialize weights for
    """
    classname = m.__class__.__name__
    if 'Conv' in classname:
        m.weight.data.normal_(0.0, 0.04)

def validate(model, dataset, batch_size):
    """
    Validates the model on a dataset and returns the accuracy.
    
    Args:
        model (nn.Module): The trained CNN model
        dataset (Dataset): The validation dataset
        batch_size (int): Number of samples per batch
        
    Returns:
        float: Validation accuracy
    """
    val_loader = data.DataLoader(dataset, batch_size=batch_size)
    correct, total = 0, 0
    for images, labels in val_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return correct / total

class FaceDataset(data.Dataset):
    """
    Custom dataset for facial expression recognition. Loads images from disk based on paths in a CSV file.
    The dataset handles image preprocessing including grayscale conversion, histogram equalization,
    and normalization.
    """
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and corresponding labels.
        """
        self.data_info = pd.read_csv(csv_file, header=None)
        self.paths = self.data_info[0].values
        self.labels = self.data_info[1].astype(int).values

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Loads and preprocesses a single image and its label.
        
        Args:
            idx (int): Index of the sample to load
            
        Returns:
            tuple: (image_tensor, label) where image_tensor is a normalized grayscale image
                  and label is the emotion class
        """
        img_path = self.paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.equalizeHist(image) / 255.0  # Normalize and perform histogram equalization
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = self.labels[idx]
        return image_tensor, label

class FaceCNN(nn.Module):
    """
    CNN model for facial expression recognition with three convolutional layers and three fully connected layers.
    The architecture includes:
    - Three convolutional blocks (Conv2d + BatchNorm + RReLU + MaxPool)
    - Four fully connected layers with dropout for regularization
    - Gaussian weight initialization for convolutional layers
    """
    def __init__(self):
        super(FaceCNN, self).__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 64, 3, padding=1),  # Input: 1 channel, Output: 64 channels
            nn.BatchNorm2d(64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second convolutional block
            nn.Conv2d(64, 128, 3, padding=1),  # Input: 64 channels, Output: 128 channels
            nn.BatchNorm2d(128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third convolutional block
            nn.Conv2d(128, 256, 3, padding=1),  # Input: 128 channels, Output: 256 channels
            nn.BatchNorm2d(256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # Dropout for regularization
            nn.Linear(256*6*6, 4096),  # First fully connected layer
            nn.RReLU(inplace=True),
            nn.Dropout(0.5),  # Higher dropout for regularization
            nn.Linear(4096, 1024),  # Second fully connected layer
            nn.RReLU(inplace=True),
            nn.Linear(1024, 256),  # Third fully connected layer
            nn.RReLU(inplace=True),
            nn.Linear(256, 7)  # Output layer (7 emotion classes)
        )
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 7)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Applies Gaussian weight initialization to all convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.04)

def train(model, train_loader, val_loader, epochs, optimizer, criterion):
    """
    Trains and validates the CNN model.
    
    Args:
        model (nn.Module): The CNN model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        epochs (int): Number of training epochs
        optimizer (Optimizer): Optimizer for training
        criterion (Loss): Loss function for training
    """
    for epoch in range(epochs):
        # Training phase
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation phase (every 5 epochs and at the end)
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            train_acc = validate(model, train_loader.dataset, len(train_loader.dataset))
            val_acc = validate(model, val_loader.dataset, len(val_loader.dataset))
            print(f'Epoch {epoch+1}/{epochs}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}')

def main():
    """
    Main function that:
    1. Loads and prepares the datasets
    2. Initializes the model, optimizer, and loss function
    3. Trains the model
    4. Saves the trained model
    """
    # Load datasets
    train_dataset = FaceDataset('train.csv')
    val_dataset = FaceDataset('val.csv')
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Initialize model and training components
    model = FaceCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, train_loader, val_loader, epochs=50, optimizer=optimizer, criterion=criterion)

    # Save model checkpoint
    torch.save(model.state_dict(), 'model_cnn.pth')

if __name__ == '__main__':
    main()
