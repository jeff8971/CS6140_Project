#!/usr/bin/env python3
"""
Filename: test_MTCNN_RESNET.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script implements real-time facial expression recognition using MTCNN for face detection and a ResNet model for classification.
             It processes video input to detect faces, then classifies each detected face into one of seven emotion categories using the trained ResNet model.
             Results are displayed in real-time with labeled bounding boxes around detected faces. The system is designed to run on CPU with optimizations for real-time performance.
Date: 2025-04-09
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from facenet_pytorch import MTCNN

def preprocess_input(images):
    """
    Normalize facial data by mapping pixel values from 0-255 to 0-1.
    
    Args:
        images (numpy.ndarray): Input image array with pixel values in range [0, 255]
        
    Returns:
        numpy.ndarray: Normalized image array with pixel values in range [0, 1]
    """
    return images / 255.0

class GlobalAvgPool2d(nn.Module):
    """
    Global Average Pooling 2D layer.
    Reduces spatial dimensions to 1x1 by taking the average of all values in each channel.
    """
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class Residual(nn.Module):
    """
    Residual block for ResNet architecture.
    Implements the skip connection that allows gradients to flow through the network more effectively.
    """
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            use_1x1conv (bool): Whether to use 1x1 convolution in the shortcut connection
            stride (int): Stride for the first convolution
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        """
        Forward pass through the residual block.
        
        Args:
            X (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after applying the residual block
        """
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    """
    Creates a block of residual layers for the ResNet architecture.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_residuals (int): Number of residual blocks in this stage
        first_block (bool): Whether this is the first block in the network
        
    Returns:
        nn.Sequential: A sequence of residual blocks
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

class ResNet(nn.Module):
    """
    Full ResNet model for emotion classification.
    The architecture consists of:
    - Initial convolution and pooling
    - Four stages of residual blocks
    - Global average pooling
    - Final fully connected layer
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Initial convolution and pooling
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # Input: 1 channel, Output: 64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual blocks
            resnet_block(64, 64, 2, first_block=True),  # First stage
            resnet_block(64, 128, 2),  # Second stage
            resnet_block(128, 256, 2),  # Third stage
            resnet_block(256, 512, 2),  # Fourth stage
            
            # Final layers
            GlobalAvgPool2d(),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(512, 7)  # Output layer for 7 emotion categories
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 7)
        """
        return self.network(x)

def initialize_models():
    """
    Initialize MTCNN for face detection and ResNet for emotion classification.
    
    Returns:
        tuple: (mtcnn, emotion_classifier, emotion_labels)
    """
    mtcnn = MTCNN(keep_all=True, device='cpu')  # Initialize MTCNN face detector
    emotion_classifier = ResNet()  # Create ResNet model
    emotion_classifier.load_state_dict(torch.load('model/model_resnet.pkl', map_location=torch.device('cpu')))  # Load trained weights
    emotion_classifier.eval()  # Set to evaluation mode
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    return mtcnn, emotion_classifier, emotion_labels

def process_frame(frame, mtcnn, emotion_classifier, emotion_labels):
    """
    Process each frame to detect faces and classify emotions.
    
    Args:
        frame (numpy.ndarray): Input video frame
        mtcnn (MTCNN): Face detection model
        emotion_classifier (nn.Module): Emotion classification model
        emotion_labels (dict): Mapping of emotion indices to labels
        
    Returns:
        numpy.ndarray: Processed frame with face detection and emotion labels
    """
    frame = cv2.flip(frame, 1)  # Horizontal flip for mirror effect
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            x, y, x2, y2 = map(int, box)
            # Extract and preprocess face
            face = cv2.cvtColor(frame[y:y2, x:x2], cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = preprocess_input(np.array(face, dtype=np.float32)).reshape(1, 1, 48, 48)
            face_tensor = torch.from_numpy(face).type(torch.FloatTensor)
            
            # Predict emotion
            with torch.no_grad():
                emotion_pred = emotion_classifier(face_tensor)
                emotion_arg = torch.argmax(emotion_pred).item()
                emotion = emotion_labels[emotion_arg]
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x2, y2), (84, 255, 159), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def main():
    """
    Main function that:
    1. Initializes models and video capture
    2. Processes video frames in real-time
    3. Displays results and handles user input
    """
    # Initialize models and video capture
    mtcnn, emotion_classifier, emotion_labels = initialize_models()
    video_capture = cv2.VideoCapture(0)  # Open default camera
    cv2.namedWindow('window_frame')  # Create display window
    
    # Main processing loop
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        # Process frame and display results
        frame = process_frame(frame, mtcnn, emotion_classifier, emotion_labels)
        cv2.imshow('window_frame', frame)
        
        # Handle quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
