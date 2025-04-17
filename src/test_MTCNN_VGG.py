#!/usr/bin/env python3
"""
Filename: test_MTCNN_VGG.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script combines MTCNN for real-time face detection with a VGG-style convolutional neural network for emotion recognition.
             It operates on live video streams, identifying faces and classifying their emotional expressions into categories such as anger, happiness, and sadness.
             Detected emotions are displayed as overlays on the video feed with corresponding labels.
Date: 2025-04-09
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from facenet_pytorch import MTCNN

def preprocess_input(images):
    """
    Normalize image pixels from 0-255 to 0-1.
    
    Args:
        images (numpy.ndarray): Input image array with pixel values in range [0, 255]
        
    Returns:
        numpy.ndarray: Normalized image array with pixel values in range [0, 1]
    """
    return images / 255.0

class VGG(nn.Module):
    """
    VGG-style neural network for emotion classification.
    The architecture consists of:
    - Three blocks of convolutional layers with max pooling
    - Three fully connected layers with dropout
    """
    def __init__(self):
        super(VGG, self).__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block: two 3x3 conv layers with 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: 1 channel, Output: 32 channels
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Input: 32 channels, Output: 32 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample by factor of 2
            
            # Second block: two 3x3 conv layers with 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Input: 32 channels, Output: 64 channels
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Input: 64 channels, Output: 64 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample by factor of 2
            
            # Third block: two 3x3 conv layers with 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Input: 64 channels, Output: 128 channels
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Input: 128 channels, Output: 128 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Downsample by factor of 2
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 4096),  # First fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(4096, 4096),  # Second fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(4096, 7)  # Output layer for 7 emotion categories
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 7)
        """
        x = self.features(x)  # Extract features
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.classifier(x)  # Classify
        return x

def initialize_models():
    """
    Initialize MTCNN for face detection and VGG model for emotion recognition.
    
    Returns:
        tuple: (mtcnn, emotion_classifier, emotion_labels)
    """
    mtcnn = MTCNN(keep_all=True, device='cpu')  # Initialize MTCNN face detector
    emotion_classifier = VGG()  # Create VGG model
    emotion_classifier.load_state_dict(torch.load('./model/model_vgg.pkl', map_location=torch.device('cpu')))  # Load trained weights
    emotion_classifier.eval()  # Set to evaluation mode
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    return mtcnn, emotion_classifier, emotion_labels

def process_frame(frame, mtcnn, emotion_classifier, emotion_labels):
    """
    Process each video frame to detect faces and classify emotions.
    
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
            x, y, x2, y2 = box
            w = x2 - x
            h = y2 - y
            if w > 0 and h > 0:
                # Extract and preprocess face
                face = frame[int(y):int(y2), int(x):int(x2)]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_normalized = preprocess_input(np.array(face_resized, dtype=np.float32)).reshape(1, 1, 48, 48)
                face_tensor = torch.from_numpy(face_normalized).type(torch.FloatTensor)

                # Predict emotion
                with torch.no_grad():
                    emotion_pred = emotion_classifier(face_tensor)
                    emotion_arg = torch.argmax(emotion_pred).item()
                    emotion = emotion_labels[emotion_arg]
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (84, 255, 159), 2)
                    cv2.putText(frame, emotion, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
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
